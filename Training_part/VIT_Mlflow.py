import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import mlflow
import mlflow.pytorch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp import ShardingStrategy
import random
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ])
            for _ in range(depth)
        ])
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0., conv=64,conv1=128,**kwargs):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=conv, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv),
            nn.ReLU(),
            nn.Conv2d(conv, conv, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv, out_channels=conv1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv1),
            nn.ReLU(),
            nn.Conv2d(conv1, conv1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv1),
            nn.ReLU()
        )

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(conv1, dim),
        )
        image_height, image_width = pair(image_size)
        new_image_height = image_height // 4
        new_image_width = image_width // 4
        num_patches = new_image_height * new_image_width
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, img):
        x = self.conv_stem(img)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

class LitViT(L.LightningModule):
    def __init__(self, hparams, trainer_ckpt_path=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = ViT(
            image_size=self.hparams.image_size,
            patch_size=self.hparams.patch_size,
            num_classes=self.hparams.num_classes,
            dim=self.hparams.dim,
            depth=self.hparams.depth,
            heads=self.hparams.heads,
            mlp_dim=self.hparams.mlp_dim,
            channels=self.hparams.channels,
            dropout=self.hparams.dropout,
            emb_dropout=self.hparams.emb_dropout
        )
        self.trainer_ckpt_path = trainer_ckpt_path

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = F.log_softmax(self.model(images), dim=1)
        loss = F.nll_loss(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = F.log_softmax(self.model(images), dim=1)
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, dim=1)
        accuracy = preds.eq(labels).sum().item() / len(labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = F.log_softmax(self.model(images), dim=1)
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, dim=1)
        accuracy = preds.eq(labels).sum().item() / len(labels)
        self.log("Test_loss", loss, prog_bar=True)
        self.log("Test_accuracy", accuracy, prog_bar=True)
        return {"Test_loss": loss, "Test_accuracy": accuracy}

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.hparams.lr)

def get_dataloaders(batch_size, file_path, num_workers=16):
    transform_medical = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_path = os.path.join(file_path, 'train')
    Test_path = os.path.join(file_path, 'test')
    valid_path = os.path.join(file_path, 'val')
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform_medical)
    Test_dataset = datasets.ImageFolder(root=Test_path, transform=transform_medical)
    valid_dataset = datasets.ImageFolder(root=valid_path, transform=transform_medical)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
@rank_zero_only
def setup_mlflow(hparams):
    mlflow.set_tracking_uri("http://129.114.27.23:8000")
    mlflow.set_experiment("classifier")
    mlflow.autolog(log_models=False)
    try:
        mlflow.end_run()
    except:
        pass
    # mlflow.start_run(log_system_metrics=True)
    finally:
        mlflow.start_run(log_system_metrics=True)
        mlflow.log_params(hparams)
    try:
        info = subprocess.check_output("rocm-smi", shell=True).decode()
        mlflow.log_text(info, "gpu-info.txt")
    except:
        pass
    
def add(train_loader, val_loader, test_loader, batch_size, file_path, num_workers: int =16, ratio: float=0.1 ,seed: int = 42):
    
    file = input("Enter file name you want to add the list is [lung-covid   lung-oct-cnv  lung-oct-drusen  lung-opacity lung-viral-pneumonia lung-normal  lung-oct-dme  lung-oct-normal  lung-tuberculosis] ").strip()
    offset = int(input("Enter number of times iteration for select object [0-9]: "))
    
    transform_medical = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    New_path = os.path.join(file_path, 'final_eval', file)
    new_dataset = datasets.ImageFolder(root=New_path, transform=transform_medical)
    
    class_size=int(ratio*len(new_dataset))
    start_idx =offset*class_size
    if offset == 9:
        end_idx=len(new_dataset)
    else:
        end_idx = start_idx +class_size
    
    if end_idx > len(new_dataset):
        raise ValueError(
            f"would exceed dataset size {end_idx}"
        )
        
    indices = list(range(start_idx,end_idx))
        
    rnd = random.Random(seed + offset)
    rnd.shuffle(indices)
    
    n_train = int(0.7 * class_size)
    n_val   = int(0.2 * class_size)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]
    
    new_train = torch.utils.data.Subset(new_dataset, train_idx)
    new_val = torch.utils.data.Subset(new_dataset, val_idx)
    new_test  = torch.utils.data.Subset(new_dataset, test_idx)
    
    updated_train = torch.utils.data.ConcatDataset([train_loader.dataset, new_train])
    updated_val = torch.utils.data.ConcatDataset([val_loader.dataset, new_val])
    updated_test = torch.utils.data.ConcatDataset([test_loader.dataset, new_test])
    
    train_loader = DataLoader(updated_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(updated_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(updated_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    add_if=False
    data_dir = "/home/jovyan/work/MLOP-Med_Image_Pred"
    zip_file_path = "/mnt/object"
    batch_size = 32
    train_loader, val_loader, test_loader = get_dataloaders(batch_size, zip_file_path)
    
    if add_if:
        train_loader, val_loader, test_loader=add(train_loader, val_loader, test_loader,batch_size, zip_file_path)
        
    
    hparams = {
        "image_size": 64,
        "patch_size": 16,
        "num_classes": 9,
        "dim": 64,
        "depth": 16,
        "heads": 16,
        "mlp_dim": 256,
        "channels": 1,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        "lr": 0.0001,
    }

    lit_vit = LitViT(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(data_dir, "checkpoints"),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="vit-{epoch:02d}-{val_loss:.4f}"
    )

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        # strategy=fsdp_strategy,
        max_epochs=12,
        precision="bf16-mixed",
        # amp_backend="native",
        callbacks=[checkpoint_callback]
    )
    if trainer.global_rank==0:
        setup_mlflow(hparams)

    # Train
    trainer.fit(lit_vit, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_vit, dataloaders=test_loader)

    if trainer.global_rank==0:
        best_ckpt = checkpoint_callback.best_model_path
        print("Loaded checkpoint:", best_ckpt)
        best_model = LitViT.load_from_checkpoint(best_ckpt)
        torch.save(best_model.state_dict(), os.path.join(data_dir, "vit_model.pth"))
        
        mlflow.end_run()
