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

def get_dataloaders(data_dir, batch_size, file_path, num_workers=16):
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
    mlflow.set_tracking_uri("http://192.5.87.147:8000")
    mlflow.set_experiment("classifier")
    mlflow.pytorch.autolog()
    try:
        mlflow.end_run()
    except:
        pass
    mlflow.start_run(log_system_metrics=True)
    mlflow.log_params(hparams)
    try:
        info = subprocess.check_output("nvidia-smi", shell=True).decode()
        mlflow.log_text(info, "gpu-info.txt")
    except:
        pass

@rank_zero_only
def log_test_metrics(trainer, model, test_loader):
    trainer.test(model, dataloaders=test_loader)
    metrics = trainer.callback_metrics
    mlflow.log_metrics({
        "test_loss":     metrics.get("test_loss", 0.0),
        "test_accuracy": metrics.get("test_accuracy", 0.0),
    })
    mlflow.end_run()
    
if __name__ == '__main__':
    data_dir = "/home/jovyan/work/MLOP-Med_Image_Pred"
    zip_file_path = "/mnt/object2/merged_dataset"
    batch_size = 32
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size, zip_file_path)
    hparams = {
        "image_size": 64,
        "patch_size": 16,
        "num_classes": 9,
        "dim": 64,
        "depth": 8,
        "heads": 16,
        "mlp_dim": 256,
        "channels": 1,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        "lr": 0.0001,
    }
    
    # fsdp_strategy = FSDPStrategy(
    #     sharding_strategy=ShardingStrategy.FULL_SHARD,
    #     auto_wrap_policy=None,
    #     cpu_offload=False,
    # )
    

    lit_vit = LitViT(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(data_dir, "checkpoints"),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="vit-{epoch:02d}-{val_loss:.4f}"
    )

    trainer = Trainer(
        devices=4,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        # strategy=fsdp_strategy,
        max_epochs=1,
        precision="bf16",
        # amp_backend="native",
        callbacks=[checkpoint_callback]
    )
    if trainer.global_rank==0:
        setup_mlflow(hparams)

    # Train
    trainer.fit(lit_vit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.global_rank==0:
        best_ckpt = checkpoint_callback.best_model_path
        print("Loaded checkpoint:", best_ckpt)
        best_model = LitViT.load_from_checkpoint(best_ckpt)
        torch.save(best_model.state_dict(), os.path.join(data_dir, "vit_model.pth"))
        log_test_metrics(trainer, best_model, test_loader)
        mlflow.end_run()
    else:
        os._exit(0)
