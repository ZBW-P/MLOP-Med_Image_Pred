import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import zipfile
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import json
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
            nn.ReLU(),  # or GELU
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
        # Adjust image size accordingly before patch embedding
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
        self.save_hyperparameters(hparams)  # Saves hyperparameters for later checkpoint recovery
        # Initialize the ViT model with the provided hyperparameters
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

    def setup(self, stage):
        # Optionally, load a checkpoint or perform additional setup if needed.
        if self.trainer_ckpt_path:
            print("Setting up from checkpoint:", self.trainer_ckpt_path)
            # Example: self.load_state_dict(torch.load(self.trainer_ckpt_path))
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = F.log_softmax(self.model(images), dim=1)
        loss = F.nll_loss(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = F.log_softmax(self.model(images), dim=1)
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, dim=1)
        correct_samples = preds.eq(labels).sum().item()
        accuracy = correct_samples / len(labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        return {"val_loss": loss, "val_accuracy": accuracy}
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = F.log_softmax(self.model(images), dim=1)
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, dim=1)
        correct_samples = preds.eq(labels).sum().item()
        accuracy = correct_samples / len(labels)
        self.log("Test_loss", loss, prog_bar=True)
        self.log("Test_accuracy", accuracy, prog_bar=True)
        return {"Test_loss": loss, "Test_accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

def get_dataloaders(data_dir, batch_size, zip_file_path, num_workers=4):
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    transform_medical = transforms.Compose([
        transforms.Resize((64, 64)),
        # Convert grayscale images to 1 channels if needed:
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    archive_path = os.path.join(data_dir, 'chest_xray')

    full_dataset = datasets.ImageFolder(root=archive_path, transform=transform_medical)

    class_to_idx = full_dataset.class_to_idx
    with open('class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f)

    from sklearn.model_selection import train_test_split
    import numpy as np

    # Get all indices and their corresponding labels
    indices = np.arange(len(full_dataset))
    labels = [full_dataset.imgs[i][1] for i in indices]
    print(labels)

    train_val_idx, test_idx, train_val_labels, test_labels = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=42)

    train_idx, val_idx, train_labels, val_labels = train_test_split(
        train_val_idx, train_val_labels, test_size=0.125, stratify=train_val_labels, random_state=42)  

    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Set paths (update as needed)
    data_dir = r"C:\Users\26435\Desktop\FF-Machine_Learning\FF_OPML\Proj"
    zip_file_path = r"C:\Users\26435\Desktop\FF-Machine_Learning\FF_OPML\Proj\chest_xray.zip"
    batch_size = 32

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size, zip_file_path)
    
    # Define hyperparameters for the ViT model
    hparams = {
        "image_size": 64,
        "patch_size": 16,
        "num_classes": 9,
        "dim": 256,
        "depth": 16,
        "heads": 16,
        "mlp_dim": 1024,
        "channels": 1,        # Use 1 for grayscale; note transform converts to 3 channels if needed
        "dropout": 0.0,
        "emb_dropout": 0.0,
        "lr": 0.0001,
    }

    # Instantiate the LightningModule
    lit_vit = LitViT(hparams)
    print(count_parameters(lit_vit))

    # Create a ModelCheckpoint callback to automatically save the model parameters
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # metric to monitor
        save_top_k=1,
        mode="min",
        filename="vit-{epoch:02d}-{val_loss:.4f}"
    )

    # Initialize the Trainer with DDP strategy and checkpointing enabled.
    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        # strategy=DDPStrategy(),
        max_epochs=5,
        precision="bf16",
        # accumulate_grad_batches=8,
        callbacks=[checkpoint_callback]
    )

    # Train the model
    trainer.fit(lit_vit, train_dataloaders=train_loader, val_dataloaders=test_loader)
    
    best_model = LitViT.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save(best_model.state_dict(), "vit_model.pth")
    
    trainer.test(best_model, dataloaders=val_loader)

    # Print the best checkpoint path
    print("Best model saved at:", checkpoint_callback.best_model_path)
    