import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data.distributed import DistributedSampler
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
import ray.train.lightning as rlt
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, FailureConfig
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, RunConfig
from ray.air.config import CheckpointConfig, RunConfig
# ------------------ Ray train_func ------------------
def train_func(config):
    data_dir = os.getenv("DATA_PATH", "/mnt/object/")
    batch_size = config["batch_size"]
    hparams = config["hparams"]

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'merged_dataset', 'train'), transform=tf)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'merged_dataset', 'val'), transform=tf)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'merged_dataset', 'test'), transform=tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=DistributedSampler(train_dataset), num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def pair(t): return t if isinstance(t, tuple) else (t, t)

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
                nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
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
            self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
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
                ]) for _ in range(depth)
            ])
        def forward(self, x):
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return x

    class ViT(nn.Module):
        def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                     pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0., conv=64, conv1=128):
            super().__init__()
            self.conv_stem = nn.Sequential(
                nn.Conv2d(channels, conv, 3, 1, 1), nn.BatchNorm2d(conv), nn.ReLU(),
                nn.Conv2d(conv, conv, 3, 1, 1), nn.BatchNorm2d(conv), nn.ReLU(),
                nn.Conv2d(conv, conv1, 3, 2, 1), nn.BatchNorm2d(conv1), nn.ReLU(),
                nn.Conv2d(conv1, conv1, 3, 2, 1), nn.BatchNorm2d(conv1), nn.ReLU()
            )
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c h w -> b (h w) c'),
                nn.Linear(conv1, dim),
            )
            h, w = pair(image_size)
            ph, pw = h // 4, w // 4
            num_patches = ph * pw
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.dropout = nn.Dropout(emb_dropout)
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
            self.pool = pool
            self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        def forward(self, img):
            x = self.conv_stem(img)
            x = self.to_patch_embedding(x)
            b, n, _ = x.shape
            cls = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls, x), dim=1)
            x = x + self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            return self.mlp_head(x)

    class LitViT(L.LightningModule):
        def __init__(self, hparams):
            super().__init__()
            self.save_hyperparameters(hparams)
            # self.model = ViT(**hparams)
            model_args = {k: v for k, v in hparams.items() if k not in ["lr", "max_epochs"]}
            self.model = ViT(**model_args)
        def forward(self, x): return self.model(x)
        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean()
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_acc", acc, prog_bar=True)
            return loss
        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean()
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)
        def test_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean()
            self.log("test_loss", loss, prog_bar=True)
            self.log("test_acc", acc, prog_bar=True)
        def configure_optimizers(self):
            return optim.Adam(self.parameters(), lr=self.hparams["lr"])

    lit = LitViT(hparams)

    trainer = Trainer(
        strategy=RayDDPStrategy(),
        enable_checkpointing=False,
        plugins=[RayLightningEnvironment()],
        max_epochs=hparams["max_epochs"],
        devices="auto",
        accelerator="gpu",
        precision="bf16-mixed"
    )
    trainer = rlt.prepare_trainer(trainer)

    trainer.fit(lit, train_loader, val_loader)
    trainer.test(lit, dataloaders=test_loader)

# ------------------ Main: submit TorchTrainer ------------------
run_config = RunConfig(
  storage_path="s3://ray", 
  checkpoint_config=CheckpointConfig(
    checkpoint_score_attribute="val_loss",   # which metric to rank on
    num_to_keep=3,                           # keep only the best 3
  )
)
scaling_config = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 8})
train_loop_config = {
    "batch_size": 32,
    "hparams": {
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
        "lr": 1e-4,
        "max_epochs": 1,
    },
}

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config,
    torch_config=TorchConfig(backend="gloo"),
    run_config=run_config,
    train_loop_config=train_loop_config
)
result = trainer.fit()
