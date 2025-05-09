import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import json
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
# --- Define the model architecture ---
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
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
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

transform_medical = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# --- Helper to load the ViT model ---
def get_vit_model():
    # 1) instantiate exactly as in training
    model = ViT(
        image_size=64,
        patch_size=16,
        num_classes=9,
        channels=1,
        dim=64,
        depth=16,
        heads=16,
        mlp_dim=256,
        emb_dropout=0.0,
        dropout=0.0
    )

    # 2) load the checkpoint dict
    ckpt = torch.load("/home/jovyan/work/MLOP-Med_Image_Pred/latest_checkpoint.pth", map_location="cpu")

    # 3) pull out just the weights
    state_dict = ckpt.get("state_dict", ckpt)

    # 4) strip the Lightning prefix
    clean_state_dict = {
        k.replace("model.", ""): v
        for k, v in state_dict.items()
    }

    # 5) load into your raw ViT
    model.load_state_dict(clean_state_dict, strict=True)

    model.eval()
    return model

def predict_image(image_path, model, transform, class_names):
    image = Image.open(image_path).convert("L")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = F.log_softmax(model(image_tensor), dim=1)
        _, pred_idx = torch.max(output, dim=1)
    predicted_class = class_names[pred_idx.item()]
    print("Predicted class:", predicted_class,"label:",pred_idx)
    return pred_idx
    
    
import sys

# --- Main script: run prediction ---
transform_medical = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

import torchvision.datasets as datasets

def get_class_mapping(data_root):
    dataset = datasets.ImageFolder(root=data_root, transform=transform_medical)
    return dataset.class_to_idx

if __name__ == "__main__":
    train_dir = "/mnt/object/train"
    test_dir  = "/mnt/object/test"
    val_dir   = "/mnt/object/val"
    image_path="/mnt/object/train/lung-covid/COVID19_Radiography-lung-covid-COVID-1.png"

    # you only need the mapping:
    class_to_idx = get_class_mapping(train_dir)

    class_names = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        class_names[idx] = name

    print("Class names:", class_names)

    vit_model = get_vit_model()
    pred_idx = predict_image(image_path, vit_model, transform_medical, class_names)
    print("Predicted class index:", pred_idx)
