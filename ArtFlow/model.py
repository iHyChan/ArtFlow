import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adain(content_feat, style_feat):
    B, C, H, W = content_feat.size()
    c_mean = content_feat.view(B, C, -1).mean(2).view(B, C, 1, 1)
    c_std  = content_feat.view(B, C, -1).std(2).view(B, C, 1, 1) + 1e-5
    s_mean = style_feat.view(B, C, -1).mean(2).view(B, C, 1, 1)
    s_std  = style_feat.view(B, C, -1).std(2).view(B, C, 1, 1) + 1e-5
    return (content_feat - c_mean) / c_std * s_std + s_mean


class VGGFeature(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4


def gram(x):
    b, c, h, w = x.size()
    f = x.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def load_image(path, size=256):
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0).to(device)


def tensor_to_pil(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    img  = (t.squeeze(0) * std + mean).clamp(0, 1)
    arr  = (img.cpu().permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def run_transfer(content_path, style_path,
                 alpha=0.8, size=256, steps=100,
                 style_w=1e6, content_w=1.0,
                 callback=None):
    vgg = VGGFeature().to(device)
    content = load_image(content_path, size)
    style   = load_image(style_path,   size)

    with torch.no_grad():
        cf1, cf2, cf3, cf4 = vgg(content)
        sf1, sf2, sf3, sf4 = vgg(style)

        t4 = adain(cf4, sf4)
        t3 = adain(cf3, sf3)
        t2 = adain(cf2, sf2)
        t1 = adain(cf1, sf1)

        target4 = (alpha * t4 + (1 - alpha) * cf4).detach()
        target3 = (alpha * t3 + (1 - alpha) * cf3).detach()
        target2 = (alpha * t2 + (1 - alpha) * cf2).detach()
        target1 = (alpha * t1 + (1 - alpha) * cf1).detach()

        gram_t4 = gram(target4)
        gram_t3 = gram(target3)
        gram_t2 = gram(target2)
        gram_t1 = gram(target1)

    opt_img   = content.clone().requires_grad_(True)
    optimizer = optim.Adam([opt_img], lr=0.02)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        f1, f2, f3, f4 = vgg(opt_img)

        c_loss = F.mse_loss(f4, target4)
        s_loss = (F.mse_loss(gram(f4), gram_t4) +
                  F.mse_loss(gram(f3), gram_t3) +
                  F.mse_loss(gram(f2), gram_t2) +
                  F.mse_loss(gram(f1), gram_t1))

        loss = content_w * c_loss + style_w * s_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            opt_img.clamp_(-2.5, 2.5)

        if callback:
            pct     = int(step / steps * 100)
            preview = None
            if step % 10 == 0 or step == steps:
                preview = tensor_to_pil(opt_img.detach())
            callback(step, steps, float(loss.item()), pct, preview)

    result = tensor_to_pil(opt_img.detach())
    os.makedirs("results", exist_ok=True)
    ts        = int(time.time())
    save_path = f"results/result_{ts}.jpg"
    result.save(save_path, quality=95)
    return result, save_path
