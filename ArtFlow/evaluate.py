import os
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from model import run_transfer, device

STYLE_PRESETS = {
    "Starry Night":       "style_images/starry_night.jpg",
    "Impression Sunrise": "style_images/impression_sunrise.jpg",
    "The Scream":         "style_images/the_scream.jpg",
    "The Great Wave":     "style_images/the_wave.jpg",
}

os.makedirs("results",     exist_ok=True)
os.makedirs("eval_output", exist_ok=True)


def get_gram_feats(path, size=256):
    vgg = models.vgg19(
        weights=models.VGG19_Weights.IMAGENET1K_V1
    ).features.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    x     = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    feats = []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in [3, 8, 17, 26]:
            b, c, h, w = x.size()
            f = x.view(b, c, h * w)
            feats.append(
                torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
            )
    return feats


def compute_gram_loss(style_path, result_path):
    sg = get_gram_feats(style_path)
    rg = get_gram_feats(result_path)
    return sum(F.mse_loss(r, s).item() for r, s in zip(rg, sg))


def compute_ssim(p1, p2, size=256):
    a = np.array(Image.open(p1).convert("RGB").resize((size, size)))
    b = np.array(Image.open(p2).convert("RGB").resize((size, size)))
    return ssim(a, b, channel_axis=2, data_range=255)


def experiment_styles(content_path):
    print("\n===== Experiment 1: Four Style Comparison =====")
    style_names, gram_losses, ssim_scores = [], [], []
    content_imgs, style_imgs, result_imgs = [], [], []

    for name, spath in STYLE_PRESETS.items():
        print(f"  Transferring style: {name} ...")
        result, rpath = run_transfer(content_path, spath,
                                     alpha=0.8, size=256, steps=100)
        gl = compute_gram_loss(spath, rpath)
        ss = compute_ssim(content_path, rpath)
        style_names.append(name)
        gram_losses.append(gl)
        ssim_scores.append(ss)
        content_imgs.append(Image.open(content_path).convert("RGB"))
        style_imgs.append(Image.open(spath).convert("RGB"))
        result_imgs.append(result)
        print(f"    Gram Loss: {gl:.6f}   SSIM: {ss:.4f}")

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    row_labels = ["Content", "Style", "Result"]
    for j in range(4):
        for r, imgs in enumerate([content_imgs, style_imgs, result_imgs]):
            axes[r][j].imshow(imgs[j])
            axes[r][j].axis("off")
            if r == 0:
                axes[r][j].set_title(style_names[j], fontsize=9, color="white")
            if r == 2:
                axes[r][j].set_xlabel(
                    f"Gram: {gram_losses[j]:.4f}\nSSIM: {ssim_scores[j]:.4f}",
                    fontsize=8, color="white")
        axes[r][0].set_ylabel(row_labels[r], fontsize=9, color="white")

    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#0d1117")
    plt.suptitle("Experiment 1: Style Transfer Comparison (α=0.8, steps=100)",
                 color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("eval_output/exp1_style_comparison.png",
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print("  Saved: eval_output/exp1_style_comparison.png")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.patch.set_facecolor("#1a1a2e")
    ax2.set_facecolor("#16213e")
    bars = ax2.bar(style_names, gram_losses,
                   color=["#4cc9f0", "#7209b7", "#06d6a0", "#ef233c"],
                   width=0.5)
    ax2.set_title("Gram Loss per Style (lower is better)",
                  color="white", fontsize=11)
    ax2.set_ylabel("Gram Loss", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#0f3460")
    for bar, val in zip(bars, gram_losses):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.01,
                 f"{val:.4f}", ha="center", va="bottom",
                 color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig("eval_output/exp1_gram_loss_bar.png",
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print("  Saved: eval_output/exp1_gram_loss_bar.png")

    return style_names, gram_losses, ssim_scores


def experiment_alpha(content_path, style_path, style_name="Starry Night"):
    print("\n===== Experiment 2: Alpha Ablation =====")
    alphas     = [0.2, 0.4, 0.6, 0.8, 1.0]
    gram_vals  = []
    ssim_vals  = []
    result_imgs = []

    for a in alphas:
        print(f"  alpha={a} ...")
        result, rpath = run_transfer(content_path, style_path,
                                     alpha=a, size=256, steps=100)
        gram_vals.append(compute_gram_loss(style_path, rpath))
        ssim_vals.append(compute_ssim(content_path, rpath))
        result_imgs.append(result)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for j, (a, img) in enumerate(zip(alphas, result_imgs)):
        axes[j].imshow(img)
        axes[j].set_title(
            f"α = {a}\nGram: {gram_vals[j]:.4f}\nSSIM: {ssim_vals[j]:.4f}",
            fontsize=8, color="white")
        axes[j].axis("off")
        axes[j].set_facecolor("#0d1117")
    plt.suptitle(f"Experiment 2: Alpha Ablation — Style: {style_name}",
                 color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig("eval_output/exp2_alpha_ablation.png",
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print("  Saved: eval_output/exp2_alpha_ablation.png")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.patch.set_facecolor("#1a1a2e")
    ax2.set_facecolor("#16213e")
    ax2.plot(alphas, gram_vals, "o-", color="#4cc9f0", label="Gram Loss")
    ax2_r = ax2.twinx()
    ax2_r.plot(alphas, ssim_vals, "s--", color="#06d6a0", label="SSIM")
    ax2.set_xlabel("Alpha", color="white")
    ax2.set_ylabel("Gram Loss", color="#4cc9f0")
    ax2_r.set_ylabel("SSIM", color="#06d6a0")
    ax2.tick_params(colors="white")
    ax2_r.tick_params(colors="white")
    ax2.set_title("Alpha vs Gram Loss & SSIM", color="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#0f3460")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#16213e", labelcolor="white")
    plt.tight_layout()
    plt.savefig("eval_output/exp2_alpha_curve.png",
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print("  Saved: eval_output/exp2_alpha_curve.png")


def experiment_steps(content_path, style_path, style_name="Starry Night"):
    print("\n===== Experiment 3: Iteration Steps Ablation =====")
    steps_list  = [50, 100, 150, 200, 300]
    gram_vals   = []
    ssim_vals   = []
    result_imgs = []

    for s in steps_list:
        print(f"  steps={s} ...")
        result, rpath = run_transfer(content_path, style_path,
                                     alpha=0.8, size=256, steps=s)
        gram_vals.append(compute_gram_loss(style_path, rpath))
        ssim_vals.append(compute_ssim(content_path, rpath))
        result_imgs.append(result)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for j, (s, img) in enumerate(zip(steps_list, result_imgs)):
        axes[j].imshow(img)
        axes[j].set_title(
            f"steps = {s}\nGram: {gram_vals[j]:.4f}\nSSIM: {ssim_vals[j]:.4f}",
            fontsize=8, color="white")
        axes[j].axis("off")
    plt.suptitle(f"Experiment 3: Steps Ablation — Style: {style_name}",
                 color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig("eval_output/exp3_steps_ablation.png",
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print("  Saved: eval_output/exp3_steps_ablation.png")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.patch.set_facecolor("#1a1a2e")
    ax2.set_facecolor("#16213e")
    ax2.plot(steps_list, gram_vals, "o-", color="#4cc9f0", label="Gram Loss")
    ax2_r = ax2.twinx()
    ax2_r.plot(steps_list, ssim_vals, "s--", color="#06d6a0", label="SSIM")
    ax2.set_xlabel("Iteration Steps", color="white")
    ax2.set_ylabel("Gram Loss", color="#4cc9f0")
    ax2_r.set_ylabel("SSIM", color="#06d6a0")
    ax2.tick_params(colors="white")
    ax2_r.tick_params(colors="white")
    ax2.set_title("Steps vs Gram Loss & SSIM", color="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#0f3460")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#16213e", labelcolor="white")
    plt.tight_layout()
    plt.savefig("eval_output/exp3_steps_curve.png",
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print("  Saved: eval_output/exp3_steps_curve.png")


def print_summary_table(style_names, gram_losses, ssim_scores):
    print("\n===== Summary Table =====")
    print(f"{'Style':<28} {'Gram Loss':>12} {'SSIM':>8}")
    print("-" * 52)
    for n, g, s in zip(style_names, gram_losses, ssim_scores):
        print(f"{n:<28} {g:>12.6f} {s:>8.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <content_image_path>")
        print("Example: python evaluate.py test_content.jpg")
        sys.exit(1)

    content = sys.argv[1]
    if not os.path.exists(content):
        print(f"Content image not found: {content}")
        sys.exit(1)

    sn, gl, ss = experiment_styles(content)
    print_summary_table(sn, gl, ss)

    experiment_alpha(content,
                     "style_images/starry_night.jpg",
                     "Starry Night")

    experiment_steps(content,
                     "style_images/starry_night.jpg",
                     "Starry Night")

    print("\nAll experiments complete. Results saved in eval_output/")
