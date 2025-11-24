import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ======================================================================
# 1. U-NET ENCODER (FIXED - Paper-faithful implementation)
# ======================================================================

def down_block(in_c, out_c, normalize=True, apply_dropout=False):
    """
    Downsampling block from the paper:
    - Conv2d with stride 2 for downsampling
    - InstanceNorm2d (if normalize=True)
    - LeakyReLU(0.2)
    - Optional dropout
    """
    layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]

    if normalize:
        # Paper uses instance normalization (affine=False for exact behavior)
        layers.append(nn.InstanceNorm2d(out_c, affine=False, track_running_stats=False))

    layers.append(nn.LeakyReLU(0.2, inplace=True))

    if apply_dropout:
        layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)


class UNetEncoder(nn.Module):
    """
    U-Net Encoder from paper (Section 3.1):
    - Input: 256×256×3
    - 8 downsampling layers
    - Feature counts: 128, 256, 512, 512, 512, 512, 512, 512
    - Bottleneck: 1×1×512 spatial resolution
    """

    def __init__(self):
        super().__init__()
        # Paper specifies no normalization on first layer
        self.enc1 = down_block(3, 128, normalize=False)  # 256×256 → 128×128
        self.enc2 = down_block(128, 256)  # 128×128 → 64×64
        self.enc3 = down_block(256, 512)  # 64×64 → 32×32
        self.enc4 = down_block(512, 512)  # 32×32 → 16×16
        self.enc5 = down_block(512, 512)  # 16×16 → 8×8
        self.enc6 = down_block(512, 512)  # 8×8 → 4×4
        self.enc7 = down_block(512, 512)  # 4×4 → 2×2

        # FIX: Last layer (2×2 → 1×1) - NO normalization to avoid error
        # InstanceNorm2d requires spatial dimensions >= 2
        self.enc8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Paper mentions "single convolutional layer with 64 output feature channels"
        # This is the bottleneck before the decoder
        self.bottleneck = nn.Conv2d(512, 64, kernel_size=1)

    def forward(self, x):
        """
        Returns list of features at each encoder stage for analysis.
        """
        e1 = self.enc1(x)  # 128×128×128
        e2 = self.enc2(e1)  # 64×64×256
        e3 = self.enc3(e2)  # 32×32×512
        e4 = self.enc4(e3)  # 16×16×512
        e5 = self.enc5(e4)  # 8×8×512
        e6 = self.enc6(e5)  # 4×4×512
        e7 = self.enc7(e6)  # 2×2×512
        e8 = self.enc8(e7)  # 1×1×512

        bottleneck = self.bottleneck(e8)  # 1×1×64

        # Return all intermediate features for analysis
        return {
            'features': [e1, e2, e3, e4, e5, e6, e7, e8],
            'bottleneck': bottleneck
        }


# ======================================================================
# 2. FROZEN DINO ENCODER
# ======================================================================

class FrozenDINOEncoder(nn.Module):
    """
    Vision Transformer (DINO) encoder for comparison.
    Extracts multi-scale patch features from intermediate layers.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        DINO expects 224×224 input.
        Returns 4 intermediate feature maps from different transformer layers.
        """
        # Resize to DINO's expected input size
        x_resized = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Get intermediate layers (4 stages for comparison with U-Net)
        feats = self.model.get_intermediate_layers(x_resized, n=4)

        maps = []
        for tok in feats:
            # Remove CLS token, keep only patch tokens
            tok = tok[:, 1:, :]  # (B, num_patches, dim)
            B, N, C = tok.shape

            # Reshape to spatial feature map
            # DINO ViT-S/16 with 224×224 input → 14×14 patches
            S = int(N ** 0.5)
            fmap = tok.transpose(1, 2).reshape(B, C, S, S)
            maps.append(fmap)

        return maps


# ======================================================================
# 3. IMAGE LOADING WITH PAPER'S LOG TRANSFORM
# ======================================================================

def load_png_as_tensor(path):
    """
    Load and preprocess image following paper's approach:
    1. Resize to 256×256
    2. Convert to [0,1] float
    3. Apply log-intensity transform (Section 3.1)
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)

    # Convert to float tensor
    arr = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, C)

    # Paper's log transform: log(x + 0.01) - log(0.01) / (log(1.01) - log(0.01))
    eps = 0.01
    arr = (torch.log(arr + eps) - np.log(eps)) / (np.log(1.01) - np.log(eps))
    arr = torch.clamp(arr, 0, 1)  # Ensure [0, 1] range

    return arr.permute(2, 0, 1)  # (C, H, W)


# ======================================================================
# 4. FEATURE COMPARISON UTILITIES
# ======================================================================

def resize_for_compare(a, b, size=32):
    """
    Resize both tensors to same spatial size for comparison.
    Returns tensors with shape (C, size, size).
    """
    if a.dim() == 3:  # (C, H, W)
        a = F.interpolate(a.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
    else:  # Already has batch dim
        a = F.interpolate(a, size=(size, size), mode="bilinear", align_corners=False)
        if a.dim() == 4:
            a = a[0]

    if b.dim() == 3:  # (C, H, W)
        b = F.interpolate(b.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
    else:  # Already has batch dim
        b = F.interpolate(b, size=(size, size), mode="bilinear", align_corners=False)
        if b.dim() == 4:
            b = b[0]

    return a, b


def global_cosine(a, b):
    """
    Compute global cosine similarity between flattened features.
    Handles different channel dimensions by averaging across channels first.
    """
    # Ensure both tensors have same spatial dimensions (should already be done)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"Spatial dimensions must match: {a.shape} vs {b.shape}")

    # Average across channels to get spatial activation maps
    a_spatial = a.mean(dim=0)  # (H, W)
    b_spatial = b.mean(dim=0)  # (H, W)

    spatial_sim = F.cosine_similarity(a_spatial.flatten(), b_spatial.flatten(), dim=0).item()

    return spatial_sim


def channel_wise_cosine(a, b):
    """
    Compute average cosine similarity across channels.
    Compares spatial patterns for each channel dimension.
    """
    # Ensure spatial dimensions match
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"Spatial dimensions must match: {a.shape} vs {b.shape}")

    # Flatten spatial dimensions: (C, H, W) -> (C, H*W)
    a_flat = a.reshape(a.size(0), -1)  # (C_a, H*W)
    b_flat = b.reshape(b.size(0), -1)  # (C_b, H*W)

    # Take minimum channels and compute similarity
    min_channels = min(a_flat.size(0), b_flat.size(0))
    a_subset = a_flat[:min_channels]  # (min_C, H*W)
    b_subset = b_flat[:min_channels]  # (min_C, H*W)

    # Compute cosine similarity for each channel, then average
    # We compare spatial patterns within each channel
    cos_per_channel = []
    for i in range(min_channels):
        cos = F.cosine_similarity(a_subset[i:i + 1], b_subset[i:i + 1], dim=1).item()
        cos_per_channel.append(cos)

    return np.mean(cos_per_channel)


def pixelwise_cosine(a, b):
    """
    Compute per-pixel cosine similarity across channels.
    Handles different channel dimensions by taking the minimum.
    """
    # Take minimum channels to ensure same dimension
    min_channels = min(a.size(0), b.size(0))
    a_subset = a[:min_channels]  # (min_C, H, W)
    b_subset = b[:min_channels]  # (min_C, H, W)

    # Compute cosine similarity per pixel across channels
    return F.cosine_similarity(a_subset, b_subset, dim=0).cpu().numpy()


def channel_statistics(tensor, name):
    """Print statistics about a feature tensor."""
    print(f"\n{name} Statistics:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Mean: {tensor.mean().item():.4f}")
    print(f"  Std: {tensor.std().item():.4f}")
    print(f"  Min: {tensor.min().item():.4f}")
    print(f"  Max: {tensor.max().item():.4f}")


def visualize_feature_map(tensor, title, save_path=None):
    """Visualize mean activation across channels."""
    img = tensor.mean(dim=0).cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.title(title, fontsize=14)
    plt.imshow(img, cmap="viridis")
    plt.colorbar()
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def visualize_cosine_heatmap(cos_map, title, save_path=None):
    """Visualize spatial cosine similarity."""
    plt.figure(figsize=(6, 6))
    plt.title(title, fontsize=14)
    im = plt.imshow(cos_map, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, label="Cosine Similarity")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def compare_encoder_outputs(unet_feats, dino_feats, stage_pairs, img_name):
    """
    Compare U-Net and DINO features at specified stages.

    Args:
        unet_feats: List of U-Net feature tensors
        dino_feats: List of DINO feature tensors
        stage_pairs: List of (unet_idx, dino_idx) pairs to compare
        img_name: Name for saving visualizations
    """
    print(f"\n{'=' * 60}")
    print(f"Comparing features for: {img_name}")
    print(f"{'=' * 60}")

    os.makedirs("comparison_vis", exist_ok=True)

    results = []
    for unet_idx, dino_idx in stage_pairs:
        un = unet_feats[unet_idx][0]  # Remove batch dimension
        di = dino_feats[dino_idx][0]

        print(f"\nU-Net Stage {unet_idx + 1} vs DINO Layer {dino_idx + 1}")
        channel_statistics(un, f"U-Net[{unet_idx + 1}]")
        channel_statistics(di, f"DINO[{dino_idx + 1}]")

        # Resize to common size for comparison
        un_r, di_r = resize_for_compare(un, di, size=32)

        # Global similarity (spatial structure)
        sim_spatial = global_cosine(un_r, di_r)
        sim_channel = channel_wise_cosine(un_r, di_r)

        print(f"  → Spatial Cosine Similarity: {sim_spatial:.4f}")
        print(f"  → Channel Cosine Similarity: {sim_channel:.4f}")
        results.append((unet_idx + 1, dino_idx + 1, sim_spatial, sim_channel))

        # Pixelwise similarity heatmap
        heat = pixelwise_cosine(un_r, di_r)

        # Visualize
        base = img_name.replace('.png', '')
        visualize_cosine_heatmap(
            heat,
            f"U-Net[{unet_idx + 1}] vs DINO[{dino_idx + 1}]: Spatial={sim_spatial:.3f}, Channel={sim_channel:.3f}",
            save_path=f"comparison_vis/{base}_unet{unet_idx + 1}_dino{dino_idx + 1}.png"
        )

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary of Similarities:")
    print(f"{'=' * 60}")
    for u, d, s_spatial, s_channel in results:
        print(f"  U-Net[{u}] ↔ DINO[{d}]: Spatial={s_spatial:.4f}, Channel={s_channel:.4f}")

    return results


# ======================================================================
# 5. MAIN PIPELINE
# ======================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # ---------------------------------------------------------------
    # Load models in eval mode
    # ---------------------------------------------------------------
    print("Initializing U-Net encoder...")
    unet = UNetEncoder().to(device)
    unet.eval()

    print("Loading DINO ViT-S/16...")
    dino_model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
    dino = FrozenDINOEncoder(dino_model).to(device)
    dino.eval()

    # ---------------------------------------------------------------
    # Process all images
    # ---------------------------------------------------------------
    IMG_DIR = "inputs"
    if not os.path.exists(IMG_DIR):
        print(f"Error: '{IMG_DIR}' directory not found!")
        exit(1)

    images = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")])
    if not images:
        print(f"No PNG files found in '{IMG_DIR}/'")
        exit(1)

    print(f"Found {len(images)} images in '{IMG_DIR}/'")

    # Create output directories
    os.makedirs("encoder_outputs/unet", exist_ok=True)
    os.makedirs("encoder_outputs/dino", exist_ok=True)

    # Extract and save features
    print("\nExtracting features from all images...\n")
    for fname in tqdm(images, desc="Processing"):
        img_path = os.path.join(IMG_DIR, fname)
        img = load_png_as_tensor(img_path).unsqueeze(0).to(device)

        with torch.no_grad():
            unet_output = unet(img)
            dino_feats = dino(img)

        base = fname.replace(".png", "")
        torch.save(unet_output, f"encoder_outputs/unet/{base}_unet.pt")
        torch.save(dino_feats, f"encoder_outputs/dino/{base}_dino.pt")

    print("\n✓ Feature extraction completed!\n")

    # ---------------------------------------------------------------
    # Detailed comparison on first image
    # ---------------------------------------------------------------
    print("=" * 60)
    print("DETAILED COMPARISON (First Image)")
    print("=" * 60)

    sample_img = images[0]
    base = sample_img.replace(".png", "")

    unet_output = torch.load(f"encoder_outputs/unet/{base}_unet.pt")
    dino_feats = torch.load(f"encoder_outputs/dino/{base}_dino.pt")

    # U-Net has 8 encoder stages, DINO has 4 intermediate layers
    # Compare at corresponding scales
    stage_pairs = [
        (2, 0),  # U-Net stage 3 (32×32) vs DINO layer 1 (14×14)
        (4, 1),  # U-Net stage 5 (8×8) vs DINO layer 2 (14×14)
        (5, 2),  # U-Net stage 6 (4×4) vs DINO layer 3 (14×14)
        (6, 3),  # U-Net stage 7 (2×2) vs DINO layer 4 (14×14)
    ]

    compare_encoder_outputs(
        unet_output['features'],
        dino_feats,
        stage_pairs,
        sample_img
    )

    print("\n" + "=" * 60)
    print("All processing complete!")
    print("=" * 60)
    print(f"- Features saved in: encoder_outputs/")
    print(f"- Visualizations saved in: comparison_vis/")
    print("=" * 60)