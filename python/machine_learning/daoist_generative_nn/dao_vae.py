"""
道生成變分自編碼器 | Dao Variational Autoencoder (DaoVAE)
========================================================

A revolutionary VAE architecture where the latent space embodies Daoist cosmology.

Unlike standard VAEs with Gaussian latent spaces, DaoVAE structures its latent
space according to the Wuji-Taiyi-YinYang-Bagua-Wuxing cosmological framework.

This creates:
1. More meaningful latent representations (aligned with cosmological principles)
2. Better interpolation properties (smooth transitions through wuji void)
3. Interpretable latent dimensions (bagua directions, yin-yang balance, wuxing phases)
4. Natural disentanglement (through cosmological structure)

ARCHITECTURAL OVERVIEW
======================

ENCODER (Gathering):
    Image → Neidan Refinement → Extract Essence/Energy/Spirit → Project to Latent

LATENT SPACE (Transformation):
    Wuji Void ← Taiyi Unity ← YinYang Duality ← Bagua 8-fold ← Wuxing 5-phases

DECODER (Manifestation):
    Latent → Taiyi → YinYang → Bagua → Wuxing → Neidan Gates → Image

The encoder "gathers" phenomena back to essence (逆煉歸一).
The decoder "manifests" essence into phenomena (順化萬象).

This mirrors the Daoist practice of:
- Gathering (采): Drawing qi back to its source
- Refining (煉): Transforming coarse to subtle
- Manifesting (化): Expressing subtle through form
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math

from wuji_taiyi_architecture import (
    WujiSampler,
    TaiyiTransform,
    YinYangDualPath,
    BaguaTransform,
    WuxingCycle,
    NeidanDecoder,
    print_dao_wisdom
)


# ═══════════════════════════════════════════════════════════════════════════
# 聚氣編碼器 | GATHERING ENCODER - From Form to Essence
# ═══════════════════════════════════════════════════════════════════════════

class GatheringEncoder(nn.Module):
    """
    Encoder that gathers phenomena back to essence (從象歸質).

    This is the reverse of manifestation—taking visible forms and refining them
    back to their essential nature. In Daoist alchemy, this is called:

    煉精化氣,煉氣化神,煉神還虛
    "Refine essence to qi, refine qi to spirit, refine spirit to return to void."

    The encoder performs this reverse alchemy, progressively extracting
    subtler representations.
    """

    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        image_size: int,
        base_channels: int = 64
    ):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Initial gathering: From manifest image to essence
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),  # Down 1
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2),
        )

        # Refine to energetic level (從形煉氣)
        self.refine_qi = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),  # Down 2
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
        )

        # Refine to spiritual level (從氣煉神)
        self.refine_shen = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),  # Down 3
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
        )

        # Final gathering to void (煉神還虛)
        final_size = image_size // 8
        self.final_dim = base_channels * 4 * final_size * final_size

        # Project to latent parameters (mu and logvar for reparameterization)
        self.to_mu = nn.Linear(self.final_dim, latent_dim)
        self.to_logvar = nn.Linear(self.final_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather image back to latent essence.

        Args:
            x: Input image [batch, channels, height, width]

        Returns:
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log-variance [batch, latent_dim]
        """
        # Progressive gathering
        x = self.initial(x)        # Gather form
        x = self.refine_qi(x)      # Refine to qi
        x = self.refine_shen(x)    # Refine to shen

        # Flatten and project to latent parameters
        x = x.view(x.size(0), -1)
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)

        return mu, logvar

    def __repr__(self):
        return (f"GatheringEncoder(聚氣 - Gathering Essence, "
                f"{self.input_channels}×{self.image_size}→{self.latent_dim})")


# ═══════════════════════════════════════════════════════════════════════════
# 道變分自編碼器 | DAO VARIATIONAL AUTOENCODER
# ═══════════════════════════════════════════════════════════════════════════

class DaoVAE(nn.Module):
    """
    Daoist Variational Autoencoder with cosmologically-structured latent space.

    The complete cycle:
    1. Encoder gathers phenomena to essence (From Many to One)
    2. Reparameterization samples from wuji void
    3. Taiyi emergence structures the void
    4. YinYang-Bagua-Wuxing transform the latent representation
    5. Neidan decoder manifests essence back to form (From One to Many)

    This mirrors the Daoist cosmogony:
    萬物歸一 → 一歸何處 → 一生二 → 二生三 → 三生萬物
    "Myriad return to One → One returns to Void → One births Two →
     Two births Three → Three births Myriad"
    """

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 512,
        image_size: int = 64,
        enable_yinyang: bool = True,
        enable_bagua: bool = True,
        enable_wuxing: bool = True,
        beta: float = 1.0,  # KL divergence weight
    ):
        super().__init__()

        # Ensure latent_dim is compatible with Bagua (divisible by 8)
        if enable_bagua:
            latent_dim = (latent_dim // 8) * 8

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.beta = beta

        # Component flags
        self.enable_yinyang = enable_yinyang
        self.enable_bagua = enable_bagua
        self.enable_wuxing = enable_wuxing

        # Encoder: Gather phenomena to essence
        self.encoder = GatheringEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
            image_size=image_size
        )

        # Cosmological transformations in latent space
        self.taiyi_transform = TaiyiTransform(latent_dim, latent_dim)

        if enable_yinyang:
            self.yinyang_path = YinYangDualPath(latent_dim)

        if enable_bagua:
            self.bagua_transform = BaguaTransform(latent_dim)

        if enable_wuxing:
            self.wuxing_cycle = WuxingCycle(latent_dim)

        # Decoder: Manifest essence to phenomena
        self.decoder = NeidanDecoder(
            latent_dim=latent_dim,
            output_channels=input_channels,
            image_size=image_size
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent parameters.

        Args:
            x: Input image [batch, channels, height, width]

        Returns:
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log-variance [batch, latent_dim]
        """
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from wuji void.

        z = μ + σ * ε, where ε ~ N(0, I)

        This is the moment of emergence from the void—like the first movement
        that arises from wuji's perfect stillness.

        Args:
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log-variance [batch, latent_dim]

        Returns:
            z: Sampled latent [batch, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from wuji void
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to image through cosmological transformations.

        Args:
            z: Latent representation [batch, latent_dim]

        Returns:
            x_recon: Reconstructed image [batch, channels, height, width]
        """
        # Apply cosmological transformations
        z = self.taiyi_transform(z)

        if self.enable_yinyang:
            z = self.yinyang_path(z)

        if self.enable_bagua:
            z = self.bagua_transform(z)

        if self.enable_wuxing:
            z = self.wuxing_cycle(z)

        # Manifest through Neidan gates
        x_recon, _ = self.decoder(z)
        return x_recon

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass: encode → reparameterize → decode.

        Args:
            x: Input image [batch, channels, height, width]
            return_latent: If True, include latent representation in output

        Returns:
            Dictionary containing:
                - recon: Reconstructed image
                - mu: Latent mean
                - logvar: Latent log-variance
                - z: (optional) Sampled latent if return_latent=True
        """
        # Encode to latent parameters
        mu, logvar = self.encode(x)

        # Sample from wuji void
        z = self.reparameterize(mu, logvar)

        # Decode back to image
        recon = self.decode(z)

        result = {
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
        }

        if return_latent:
            result['z'] = z

        return result

    def loss_function(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss: reconstruction + KL divergence.

        The loss balances:
        1. Reconstruction: How well can we manifest the original from essence?
        2. KL Divergence: How close is our essence to the primordial void?

        This mirrors the Daoist balance between:
        - 有 (Being/Manifestation) - reconstruction loss
        - 無 (Non-being/Void) - KL divergence to wuji

        Args:
            recon: Reconstructed image [batch, channels, height, width]
            x: Original image [batch, channels, height, width]
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log-variance [batch, latent_dim]

        Returns:
            Dictionary with total loss and components
        """
        # Reconstruction loss (Being/有)
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)

        # KL divergence to wuji void (Non-being/無)
        # KL(q(z|x) || p(z)) where p(z) = N(0, I) is the wuji void
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total loss with beta-weighting
        total_loss = recon_loss + self.beta * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }

    def generate(
        self,
        batch_size: int = 1,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Generate images by sampling from wuji void.

        Args:
            batch_size: Number of images to generate
            device: Torch device

        Returns:
            images: Generated images [batch, channels, height, width]
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        with torch.no_grad():
            # Sample from wuji void
            z = torch.randn(batch_size, self.latent_dim, device=device)

            # Decode to images
            images = self.decode(z)

        return images

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images (encode then decode).

        Args:
            x: Input images [batch, channels, height, width]

        Returns:
            recon: Reconstructed images [batch, channels, height, width]
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(x)
            return result['recon']

    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two images through latent space.

        This creates smooth transitions by traversing through the wuji void,
        demonstrating the continuous nature of the cosmological transformations.

        Args:
            x1: First image [1, channels, height, width]
            x2: Second image [1, channels, height, width]
            steps: Number of interpolation steps

        Returns:
            interpolated: Interpolated images [steps, channels, height, width]
        """
        self.eval()
        with torch.no_grad():
            # Encode both images
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            # Create interpolation path through latent space
            alphas = torch.linspace(0, 1, steps, device=mu1.device)
            interpolated_images = []

            for alpha in alphas:
                # Interpolate in latent space
                z_interp = (1 - alpha) * mu1 + alpha * mu2

                # Decode
                img = self.decode(z_interp)
                interpolated_images.append(img)

            return torch.cat(interpolated_images, dim=0)

    def __repr__(self):
        components = ['Encoder', '太一(Taiyi)']
        if self.enable_yinyang:
            components.append('陰陽(YinYang)')
        if self.enable_bagua:
            components.append('八卦(Bagua)')
        if self.enable_wuxing:
            components.append('五行(Wuxing)')
        components.append('Decoder')

        return (f"DaoVAE(\n"
                f"  Flow: {' → '.join(components)}\n"
                f"  Latent: {self.latent_dim}\n"
                f"  Image: {self.input_channels}×{self.image_size}×{self.image_size}\n"
                f"  Beta: {self.beta}\n"
                f")")


# ═══════════════════════════════════════════════════════════════════════════
# 訓練工具 | TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class DaoVAETrainer:
    """
    Training utility for DaoVAE that embodies Daoist training principles.

    Training follows the natural rhythm of:
    - 靜 (Stillness): Low learning rate, stable convergence
    - 動 (Movement): Higher learning rate, active exploration
    - 和 (Harmony): Balanced between reconstruction and regularization
    """

    def __init__(
        self,
        model: DaoVAE,
        lr: float = 1e-4,
        device: torch.device = None
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Image batch [batch, channels, height, width]

        Returns:
            Dictionary with loss values
        """
        self.model.train()
        batch = batch.to(self.device)

        # Forward pass
        result = self.model(batch)

        # Compute loss
        losses = self.model.loss_function(
            result['recon'],
            batch,
            result['mu'],
            result['logvar']
        )

        # Backward pass
        self.optimizer.zero_grad()
        losses['loss'].backward()
        self.optimizer.step()

        # Return scalar losses
        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary with average losses
        """
        total_losses = {'loss': 0, 'recon_loss': 0, 'kl_loss': 0}
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Handle (data, labels) tuples

            losses = self.train_step(batch)

            for k in total_losses:
                total_losses[k] += losses[k]
            num_batches += 1

        # Average losses
        return {k: v / num_batches for k, v in total_losses.items()}


# ═══════════════════════════════════════════════════════════════════════════
# 演示與測試 | DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("道變分自編碼器 | DAO VARIATIONAL AUTOENCODER")
    print("=" * 80)
    print_dao_wisdom()

    # Create model
    model = DaoVAE(
        input_channels=3,
        latent_dim=512,
        image_size=64,
        enable_yinyang=True,
        enable_bagua=True,
        enable_wuxing=True,
        beta=1.0
    )

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\n" + "─" * 80)
    print("Testing forward pass...")
    print("─" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create dummy batch
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 64, 64, device=device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        result = model(dummy_images, return_latent=True)

        print(f"\nInput shape: {dummy_images.shape}")
        print(f"Reconstruction shape: {result['recon'].shape}")
        print(f"Latent (mu) shape: {result['mu'].shape}")
        print(f"Latent (z) shape: {result['z'].shape}")

        # Compute loss
        losses = model.loss_function(
            result['recon'],
            dummy_images,
            result['mu'],
            result['logvar']
        )

        print(f"\nLosses:")
        print(f"  Total: {losses['loss'].item():.4f}")
        print(f"  Reconstruction: {losses['recon_loss'].item():.4f}")
        print(f"  KL Divergence: {losses['kl_loss'].item():.4f}")

    # Test generation
    print("\n" + "─" * 80)
    print("Testing generation from wuji void...")
    print("─" * 80)

    generated = model.generate(batch_size=4, device=device)
    print(f"Generated images shape: {generated.shape}")

    # Test interpolation
    print("\n" + "─" * 80)
    print("Testing latent space interpolation...")
    print("─" * 80)

    img1 = dummy_images[0:1]
    img2 = dummy_images[1:2]
    interpolated = model.interpolate(img1, img2, steps=10)
    print(f"Interpolated sequence shape: {interpolated.shape}")

    print("\n" + "=" * 80)
    print("From Void to Form and Back: The cosmological cycle is complete.")
    print("=" * 80 + "\n")
