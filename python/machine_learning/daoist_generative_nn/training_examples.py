"""
訓練循環示例 | Training Loop Examples
========================================

This file provides example training loops for the Daoist generative architectures:
1. DaoVAE: The Daoist Variational Autoencoder
2. DaoGAN: The Wuji-Taiyi Generative Adversarial Network

These examples use dummy data for demonstration purposes. To train on your
own data, replace the dummy `TensorDataset` with your own `torch.utils.data.Dataset`.

These examples demonstrate the Daoist principles in action:
- The VAE learns to gather (encode) and manifest (decode).
- The GAN learns to manifest (generate) against a discerning force (discriminator).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import the Daoist architectures
from dao_vae import DaoVAE, DaoVAETrainer
from wuji_taiyi_architecture import DaoGAN, print_dao_wisdom

# ═══════════════════════════════════════════════════════════════════════════
# 示例 1: 訓練道變分自編碼器 | Example 1: Training the DaoVAE
# ═══════════════════════════════════════════════════════════════════════════

def train_dao_vae_example():
    """
    An example training loop for the DaoVAE.
    This demonstrates the cycle of gathering phenomena to essence and manifesting
    it back to form (From Many to One, and back to Many).
    """
    print("=" * 80)
    print("道變分自編碼器 (DaoVAE) 訓練示例 | DaoVAE Training Example")
    print("=" * 80)

    # --- 1. Setup (準備) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Model & Data (模型與數據) ---
    image_size = 64
    latent_dim = 256
    batch_size = 16
    epochs = 3  # Keep it short for an example

    # Create DaoVAE model
    model = DaoVAE(
        input_channels=3,
        latent_dim=latent_dim,
        image_size=image_size,
        beta=1.0  # Balance between reconstruction and regularization
    ).to(device)
    print(model)

    # Create a dummy dataset (replace with your actual data loader)
    print("\nCreating a dummy dataset for demonstration...")
    dummy_data = torch.randn(100, 3, image_size, image_size)
    dummy_dataset = TensorDataset(dummy_data)
    dataloader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

    # --- 3. Trainer (訓練器) ---
    # The provided DaoVAETrainer encapsulates the training logic,
    # following the principle of 和 (Harmony).
    trainer = DaoVAETrainer(model, lr=1e-4, device=device)

    # --- 4. Training Loop (訓練循環) ---
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        # The trainer's `train_epoch` method handles the entire epoch loop.
        avg_losses = trainer.train_epoch(tqdm(dataloader, desc=f"Epoch {epoch+1}"))

        print(
            f"Epoch [{epoch+1}/{epochs}] complete | "
            f"Total Loss: {avg_losses['loss']:.4f} | "
            f"Recon Loss: {avg_losses['recon_loss']:.4f} | "
            f"KL Loss: {avg_losses['kl_loss']:.4f}"
        )

    print("\nTraining complete. The model has learned to gather and manifest.")
    print("-" * 80)

    # --- 5. Generation (生成) ---
    print("Generating a sample from the Wuji void...")
    generated_image = model.generate(batch_size=1, device=device)
    print(f"Generated image shape: {generated_image.shape}")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# 示例 2: 訓練道生成對抗網絡 | Example 2: Training the DaoGAN
# ═══════════════════════════════════════════════════════════════════════════

class SimpleDiscriminator(nn.Module):
    """
    A simple discriminator to distinguish real from generated images.
    It embodies the principle of discernment (辨別), providing the resistance
    against which the generator's creative force (生) must push.
    """
    def __init__(self, input_channels, image_size):
        super().__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.main = nn.Sequential(
            *block(input_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 1, kernel_size=image_size // 8, stride=1, padding=0),
            nn.Flatten()
        )

    def forward(self, img):
        return self.main(img)

def train_dao_gan_example():
    """
    An example training loop for the DaoGAN.
    This demonstrates the interplay between manifestation (Generator) and
    discernment (Discriminator), the creative tension of Yin and Yang.
    """
    print("\n" + "=" * 80)
    print("道生成對抗網絡 (DaoGAN) 訓練示例 | DaoGAN Training Example")
    print("=" * 80)

    # --- 1. Setup (準備) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Models & Data (模型與數據) ---
    image_size = 64
    latent_dim = 128
    batch_size = 16
    epochs = 3 # Keep it short for an example
    lr = 2e-4
    beta1 = 0.5

    # Create DaoGAN (Generator)
    generator = DaoGAN(
        latent_dim=latent_dim,
        output_channels=3,
        image_size=image_size,
    ).to(device)
    print(generator)

    # Create Discriminator
    discriminator = SimpleDiscriminator(
        input_channels=3,
        image_size=image_size
    ).to(device)
    print(f"\nDiscriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Create a dummy dataset (replace with your actual data loader)
    print("\nCreating a dummy dataset for demonstration...")
    dummy_data = torch.randn(100, 3, image_size, image_size)
    dummy_dataset = TensorDataset(dummy_data)
    dataloader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

    # --- 3. Optimizers & Loss (優化器與損失) ---
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # --- 4. Training Loop (訓練循環) ---
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for i, data in enumerate(progress_bar):
            # --- Train Discriminator (訓練辨別器) ---
            discriminator.zero_grad()

            # Train with real images
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            real_labels = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            
            d_output_real = discriminator(real_images).view(-1)
            loss_d_real = criterion(d_output_real, real_labels)
            loss_d_real.backward()

            # Train with fake images
            fake_images = generator.generate(b_size, device=device)
            fake_labels = torch.full((b_size,), 0.0, dtype=torch.float, device=device)
            
            d_output_fake = discriminator(fake_images.detach()).view(-1)
            loss_d_fake = criterion(d_output_fake, fake_labels)
            loss_d_fake.backward()
            optimizer_d.step()
            loss_d = loss_d_real + loss_d_fake

            # --- Train Generator (訓練生成器) ---
            generator.zero_grad()
            # We want the discriminator to think the fake images are real
            g_output = discriminator(fake_images).view(-1)
            loss_g = criterion(g_output, real_labels) # Use real_labels
            loss_g.backward()
            optimizer_g.step()

            progress_bar.set_postfix({
                "G_loss": f"{loss_g.item():.4f}",
                "D_loss": f"{loss_d.item():.4f}"
            })

        print(
            f"Epoch [{epoch+1}/{epochs}] complete | "
            f"Generator Loss: {loss_g.item():.4f} | "
            f"Discriminator Loss: {loss_d.item():.4f}"
        )

    print("\nTraining complete. The generator now manifests from the void with greater clarity.")
    print("-" * 80)

    # --- 5. Generation (生成) ---
    print("Generating a final sample from the trained DaoGAN...")
    generated_image = generator.generate(batch_size=1, device=device)
    print(f"Generated image shape: {generated_image.shape}")
    print("=" * 80)


if __name__ == "__main__":
    print_dao_wisdom()
    train_dao_vae_example()
    train_dao_gan_example()
    print("\nAll examples complete. The cycle of learning and manifestation continues.")
