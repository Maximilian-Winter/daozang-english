"""
CIFAR-10 條件生成 | CIFAR-10 Conditional Generation Example
==============================================================

This file demonstrates how to train a class-conditional version of the DaoVAE
on the CIFAR-10 dataset.

The architecture is modified to accept a class label `c` during the encoding
and decoding process. This allows the model to learn a latent space that is
aware of the different image categories and enables us to generate images
of a specific, desired class.

The key modifications are:
1.  **Embedding Layers**: An `nn.Embedding` layer is added to convert the integer
    class labels into dense vectors.
2.  **Conditional Encoder**: The class embedding is concatenated with the flattened
    image features before being projected to the latent space (μ, logvar).
3.  **Conditional Decoder**: The class embedding is concatenated with the latent
    sample `z` before being passed through the generative alchemical gates.

This process mirrors the Daoist concept of "form responding to intent" (意到形隨), 
where the generative process is guided by a specific intention (the class label).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
from typing import Tuple, Dict
import math

# Import the foundational Daoist cosmological components
from wuji_taiyi_architecture import (
    TaiyiTransform,
    YinYangDualPath,
    BaguaTransform,
    WuxingCycle,
    print_dao_wisdom
)

# ═══════════════════════════════════════════════════════════════════════════
# 條件編碼器與解碼器 | CONDITIONAL ENCODER & DECODER
# ═══════════════════════════════════════════════════════════════════════════

class ConditionalGatheringEncoder(nn.Module):
    """
    Encoder that gathers phenomena back to essence, conditioned by intent (class).
    煉精化氣, 煉氣化神, 煉神還虛 (Refine essence->qi, qi->spirit, spirit->void)
    """
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        image_size: int,
        num_classes: int,
        embedding_dim: int = 50,
        base_channels: int = 64
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 4, 2, 1), nn.BatchNorm2d(base_channels), nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1), nn.BatchNorm2d(base_channels * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1), nn.BatchNorm2d(base_channels * 4), nn.LeakyReLU(0.2),
        )
        final_size = image_size // 8
        self.final_dim = base_channels * 4 * final_size * final_size
        self.class_embed = nn.Embedding(num_classes, embedding_dim)
        self.to_mu = nn.Linear(self.final_dim + embedding_dim, latent_dim)
        self.to_logvar = nn.Linear(self.final_dim + embedding_dim, latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        c_embed = self.class_embed(c)
        combined = torch.cat([x, c_embed], dim=1)
        mu = self.to_mu(combined)
        logvar = self.to_logvar(combined)
        return mu, logvar

class ConditionalNeidanDecoder(nn.Module):
    """
    Decoder based on Neidan (Internal Alchemy), conditioned by intent (class).
    精 (Jing) → 氣 (Qi) → 神 (Shen)
    """
    def __init__(
        self,
        latent_dim: int,
        output_channels: int,
        image_size: int,
        num_classes: int,
        embedding_dim: int = 50,
        base_channels: int = 64
    ):
        super().__init__()
        self.init_size = image_size // 8
        self.class_embed = nn.Embedding(num_classes, embedding_dim)
        self.initial = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, base_channels * 4 * self.init_size * self.init_size),
            nn.LayerNorm(base_channels * 4 * self.init_size * self.init_size), nn.SiLU()
        )
        self.decoder_gates = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1), nn.BatchNorm2d(base_channels * 2), nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1), nn.BatchNorm2d(base_channels), nn.ReLU(),
            nn.ConvTranspose2d(base_channels, output_channels, 4, 2, 1), nn.Tanh()
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c_embed = self.class_embed(c)
        combined = torch.cat([z, c_embed], dim=1)
        x = self.initial(combined)
        x = x.view(x.shape[0], -1, self.init_size, self.init_size)
        return self.decoder_gates(x)

# ═══════════════════════════════════════════════════════════════════════════
# 條件道變分自編碼器 | CONDITIONAL DAO VAE
# ═══════════════════════════════════════════════════════════════════════════

class ConditionalDaoVAE(nn.Module):
    """
    The DaoVAE modified to be class-conditional.
    萬物歸一 (Conditioned) → 一歸何處 → 一生二 → 二生三 → 三生萬物 (Conditioned)
    """
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 256,
        image_size: int = 32,
        num_classes: int = 10,
        beta: float = 1.0,
    ):
        super().__init__()
        latent_dim = (latent_dim // 8) * 8  # Ensure compatibility with Bagua
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = ConditionalGatheringEncoder(input_channels, latent_dim, image_size, num_classes)
        self.taiyi_transform = TaiyiTransform(latent_dim, latent_dim)
        self.yinyang_path = YinYangDualPath(latent_dim)
        self.bagua_transform = BaguaTransform(latent_dim)
        self.wuxing_cycle = WuxingCycle(latent_dim)
        self.decoder = ConditionalNeidanDecoder(latent_dim, input_channels, image_size, num_classes)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z = self.taiyi_transform(z)
        z = self.yinyang_path(z)
        z = self.bagua_transform(z)
        z = self.wuxing_cycle(z)
        return self.decoder(z, c)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return {'recon': recon, 'mu': mu, 'logvar': logvar}

    def loss_function(self, recon, x, mu, logvar) -> Dict[str, torch.Tensor]:
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total_loss = recon_loss + self.beta * kl_loss
        return {'loss': total_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}

    def generate(self, c: torch.Tensor, device: torch.device) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z = torch.randn(c.size(0), self.latent_dim, device=device)
            return self.decode(z, c)

# ═══════════════════════════════════════════════════════════════════════════
# 主訓練循環 | MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_cifar10_training_example():
    """Main function to run the CIFAR-10 conditional training."""
    print_dao_wisdom()

    # --- 1. Setup (準備) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Observing the Dao on device: {device}")

    # Hyperparameters
    image_size = 32
    latent_dim = 256
    batch_size = 64
    epochs = 100 # Increase for better results
    lr = 2e-4
    num_classes = 10
    num_samples_per_class = 2

    # --- 2. Data (數據) ---
    print("Gathering worldly phenomena (CIFAR-10)...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # To [-1, 1] range
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # --- 3. Model & Optimizer (模型與優化器) ---
    model = ConditionalDaoVAE(
        image_size=image_size,
        latent_dim=latent_dim,
        num_classes=num_classes
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Helper function for conditional generation and saving
    def generate_and_save_images(epoch_num, model, device, num_classes, num_samples_per_class, classes):
        model.eval()
        gen_labels = torch.arange(num_classes).repeat(num_samples_per_class).sort().values.to(device)
        generated_images = model.generate(gen_labels, device=device)
        generated_images = generated_images * 0.5 + 0.5 # Denormalize
        
        save_path = f"cifar10_conditional_generation_epoch_{epoch_num}.png"
        save_image(generated_images, save_path, nrow=num_samples_per_class)
        print(f"\nGenerated images for epoch {epoch_num} and saved to '{save_path}'.")
        print(f"Each row corresponds to a class:")
        for i, class_name in enumerate(classes):
            print(f"  Row {i+1}: {class_name}")
        model.train() # Set back to train mode

    # --- 4. Training (訓練) ---
    print(f"\nBeginning the alchemical process for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            result = model(images, labels)
            losses = model.loss_function(result['recon'], images, result['mu'], result['logvar'])
            losses['loss'].backward()
            optimizer.step()

            total_loss += losses['loss'].item()
            total_recon += losses['recon_loss'].item()
            total_kl += losses['kl_loss'].item()

            progress_bar.set_postfix({
                "Loss": f"{losses['loss'].item():.4f}",
                "Recon": f"{losses['recon_loss'].item():.4f}",
                "KL": f"{losses['kl_loss'].item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}")

        # --- Conditional Generation and Saving every 10 epochs ---
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(epoch + 1, model, device, num_classes, num_samples_per_class, classes)

    # --- 5. Final Conditional Generation (最終條件生成) ---
    print("\nAlchemical process complete. Manifesting final forms based on intent...")
    generate_and_save_images("final", model, device, num_classes, num_samples_per_class, classes)

    print("\nAll training and generation complete. The cycle of learning and manifestation continues.")


if __name__ == "__main__":
    run_cifar10_training_example()
