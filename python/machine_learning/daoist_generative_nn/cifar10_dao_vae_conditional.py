"""
CIFAR-10 條件道VAE | CIFAR-10 Conditional DaoVAE
=====================================================

This file adapts the original `DaoVAE` architecture from `dao_vae.py` to perform
class-conditional generation on the CIFAR-10 dataset.

It uses inheritance to clearly show the modifications needed to make the
original VAE conditional:
1.  `ConditionalGatheringEncoder` inherits from `GatheringEncoder`.
2.  `ConditionalNeidanDecoder` inherits from `NeidanDecoder`.
3.  `ConditionalDaoVAE` inherits from `DaoVAE` and uses the new conditional
    encoder/decoder.

The training loop is structured to train for 100 epochs, saving a sample of
generated images every 10 epochs to visualize the learning process.
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

# Import the original, non-conditional architectures as a base
from dao_vae import DaoVAE, GatheringEncoder
from wuji_taiyi_architecture import NeidanDecoder, print_dao_wisdom

# ═══════════════════════════════════════════════════════════════════════════
# 條件化模塊 (繼承) | CONDITIONAL MODULES (VIA INHERITANCE)
# ═══════════════════════════════════════════════════════════════════════════

class ConditionalGatheringEncoder(GatheringEncoder):
    """Inherits from GatheringEncoder and adds class-conditioning."""
    def __init__(self, num_classes, embedding_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.class_embed = nn.Embedding(num_classes, embedding_dim)
        # Redefine the final layers to accept the concatenated class embedding
        self.to_mu = nn.Linear(self.final_dim + embedding_dim, self.latent_dim)
        self.to_logvar = nn.Linear(self.final_dim + embedding_dim, self.latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Perform original convolutions
        x = self.initial(x)
        x = self.refine_qi(x)
        x = self.refine_shen(x)
        # Flatten and concatenate with class embedding
        x = x.view(x.size(0), -1)
        c_embed = self.class_embed(c)
        combined = torch.cat([x, c_embed], dim=1)
        # Project to latent space
        mu = self.to_mu(combined)
        logvar = self.to_logvar(combined)
        return mu, logvar

class ConditionalNeidanDecoder(NeidanDecoder):
    """Inherits from NeidanDecoder and adds class-conditioning."""
    def __init__(self, num_classes, embedding_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.class_embed = nn.Embedding(num_classes, embedding_dim)
        # Redefine the initial layer to accept the concatenated class embedding
        self.initial = nn.Sequential(
            nn.Linear(self.latent_dim + embedding_dim, self.decoder.initial[0].out_features),
            nn.LayerNorm(self.decoder.initial[0].out_features),
            nn.SiLU()
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Concatenate latent vector with class embedding
        c_embed = self.class_embed(c)
        combined = torch.cat([z, c_embed], dim=1)
        # Pass through the original decoder's forward path
        return super().forward(combined)

class ConditionalDaoVAE(DaoVAE):
    """
    Inherits from DaoVAE, replacing the encoder/decoder with conditional versions
    and overriding methods to handle class labels.
    """
    def __init__(self, num_classes: int = 10, **kwargs):
        # Initialize the original DaoVAE to get its structure
        super().__init__(**kwargs)
        embedding_dim = 50 # Internal dimension for class embeddings

        # Replace the encoder and decoder with our new conditional versions
        self.encoder = ConditionalGatheringEncoder(
            input_channels=self.input_channels,
            latent_dim=self.latent_dim,
            image_size=self.image_size,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )
        self.decoder = ConditionalNeidanDecoder(
            latent_dim=self.latent_dim,
            output_channels=self.input_channels,
            image_size=self.image_size,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )
        print("Replaced original DaoVAE encoder/decoder with conditional versions.")

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, c)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Apply the original cosmological transformations
        z = super().decode(z) # This calls the parent's decode path up to the Neidan part
        # Now, use the conditional Neidan decoder
        x_recon, _ = self.decoder(z, c)
        return x_recon

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return {'recon': recon, 'mu': mu, 'logvar': logvar}

    def generate(self, c: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Generates images conditioned on class labels `c`. """
        self.eval()
        with torch.no_grad():
            # Sample from wuji void
            z = torch.randn(c.size(0), self.latent_dim, device=device)
            # Decode to images with specified intent `c`
            images = self.decode(z, c)
        return images

# ═══════════════════════════════════════════════════════════════════════════
# 主訓練循環 | MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_conditional_dao_vae_training():
    """Main function to run the CIFAR-10 conditional training."""
    print_dao_wisdom()

    # --- 1. Setup (準備) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Observing the Dao on device: {device}")

    # Hyperparameters
    image_size = 32
    latent_dim = 256
    batch_size = 64
    epochs = 100
    lr = 2e-4
    num_classes = 10
    num_samples_per_class = 8 # For generation

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

    # --- Helper function for generation ---
    def generate_and_save_images(epoch_num):
        model.eval()
        gen_labels = torch.arange(num_classes).repeat(num_samples_per_class).sort().values.to(device)
        generated_images = model.generate(gen_labels, device=device)
        generated_images = generated_images * 0.5 + 0.5 # Denormalize
        
        save_path = f"cifar10_dao_vae_epoch_{epoch_num}.png"
        save_image(generated_images, save_path, nrow=num_samples_per_class)
        print(f"\nGenerated images for epoch {epoch_num} and saved to '{save_path}'.")
        model.train() # Set back to train mode

    # --- 4. Training (訓練) ---
    print(f"\nBeginning the alchemical process for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            result = model(images, labels)
            losses = model.loss_function(result['recon'], images, result['mu'], result['logvar'])
            losses['loss'].backward()
            optimizer.step()

            progress_bar.set_postfix({
                "Loss": f"{losses['loss'].item():.4f}",
                "Recon": f"{losses['recon_loss'].item():.4f}",
            })

        # --- Generate images every 10 epochs ---
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(epoch + 1)

    # --- 5. Final Generation (最終生成) ---
    print("\nAlchemical process complete. Manifesting final forms...")
    generate_and_save_images("final")
    print("\nAll training and generation complete.")

if __name__ == "__main__":
    run_conditional_dao_vae_training()
