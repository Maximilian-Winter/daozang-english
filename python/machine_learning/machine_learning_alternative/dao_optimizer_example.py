"""
DaoOptimizer Example & Benchmark
═════════════════════════════════

Demonstrates the power of Taoist-inspired optimization compared to
standard optimizers (Adam, SGD, RMSprop).

"The Dao that can be told is not the eternal Dao"
...but we can certainly benchmark it! 😊

This script:
1. Trains a neural network on MNIST/CIFAR-10
2. Compares DaoOptimizer against Adam, SGD, RMSprop
3. Visualizes the training dynamics
4. Shows the Taoist principles in action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from dao_optimizer import DaoOptimizer
import time
from typing import Dict, List


# ═══════════════════════════════════════════════════════════════════
# Model Architecture: Simple CNN
# ═══════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """
    A simple CNN for image classification.
    Nothing fancy - let the optimizer do the magic.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# Training Functions
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Optimizer step - the Dao flows here
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ═══════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════

def benchmark_optimizers(
    dataset_name='mnist',
    num_epochs=10,
    batch_size=128,
    learning_rate=0.01,
    device=None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Benchmark DaoOptimizer against standard optimizers.

    Args:
        dataset_name: 'mnist' or 'cifar10'
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Base learning rate
        device: torch.device (auto-detected if None)

    Returns:
        Dictionary containing training histories for each optimizer
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"🏮 DaoOptimizer Benchmark")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"{'='*60}\n")

    # ═══════════════════════════════════════════════════════════════
    # Load Dataset
    # ═══════════════════════════════════════════════════════════════

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST stats
    ])

    if dataset_name.lower() == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ═══════════════════════════════════════════════════════════════
    # Define Optimizers to Benchmark
    # ═══════════════════════════════════════════════════════════════

    optimizer_configs = {
        'DaoOptimizer': lambda params: DaoOptimizer(
            params,
            lr=learning_rate,
            beta_yang=0.9,
            beta_yin=0.999,
            weight_decay=1e-4,
            orbit_cycle=365,
            orbit_amplitude=0.1
        ),
        'Adam': lambda params: torch.optim.Adam(
            params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        ),
        'SGD+Momentum': lambda params: torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        ),
        'RMSprop': lambda params: torch.optim.RMSprop(
            params,
            lr=learning_rate,
            alpha=0.99,
            weight_decay=1e-4
        ),
    }

    # ═══════════════════════════════════════════════════════════════
    # Train Each Optimizer
    # ═══════════════════════════════════════════════════════════════

    results = {}
    criterion = nn.CrossEntropyLoss()

    for optimizer_name, optimizer_fn in optimizer_configs.items():
        print(f"\n{'─'*60}")
        print(f"Training with {optimizer_name}")
        print(f"{'─'*60}")

        # Create fresh model
        model = SimpleCNN(num_classes=num_classes).to(device)
        optimizer = optimizer_fn(model.parameters())

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epoch_time': []
        }

        # Train for num_epochs
        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            epoch_time = time.time() - start_time

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['epoch_time'].append(epoch_time)

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")

            # Special: Print Dao state for DaoOptimizer
            if optimizer_name == 'DaoOptimizer':
                dao_state = optimizer.get_dao_state()
                print(f"  🏮 Dao State: Step={dao_state['avg_step']:.0f}, "
                      f"Orbit={dao_state['orbit_progress']}, "
                      f"Yang={dao_state['avg_yang_momentum_norm']:.6f}, "
                      f"Yin={dao_state['avg_yin_variance_norm']:.6f}")

        results[optimizer_name] = history

    return results


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_results(results: Dict[str, Dict[str, List[float]]], save_path='dao_benchmark.png'):
    """
    Plot training dynamics for all optimizers.

    "A picture is worth a thousand gradients" - Grace Hopper (probably)
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DaoOptimizer vs Standard Optimizers\n道优化器 vs 标准优化器', fontsize=16, fontweight='bold')

    # Define colors for each optimizer
    colors = {
        'DaoOptimizer': '#FF6B6B',  # Red (Fire - 火)
        'Adam': '#4ECDC4',          # Cyan (Water - 水)
        'SGD+Momentum': '#95E1D3',  # Light green (Wood - 木)
        'RMSprop': '#F3A683',       # Orange (Earth - 土)
    }

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for optimizer_name, history in results.items():
        ax.plot(history['train_loss'], label=optimizer_name,
                color=colors.get(optimizer_name, 'gray'), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss (Lower is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Test Accuracy
    ax = axes[0, 1]
    for optimizer_name, history in results.items():
        ax.plot(history['test_acc'], label=optimizer_name,
                color=colors.get(optimizer_name, 'gray'), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy (Higher is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Test Loss
    ax = axes[1, 0]
    for optimizer_name, history in results.items():
        ax.plot(history['test_loss'], label=optimizer_name,
                color=colors.get(optimizer_name, 'gray'), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss (Lower is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Final Comparison Bar Chart
    ax = axes[1, 1]
    optimizer_names = list(results.keys())
    final_test_accs = [results[name]['test_acc'][-1] for name in optimizer_names]
    bars = ax.bar(range(len(optimizer_names)), final_test_accs,
                  color=[colors.get(name, 'gray') for name in optimizer_names])
    ax.set_xticks(range(len(optimizer_names)))
    ax.set_xticklabels(optimizer_names, rotation=15, ha='right')
    ax.set_ylabel('Final Test Accuracy (%)')
    ax.set_title('Final Test Accuracy Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, final_test_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Results saved to: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# Summary Statistics
# ═══════════════════════════════════════════════════════════════════

def print_summary(results: Dict[str, Dict[str, List[float]]]):
    """Print summary statistics for all optimizers."""

    print("\n" + "="*60)
    print("📈 FINAL RESULTS SUMMARY")
    print("="*60)

    summary_data = []
    for optimizer_name, history in results.items():
        final_train_acc = history['train_acc'][-1]
        final_test_acc = history['test_acc'][-1]
        best_test_acc = max(history['test_acc'])
        avg_epoch_time = np.mean(history['epoch_time'])

        summary_data.append({
            'Optimizer': optimizer_name,
            'Final Train Acc': f"{final_train_acc:.2f}%",
            'Final Test Acc': f"{final_test_acc:.2f}%",
            'Best Test Acc': f"{best_test_acc:.2f}%",
            'Avg Epoch Time': f"{avg_epoch_time:.2f}s"
        })

    # Print as table
    headers = ['Optimizer', 'Final Train Acc', 'Final Test Acc', 'Best Test Acc', 'Avg Epoch Time']
    col_widths = [max(len(str(row.get(h, ''))) for row in summary_data + [{'Optimizer': h}]) + 2
                  for h in headers]

    # Print header
    header_row = "│".join(f" {h:<{w}} " for h, w in zip(headers, col_widths))
    print("┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐")
    print("│" + header_row + "│")
    print("├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤")

    # Print rows
    for row_data in summary_data:
        row = "│".join(f" {str(row_data[h]):<{w}} " for h, w in zip(headers, col_widths))
        print("│" + row + "│")

    print("└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘")

    # Find winner
    best_optimizer = max(summary_data, key=lambda x: float(x['Best Test Acc'].rstrip('%')))
    print(f"\n🏆 Winner: {best_optimizer['Optimizer']} with {best_optimizer['Best Test Acc']} test accuracy!")

    # Wisdom quote
    print("\n💭 Wisdom from the Daozang:")
    print("   \"The softest under heaven gallops through the hardest.\"")
    print("   \"Through non-action, nothing is left undone.\"")
    print("   \"道法自然 (The Dao follows nature)\"\n")


# ═══════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║           🏮 DaoOptimizer Benchmark 🏮                     ║
    ║              道优化器性能测试                                 ║
    ║                                                            ║
    ║  "The Dao gives them life; Virtue nurtures them."         ║
    ║  "Water benefits all things yet does not contend."        ║
    ║                                                            ║
    ║  Comparing Taoist-inspired optimization against            ║
    ║  standard methods (Adam, SGD, RMSprop)                     ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Run benchmark
    results = benchmark_optimizers(
        dataset_name='mnist',
        num_epochs=10,
        batch_size=128,
        learning_rate=0.01
    )

    # Visualize results
    plot_results(results, save_path='dao_benchmark.png')

    # Print summary
    print_summary(results)

    print("\n✨ Benchmark complete! May the Dao be with your gradients. ✨\n")
