"""
天機優化器演示 - TianJi Optimizer Demonstration
=============================================

Demonstrates the TianJi Optimizer on MNIST dataset, comparing it with
traditional optimizers (Adam, SGD) to show the natural convergence
characteristics inspired by Daoist principles.

"應其機而動則萬化安"
"Moving in response to the moment, the ten thousand transformations are at peace."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tianji_optimizer import TianJiOptimizer
import time


# Simple CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_epoch(model, device, train_loader, optimizer, epoch, optimizer_name):
    """Train for one epoch and return loss history."""
    model.train()
    losses = []
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()

        # For TianJi optimizer, pass loss value to observe Heaven's mechanism
        if isinstance(optimizer, TianJiOptimizer):
            optimizer.step(loss=loss.item())
        else:
            optimizer.step()

        losses.append(loss.item())

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            acc = 100. * correct / total
            print(f'{optimizer_name} - Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\tAcc: {acc:.2f}%')

    epoch_loss = np.mean(losses)
    epoch_acc = 100. * correct / total
    return losses, epoch_loss, epoch_acc


def test(model, device, test_loader):
    """Test the model and return accuracy and loss."""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy


def run_experiment(optimizer_name, optimizer_fn, train_loader, test_loader, device, epochs=5):
    """Run training experiment with given optimizer."""
    print(f"\n{'='*60}")
    print(f"Training with {optimizer_name}")
    print(f"{'='*60}\n")

    model = SimpleCNN().to(device)
    optimizer = optimizer_fn(model.parameters())

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        batch_losses, epoch_loss, epoch_acc = train_epoch(
            model, device, train_loader, optimizer, epoch, optimizer_name
        )
        train_losses.extend(batch_losses)
        train_accs.append(epoch_acc)

        test_loss, test_acc = test(model, device, test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # For TianJi optimizer, print mechanism state
        if isinstance(optimizer, TianJiOptimizer):
            mechanism_state = optimizer.get_mechanism_state()
            print(f"天機狀態 (Mechanism State):")
            print(f"  Direction: {mechanism_state['tian_ji_direction']}")
            print(f"  Wu Wei state: {mechanism_state['in_wu_wei']}")
            print(f"  Step: {mechanism_state['step']}")

    elapsed_time = time.time() - start_time

    return {
        'model': model,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'time': elapsed_time,
    }


def plot_comparison(results_dict, save_path='tianji_comparison.png'):
    """Plot comparison of different optimizers."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Training loss curves (smoothed)
    ax1 = axes[0, 0]
    for name, results in results_dict.items():
        losses = results['train_losses']
        # Smooth with moving average
        window = 50
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax1.plot(smoothed, label=name, linewidth=2)
    ax1.set_xlabel('Training Steps (Smoothed)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison\n觀其機 - Observing the Mechanism', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test accuracy
    ax2 = axes[0, 1]
    for name, results in results_dict.items():
        epochs = range(1, len(results['test_accs']) + 1)
        ax2.plot(epochs, results['test_accs'], marker='o', label=name, linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy Comparison\n萬化安 - All Transformations at Peace', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Training time
    ax3 = axes[1, 0]
    names = list(results_dict.keys())
    times = [results_dict[name]['time'] for name in names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax3.bar(names, times, color=colors[:len(names)])
    ax3.set_ylabel('Training Time (seconds)', fontsize=12)
    ax3.set_title('Training Time Comparison\n無為而治 - Govern Through Non-Action', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=10)

    # Plot 4: Final performance summary
    ax4 = axes[1, 1]
    final_accs = [results_dict[name]['test_accs'][-1] for name in names]
    x = np.arange(len(names))
    bars = ax4.bar(x, final_accs, color=colors[:len(names)])
    ax4.set_ylabel('Final Test Accuracy (%)', fontsize=12)
    ax4.set_title('Final Test Accuracy\n至靜至神 - Utmost Stillness, Utmost Spirit', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.set_ylim([min(final_accs) - 1, 100])
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {save_path}")
    plt.close()


def main():
    """
    Main demonstration function.

    Trains three models with different optimizers:
    1. TianJi Optimizer (天機優化器) - Our Daoist optimizer
    2. Adam - Standard adaptive optimizer
    3. SGD with Momentum - Classical optimizer
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    天機優化器演示                              ║
    ║              TianJi Optimizer Demonstration                   ║
    ║                                                               ║
    ║  "觀天之道，執天之行，盡矣"                                      ║
    ║  "Observe the Dao of Heaven, grasp its operations—           ║
    ║   thus all is complete."                                      ║
    ║                                                               ║
    ║  Based on wisdom from:                                        ║
    ║  - 天機經 (Classic of Heaven's Mechanism)                     ║
    ║  - 黃帝陰符經 (Yellow Emperor's Yin Fu Jing)                  ║
    ║  - 化書 (Book of Transformations)                            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("\nLoading MNIST dataset...")
    train_dataset = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data/MNIST', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Define optimizers
    optimizers = {
        'TianJi (天機)': lambda params: TianJiOptimizer(
            params,
            lr=1e-3,
            beta_tian=0.9,
            beta_di=0.999,
            beta_ren=0.1,
            yin_yang_cycle=100,
        ),
        'Adam': lambda params: torch.optim.Adam(params, lr=1e-3),
        'SGD+Momentum': lambda params: torch.optim.SGD(params, lr=1e-2, momentum=0.9),
    }

    # Run experiments
    results = {}
    for name, opt_fn in optimizers.items():
        results[name] = run_experiment(
            name, opt_fn, train_loader, test_loader, device, epochs=5
        )

    # Plot comparison
    plot_comparison(results)

    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Final Test Accuracy: {res['test_accs'][-1]:.2f}%")
        print(f"  Training Time: {res['time']:.2f}s")
        print(f"  Final Test Loss: {res['test_losses'][-1]:.4f}")

    print("""
    \n╔═══════════════════════════════════════════════════════════════╗
    ║                    Cosmic Observations                        ║
    ╚═══════════════════════════════════════════════════════════════╝

    The TianJi Optimizer demonstrates several Daoist principles:

    🐉 天機 (Tian Ji - Celestial Mechanism):
       Observes global training trends and adapts accordingly

    🌍 地機 (Di Ji - Earthly Mechanism):
       Responds to local gradient landscape

    ❤️  人機 (Ren Ji - Heart Mechanism):
       Mediates between heaven and earth through internal state

    ☯️  陰陽調和 (Yin-Yang Balance):
       Oscillates between exploration (yang) and exploitation (yin)

    🍃 無為而治 (Wu Wei - Non-Action):
       Natural dampening as convergence approaches

    🌟 五賊合一 (Five Thieves United):
       Gathers wisdom from fate, things, time, merit, and numinous

    "應其機而動則萬化安"
    "Moving in response to the moment,
     the ten thousand transformations are at peace."
    """)


if __name__ == "__main__":
    main()
