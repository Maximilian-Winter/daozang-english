"""
MNIST Training Example with Dao Optimizer
==========================================

This example demonstrates using the Dao Optimizer to train a simple
neural network on the MNIST dataset.

The training process embodies the Daoist principle of balance:
- Early training: More exploration (Yang)
- Late training: More exploitation (Yin)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import dao_optimizer
sys.path.append(str(Path(__file__).parent.parent))
from dao_optimizer import DaoOptimizer, DaoScheduler, diagnose_qi_flow, print_dao_wisdom


class SimpleNet(nn.Module):
    """A simple fully-connected network for MNIST."""

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train_epoch(model, device, train_loader, optimizer, epoch, log_interval=100):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

        if batch_idx % log_interval == 0:
            phase, progress = optimizer.get_current_phase()
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct / total

            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\t'
                  f'Acc: {accuracy:.2f}%\t'
                  f'Phase: {phase} ({progress:.0%})')

    return total_loss / len(train_loader), 100. * correct / total


def test(model, device, test_loader):
    """Evaluate on test set."""
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


def main():
    # Print Dao wisdom at the start
    print_dao_wisdom()

    # Configuration
    batch_size = 128
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model
    model = SimpleNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}\n")

    # Optimizer - The Dao way!
    optimizer = DaoOptimizer(
        model.parameters(),
        lr=1e-3,
        tian_beta=0.9,         # Heaven's momentum
        di_beta=0.999,         # Earth's second moment
        wu_wei_factor=0.1,     # Non-forcing exploration
        yin_yang_balance=0.5,  # Balanced exploration/exploitation
        adaptive_qi=True,      # Enable adaptive momentum
        wuxing_cycle=len(train_loader) * 2  # Complete one Wu Xing cycle every 2 epochs
    )

    # Scheduler - Seasonal learning rate
    total_steps = len(train_loader) * epochs
    scheduler = DaoScheduler(optimizer, total_steps=total_steps, min_lr_factor=0.1)

    print(optimizer)
    print("\n" + "="*70)
    print("開始修煉 | Beginning Cultivation (Training)")
    print("="*70 + "\n")

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'phase': []
    }

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['phase'].append(optimizer.get_current_phase()[0])

        # Step scheduler
        for _ in range(len(train_loader)):
            scheduler.step()

        # Diagnose Qi flow at end of each epoch
        if epoch % 2 == 0:
            print("\n" + "-"*70)
            print("氣流診斷 | Qi Flow Diagnostics")
            print("-"*70)
            diag = diagnose_qi_flow(optimizer)
            print(f"Current Phase: {diag['phase']} ({diag['phase_progress']:.1%} complete)")
            print(f"Global Step: {diag['step']}")
            if diag['param_groups'][0]['params_info']:
                avg_qi = sum(p['qi_strength'] for p in diag['param_groups'][0]['params_info']) / \
                        len(diag['param_groups'][0]['params_info'])
                avg_grad = sum(p['grad_strength'] for p in diag['param_groups'][0]['params_info']) / \
                          len(diag['param_groups'][0]['params_info'])
                print(f"Average Qi Strength: {avg_qi:.4f}")
                print(f"Average Grad Strength: {avg_grad:.4f}")
            print("-"*70 + "\n")

    print("\n" + "="*70)
    print("修煉完成 | Cultivation Complete")
    print("="*70 + "\n")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['test_loss'], label='Test Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('損失曲線 | Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['test_acc'], label='Test Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('準確率曲線 | Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Phase progression
    phases = history['phase']
    phase_colors = {'Wood': 'green', 'Fire': 'red', 'Earth': 'brown',
                   'Metal': 'gray', 'Water': 'blue'}
    colors = [phase_colors.get(p, 'black') for p in phases]
    axes[1, 0].scatter(range(len(phases)), [0]*len(phases), c=colors, s=100, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_yticks([])
    axes[1, 0].set_title('五行相 | Wu Xing Phases')
    axes[1, 0].grid(True, alpha=0.3)

    # Final summary
    axes[1, 1].axis('off')
    summary_text = f"""
    最終結果 | Final Results
    ========================

    Best Test Accuracy: {max(history['test_acc']):.2f}%
    Final Test Accuracy: {history['test_acc'][-1]:.2f}%
    Final Train Loss: {history['train_loss'][-1]:.4f}
    Final Test Loss: {history['test_loss'][-1]:.4f}

    Phases Traversed:
    {' → '.join(set(phases))}

    The Dao has guided this model
    to convergence with grace! 🌊
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig('mnist_dao_training.png', dpi=150, bbox_inches='tight')
    print(f"Results saved to: mnist_dao_training.png")

    # Final wisdom
    print_dao_wisdom()


if __name__ == "__main__":
    main()
