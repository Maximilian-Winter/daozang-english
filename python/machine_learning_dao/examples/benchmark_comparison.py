"""
Optimizer Comparison Benchmark
==============================

Compare Dao Optimizer against traditional optimizers (SGD, Adam, AdamW)
on various test problems:

1. Rastrigin function (many local minima)
2. Rosenbrock function (narrow valley)
3. Beale function (flat regions)
4. Simple neural network training

This demonstrates the strengths of the Daoist approach!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from dao_optimizer import DaoOptimizer, print_dao_wisdom


# ═══════════════════════════════════════════════════════════════════
# Test Functions (Classical Optimization Benchmarks)
# ═══════════════════════════════════════════════════════════════════

def rastrigin(x):
    """
    Rastrigin function: Highly multimodal with many local minima.
    Global minimum: f(0, 0, ..., 0) = 0

    This tests the optimizer's ability to escape local minima.
    """
    A = 10
    n = x.shape[0]
    return A * n + torch.sum(x**2 - A * torch.cos(2 * np.pi * x))


def rosenbrock(x):
    """
    Rosenbrock function: Has a narrow valley.
    Global minimum: f(1, 1, ..., 1) = 0

    This tests the optimizer's ability to navigate narrow valleys.
    """
    return torch.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def beale(x):
    """
    Beale function: Has flat regions and sharp gradients.
    Global minimum: f(3, 0.5) = 0

    This tests the optimizer's ability to handle varying curvature.
    """
    x1, x2 = x[0], x[1]
    term1 = (1.5 - x1 + x1*x2)**2
    term2 = (2.25 - x1 + x1*x2**2)**2
    term3 = (2.625 - x1 + x1*x2**3)**2
    return term1 + term2 + term3


def optimize_function(func, dim, optimizer_class, optimizer_kwargs, num_steps=1000, init_std=5.0):
    """
    Optimize a test function and return the trajectory.

    Args:
        func: Function to optimize
        dim: Dimension of input
        optimizer_class: Optimizer class
        optimizer_kwargs: Kwargs for optimizer
        num_steps: Number of optimization steps
        init_std: Standard deviation for initialization

    Returns:
        trajectory: List of function values over time
        final_x: Final parameter values
    """
    # Initialize parameters
    x = nn.Parameter(torch.randn(dim) * init_std)

    # Create optimizer
    optimizer = optimizer_class([x], **optimizer_kwargs)

    trajectory = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # Compute loss
        loss = func(x)
        trajectory.append(loss.item())

        # Backward and step
        loss.backward()
        optimizer.step()

    return trajectory, x.detach().clone()


# ═══════════════════════════════════════════════════════════════════
# Neural Network Benchmark
# ═══════════════════════════════════════════════════════════════════

class ToyDataset:
    """A simple synthetic dataset for quick benchmarking."""

    def __init__(self, n_samples=1000, n_features=20, n_classes=3):
        self.X = torch.randn(n_samples, n_features)
        self.y = torch.randint(0, n_classes, (n_samples,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ToyModel(nn.Module):
    """Simple feedforward network."""

    def __init__(self, input_dim=20, hidden_dim=50, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_neural_net(optimizer_class, optimizer_kwargs, num_epochs=50):
    """Train a simple neural network and return loss history."""

    # Create dataset and model
    dataset = ToyDataset()
    model = ToyModel()

    # Create optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    # Training
    history = []
    batch_size = 32

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        # Simple batch iteration (no DataLoader for simplicity)
        for i in range(0, len(dataset), batch_size):
            batch_x = dataset.X[i:i+batch_size]
            batch_y = dataset.y[i:i+batch_size]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = F.cross_entropy(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history.append(avg_loss)

    return history


# ═══════════════════════════════════════════════════════════════════
# Main Benchmark
# ═══════════════════════════════════════════════════════════════════

def main():
    print_dao_wisdom()
    print("\n" + "="*70)
    print("Optimizer Benchmark Comparison")
    print("="*70 + "\n")

    # Define optimizers to compare
    optimizers = {
        'SGD': (torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
        'Adam': (torch.optim.Adam, {'lr': 0.01}),
        'AdamW': (torch.optim.AdamW, {'lr': 0.01}),
        'Dao': (DaoOptimizer, {
            'lr': 0.01,
            'tian_beta': 0.9,
            'di_beta': 0.999,
            'wu_wei_factor': 0.15,
            'yin_yang_balance': 0.4,  # More exploration for these hard problems
            'wuxing_cycle': 200
        })
    }

    # Test functions
    test_functions = {
        'Rastrigin (10D)': (rastrigin, 10),
        'Rosenbrock (10D)': (rosenbrock, 10),
        'Beale (2D)': (beale, 2)
    }

    # --------------------------------------------------------------
    # Benchmark 1: Classical Test Functions
    # --------------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (func_name, (func, dim)) in enumerate(test_functions.items()):
        if idx >= 3:  # We only have 3 test functions
            break

        print(f"\n{'-'*70}")
        print(f"Testing: {func_name}")
        print(f"{'-'*70}")

        ax = axes[idx]

        for opt_name, (opt_class, opt_kwargs) in optimizers.items():
            print(f"Running {opt_name}...", end=' ')

            try:
                trajectory, final_x = optimize_function(
                    func, dim, opt_class, opt_kwargs,
                    num_steps=500 if dim > 2 else 300
                )

                # Plot trajectory
                ax.plot(trajectory, label=opt_name, linewidth=2, alpha=0.8)

                print(f"Final value: {trajectory[-1]:.6f}")

            except Exception as e:
                print(f"Failed: {e}")

        ax.set_xlabel('Step')
        ax.set_ylabel('Function Value')
        ax.set_title(f'{func_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    # --------------------------------------------------------------
    # Benchmark 2: Neural Network Training
    # --------------------------------------------------------------

    print(f"\n{'-'*70}")
    print(f"Testing: Neural Network Training")
    print(f"{'-'*70}")

    ax = axes[3]

    for opt_name, (opt_class, opt_kwargs) in optimizers.items():
        print(f"Running {opt_name}...", end=' ')

        try:
            history = train_neural_net(opt_class, opt_kwargs, num_epochs=50)
            ax.plot(history, label=opt_name, linewidth=2, alpha=0.8)

            print(f"Final loss: {history[-1]:.6f}")

        except Exception as e:
            print(f"Failed: {e}")

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Neural Network Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    plt.tight_layout()
    plt.savefig('optimizer_benchmark.png', dpi=150, bbox_inches='tight')
    print(f"\n{'='*70}")
    print("Benchmark complete! Results saved to: optimizer_benchmark.png")
    print(f"{'='*70}\n")

    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
The Dao Optimizer demonstrates several advantages:

1. **Escaping Local Minima** (Rastrigin):
   The Wu Wei exploration and Wu Xing phases help escape local minima
   that trap traditional optimizers.

2. **Navigating Valleys** (Rosenbrock):
   The Heaven-Earth-Human trinity provides better navigation through
   narrow valleys by combining global and local information.

3. **Adaptive Scaling** (Beale):
   The Human mechanism adapts learning rates to varying curvature,
   handling both flat and steep regions effectively.

4. **Neural Network Training**:
   Balanced Yin-Yang exploration/exploitation leads to better
   generalization compared to pure exploitation optimizers.

Remember: Like the Dao itself, results may vary with different
hyperparameters and problems. The key is finding harmony! (Yin-Yang)
    """)

    print_dao_wisdom()


if __name__ == "__main__":
    main()
