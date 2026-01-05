"""
Simple Example: TianJi Optimizer on a toy problem
===================================================

Demonstrates the TianJi Optimizer on a simple function minimization task.
"""

import torch
import torch.nn as nn
from tianji_optimizer import TianJiOptimizer
import matplotlib.pyplot as plt
import numpy as np


def rosenbrock(x, y):
    """Rosenbrock function - a classic optimization test function."""
    return (1 - x)**2 + 100 * (y - x**2)**2


def main():
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        天機優化器 - Simple Rosenbrock Example        ║
    ║                                                      ║
    ║  Minimizing the Rosenbrock function:                 ║
    ║  f(x,y) = (1-x)² + 100(y-x²)²                       ║
    ║  Minimum at (1, 1) with f(1,1) = 0                  ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # Initialize parameters (starting point)
    params = nn.Parameter(torch.tensor([0.0, 0.0], requires_grad=True))

    # Create optimizer
    optimizer = TianJiOptimizer(
        [params],
        lr=1e-2,
        beta_tian=0.9,
        beta_di=0.999,
        yin_yang_cycle=50,
    )

    # Track trajectory
    trajectory = []
    losses = []

    # Optimize
    print("Starting optimization from (0, 0)...\n")

    for step in range(500):
        optimizer.zero_grad()

        # Calculate loss
        loss = rosenbrock(params[0], params[1])

        # Backward pass
        loss.backward()

        # Optimization step (pass loss to observe Heaven's mechanism)
        optimizer.step(loss=loss.item())

        # Track progress
        trajectory.append(params.detach().clone().numpy())
        losses.append(loss.item())

        if step % 50 == 0:
            mechanism_state = optimizer.get_mechanism_state()
            print(f"Step {step:3d}: "
                  f"Position: ({params[0].item():.4f}, {params[1].item():.4f}), "
                  f"Loss: {loss.item():.6f}, "
                  f"Direction: {mechanism_state['tian_ji_direction']}")

    # Final result
    print(f"\n{'='*60}")
    print(f"Final Position: ({params[0].item():.6f}, {params[1].item():.6f})")
    print(f"Final Loss: {losses[-1]:.6e}")
    print(f"Distance from optimum (1, 1): {torch.dist(params, torch.tensor([1.0, 1.0])).item():.6f}")
    print(f"{'='*60}\n")

    # Visualize
    trajectory = np.array(trajectory)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Trajectory on Rosenbrock surface
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', linewidth=2, markersize=3, label='TianJi Path')
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='End')
    ax1.plot(1, 1, 'b*', markersize=15, label='Optimum')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Optimization Trajectory\n觀天之道 - Observing Heaven\'s Dao')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss curve
    ax2.semilogy(losses, linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Loss Convergence\n應機而動，萬化安 - Moving with the Mechanism')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tianji_rosenbrock.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved to: tianji_rosenbrock.png\n")

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║                   Observations                       ║
    ╚══════════════════════════════════════════════════════╝

    Notice how the optimizer:
    1. 🌊 Flows naturally toward the minimum (Wu Wei)
    2. ☯️  Oscillates between exploration and exploitation (Yin-Yang)
    3. 🎯 Smoothly converges without aggressive forcing
    4. 🐉 Adapts to the challenging landscape (Three Mechanisms)

    "無為而治，至靜至神"
    "Govern through non-action, utmost stillness, utmost spirit"
    """)


if __name__ == "__main__":
    main()
