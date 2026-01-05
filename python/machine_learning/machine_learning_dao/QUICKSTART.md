# 快速開始 | Quick Start Guide

Get started with the Dao Optimizer in 5 minutes! 🚀

## Installation

No installation needed! Just have PyTorch installed:

```bash
pip install torch torchvision
```

## Basic Usage

### 1. Import the Optimizer

```python
from dao_optimizer import DaoOptimizer

# That's it! No other dependencies needed.
```

### 2. Replace Your Current Optimizer

```python
# Before: Using Adam
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# After: Using Dao
optimizer = DaoOptimizer(model.parameters(), lr=1e-3)
```

### 3. Train Normally

```python
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()  # The Dao works its magic here!
```

That's it! The Dao Optimizer works as a drop-in replacement for any PyTorch optimizer.

## Common Use Cases

### Case 1: Training is Stuck? Try More Exploration

```python
optimizer = DaoOptimizer(
    model.parameters(),
    lr=1e-3,
    yin_yang_balance=0.3,  # More Yang (exploration)
    wu_wei_factor=0.2      # Higher non-forcing exploration
)
```

**Why**: More exploration helps escape local minima.

### Case 2: Need Faster Convergence?

```python
optimizer = DaoOptimizer(
    model.parameters(),
    lr=1e-3,
    yin_yang_balance=0.7,  # More Yin (exploitation)
    wu_wei_factor=0.05     # Less exploration, more focus
)
```

**Why**: More exploitation = faster convergence on known good direction.

### Case 3: Long Training Run?

```python
from dao_optimizer import DaoOptimizer, DaoScheduler

optimizer = DaoOptimizer(model.parameters(), lr=1e-2)

# Add seasonal learning rate schedule
scheduler = DaoScheduler(
    optimizer,
    total_steps=len(dataloader) * num_epochs,
    min_lr_factor=0.01  # LR will decay to 1% of initial
)

for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step the scheduler too!
```

**Why**: Seasonal learning rate mimics nature's rhythms - high energy early, refinement late.

## Monitoring & Debugging

### Check Current Phase

```python
phase, progress = optimizer.get_current_phase()
print(f"Currently in {phase} phase ({progress:.1%} complete)")

# Output: Currently in Fire phase (45.2% complete)
```

### Diagnose Qi Flow

```python
from dao_optimizer import diagnose_qi_flow

diagnostics = diagnose_qi_flow(optimizer)
print(f"Qi Strength: {diagnostics['param_groups'][0]['params_info'][0]['qi_strength']:.4f}")
```

**What is Qi Strength?**: Magnitude of momentum. High Qi = strong directional commitment.

### Get Dao Wisdom

```python
from dao_optimizer import print_dao_wisdom

# When feeling stuck, consult the Dao!
print_dao_wisdom()
```

## Hyperparameter Cheat Sheet

| What You Want | Hyperparameters | Example Values |
|---------------|----------------|----------------|
| **Fast convergence** | High Yin, low Wu Wei | `yin_yang_balance=0.7, wu_wei_factor=0.05` |
| **Escape local minima** | High Yang, high Wu Wei | `yin_yang_balance=0.3, wu_wei_factor=0.2` |
| **Best generalization** | Balanced, medium Wu Wei | `yin_yang_balance=0.5, wu_wei_factor=0.1` |
| **Long training** | Add seasonal scheduler | `DaoScheduler(optimizer, total_steps)` |
| **Faster phase changes** | Shorter Wu Xing cycle | `wuxing_cycle=500` |
| **Slower phase changes** | Longer Wu Xing cycle | `wuxing_cycle=2000` |

## Default Parameters (Good Starting Point)

```python
DaoOptimizer(
    params,
    lr=1e-3,                 # Like Adam's default
    tian_beta=0.9,           # Heaven's momentum (like β₁ in Adam)
    di_beta=0.999,           # Earth's second moment (like β₂ in Adam)
    wu_wei_factor=0.1,       # 10% exploration
    yin_yang_balance=0.5,    # Perfect balance
    adaptive_qi=True,        # Enable adaptive momentum
    wuxing_cycle=1000        # One full cycle per 1000 steps
)
```

## Examples

### Run MNIST Example

```bash
cd examples
python mnist_example.py
```

This will:
1. Download MNIST dataset
2. Train a simple neural network
3. Show Wu Xing phase progression
4. Generate training curves
5. Print Dao wisdom!

### Run Benchmarks

```bash
cd examples
python benchmark_comparison.py
```

Compare Dao Optimizer against SGD, Adam, and AdamW on:
- Rastrigin function (many local minima)
- Rosenbrock function (narrow valley)
- Beale function (varying curvature)
- Neural network training

## Understanding the Output

### During Training

```
Epoch 1 [0/60000 (0%)]     Loss: 2.3015    Acc: 10.16%    Phase: Wood (23%)
Epoch 1 [12800/60000 (21%)] Loss: 0.5234    Acc: 84.38%    Phase: Fire (67%)
```

- **Phase**: Current Wu Xing element
- **Percentage**: Progress through current phase

### Phases and What They Mean

| Phase | When | Characteristics | What's Happening |
|-------|------|----------------|------------------|
| **Wood** 🌱 | Steps 0-199 | Growth, exploration | Larger steps, trying new directions |
| **Fire** 🔥 | Steps 200-399 | High energy, momentum | Following strong gradients |
| **Earth** 🌍 | Steps 400-599 | Balance, stability | Trusting local landscape |
| **Metal** ⚙️ | Steps 600-799 | Refinement, precision | Smaller, careful steps |
| **Water** 💧 | Steps 800-999 | Adaptation, flow | Maximum Wu Wei, natural flow |

(Assuming `wuxing_cycle=1000`)

## Troubleshooting

### Training Loss Explodes

**Problem**: Loss goes to infinity or NaN.

**Solution**:
```python
# Reduce learning rate
optimizer = DaoOptimizer(params, lr=1e-4)  # Was 1e-3

# Or increase Yin (more conservative)
optimizer = DaoOptimizer(params, yin_yang_balance=0.8)
```

### Training Too Slow

**Problem**: Loss decreases but very slowly.

**Solution**:
```python
# Increase learning rate
optimizer = DaoOptimizer(params, lr=1e-2)  # Was 1e-3

# Or increase Yang (more momentum)
optimizer = DaoOptimizer(params, yin_yang_balance=0.3)
```

### Stuck in Local Minimum

**Problem**: Loss plateaus early.

**Solution**:
```python
# Increase exploration
optimizer = DaoOptimizer(
    params,
    yin_yang_balance=0.2,   # Much more Yang
    wu_wei_factor=0.3       # Much more Wu Wei
)
```

### Can't Match Adam's Performance

**Problem**: Dao Optimizer converges slower than Adam.

**Solution**:
```python
# Make it more Adam-like
optimizer = DaoOptimizer(
    params,
    lr=1e-3,
    yin_yang_balance=0.9,   # Mostly Yin (like Adam)
    wu_wei_factor=0.0,      # No Wu Wei
    wuxing_cycle=1000000    # Effectively disable Wu Xing
)
```

But then... why not just use Adam? 😄 The Dao's strength is in its *difference*!

## Philosophy Corner

> **道常無為而無不為**
> *The Dao constantly practices non-action, yet nothing is left undone.*
> — Dao De Jing, Chapter 37

The Dao Optimizer doesn't *force* convergence through aggressive gradient descent. Instead, it:

1. **Observes** the loss landscape from multiple levels (Heaven-Earth-Human)
2. **Balances** exploration and exploitation (Yin-Yang)
3. **Adapts** its strategy over time (Wu Xing cycles)
4. **Flows** naturally to minima (Wu Wei, like water)

Trust the process. Trust the Dao. 🌊

## Next Steps

1. ✅ Try it on your current project
2. 📖 Read [README_DAO_OPTIMIZER.md](README_DAO_OPTIMIZER.md) for details
3. 🧘 Study [PHILOSOPHY.md](PHILOSOPHY.md) for deep understanding
4. 🧪 Run [examples/benchmark_comparison.py](examples/benchmark_comparison.py)
5. 🧠 Explore the source code in [dao_optimizer.py](dao_optimizer.py)

## Need Help?

- 📚 **Full Documentation**: [README_DAO_OPTIMIZER.md](README_DAO_OPTIMIZER.md)
- 🧘 **Philosophy Guide**: [PHILOSOPHY.md](PHILOSOPHY.md)
- 🧪 **Examples**: [examples/](examples/)
- 🧬 **Tests**: [tests/test_dao_optimizer.py](tests/test_dao_optimizer.py)

---

**Remember**:

> **千里之行，始於足下**
> *A journey of a thousand miles begins with a single step.*
> — Dao De Jing, Chapter 64

Your first step: `optimizer = DaoOptimizer(model.parameters())`

May your gradients flow like water! 🌊✨
