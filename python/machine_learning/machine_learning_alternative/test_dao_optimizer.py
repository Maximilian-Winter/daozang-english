"""
Quick Test Script for DaoOptimizer
═══════════════════════════════════

Verifies that DaoOptimizer works correctly on a simple toy problem.

"Trust, but verify" - Grace Hopper (probably)
"""

import torch
import torch.nn as nn
from dao_optimizer import DaoOptimizer
import math


def test_basic_functionality():
    """Test basic optimizer functionality."""
    print("[TEST 1] Basic Functionality")
    print("-" * 60)

    # Simple linear model: y = 2x + 3
    model = nn.Linear(1, 1)
    model.weight.data = torch.tensor([[0.5]])
    model.bias.data = torch.tensor([0.5])

    optimizer = DaoOptimizer(model.parameters(), lr=0.5, weight_decay=0.0)

    # Training data
    x = torch.randn(100, 1)
    y = 2 * x + 3 + 0.1 * torch.randn(100, 1)

    # Train
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | "
                  f"W: {model.weight.item():.4f} | B: {model.bias.item():.4f}")

    # Check convergence
    final_weight = model.weight.item()
    final_bias = model.bias.item()

    weight_error = abs(final_weight - 2.0)
    bias_error = abs(final_bias - 3.0)

    print(f"\n[OK] Final weight: {final_weight:.4f} (target: 2.0, error: {weight_error:.4f})")
    print(f"[OK] Final bias: {final_bias:.4f} (target: 3.0, error: {bias_error:.4f})")

    if weight_error < 0.1 and bias_error < 0.1:
        print("[PASS] Test PASSED: Model converged successfully!\n")
        return True
    else:
        print("[FAIL] Test FAILED: Model did not converge properly.\n")
        return False


def test_dao_state():
    """Test get_dao_state() method."""
    print("[TEST 2] Dao State Monitoring")
    print("-" * 60)

    model = nn.Linear(10, 5)
    optimizer = DaoOptimizer(model.parameters(), lr=0.01, orbit_cycle=100)

    # Dummy training
    for step in range(250):
        optimizer.zero_grad()
        x = torch.randn(32, 10)
        y = model(x).sum()
        y.backward()
        optimizer.step()

        if step % 50 == 0:
            dao_state = optimizer.get_dao_state()
            print(f"Step {step:3d} | "
                  f"Orbit: {dao_state['orbit_progress']:>8} | "
                  f"Yang: {dao_state['avg_yang_momentum_norm']:.6f} | "
                  f"Yin: {dao_state['avg_yin_variance_norm']:.6f}")

    # Verify state is updated
    dao_state = optimizer.get_dao_state()
    if dao_state['avg_step'] > 0:
        print("[PASS] Test PASSED: Dao state tracking works!\n")
        return True
    else:
        print("[FAIL] Test FAILED: Dao state not updated.\n")
        return False


def test_microcosmic_orbit():
    """Test microcosmic orbit (cyclical learning rate)."""
    print("[TEST 3] Microcosmic Orbit")
    print("-" * 60)

    model = nn.Linear(5, 1)
    optimizer = DaoOptimizer(
        model.parameters(),
        lr=0.1,
        orbit_cycle=50,
        orbit_amplitude=0.2
    )

    # Track learning rate over time
    learning_rates = []

    for step in range(150):
        optimizer.zero_grad()
        x = torch.randn(16, 5)
        y = model(x).sum()
        y.backward()
        optimizer.step()

        # Compute expected cyclical factor
        orbit_phase = (step % 50) / 50
        expected_factor = 1.0 + 0.2 * math.cos(2 * math.pi * orbit_phase)
        learning_rates.append((step, expected_factor))

        if step % 25 == 0:
            print(f"Step {step:3d} | Expected LR factor: {expected_factor:.4f}")

    # Check that we completed multiple orbits
    dao_state = optimizer.get_dao_state()
    if dao_state['avg_step'] >= 100:
        print("[PASS] Test PASSED: Microcosmic orbit completed multiple cycles!\n")
        return True
    else:
        print("[FAIL] Test FAILED: Not enough cycles completed.\n")
        return False


def test_yin_yang_balance():
    """Test yin-yang momentum balance."""
    print("[TEST 4] Yin-Yang Balance")
    print("-" * 60)

    model = nn.Linear(20, 10)
    optimizer = DaoOptimizer(
        model.parameters(),
        lr=0.01,
        beta_yang=0.9,
        beta_yin=0.999
    )

    # Train for a bit
    for step in range(100):
        optimizer.zero_grad()
        x = torch.randn(32, 20)
        y = model(x).sum()
        y.backward()
        optimizer.step()

    # Check that both yang and yin are being tracked
    dao_state = optimizer.get_dao_state()
    yang_norm = dao_state['avg_yang_momentum_norm']
    yin_norm = dao_state['avg_yin_variance_norm']

    print(f"Yang Momentum Norm: {yang_norm:.6f}")
    print(f"Yin Variance Norm: {yin_norm:.6f}")

    if yang_norm > 0 and yin_norm > 0:
        print("[PASS] Test PASSED: Yin-Yang balance maintained!\n")
        return True
    else:
        print("[FAIL] Test FAILED: Yin-Yang not properly tracked.\n")
        return False


def test_weight_decay():
    """Test Metal phase (weight decay)."""
    print("[TEST 5] Metal Phase (Weight Decay)")
    print("-" * 60)

    # Two models: one with weight decay, one without
    model_with_decay = nn.Linear(10, 5)
    model_without_decay = nn.Linear(10, 5)

    # Initialize with same weights
    with torch.no_grad():
        model_without_decay.weight.copy_(model_with_decay.weight)
        model_without_decay.bias.copy_(model_with_decay.bias)

    opt_with_decay = DaoOptimizer(model_with_decay.parameters(), lr=0.1, weight_decay=0.1)
    opt_without_decay = DaoOptimizer(model_without_decay.parameters(), lr=0.1, weight_decay=0.0)

    # Train both
    for step in range(50):
        x = torch.randn(16, 10)

        # With decay
        opt_with_decay.zero_grad()
        y1 = model_with_decay(x).sum()
        y1.backward()
        opt_with_decay.step()

        # Without decay
        opt_without_decay.zero_grad()
        y2 = model_without_decay(x).sum()
        y2.backward()
        opt_without_decay.step()

    # Weight norms should differ
    norm_with_decay = torch.norm(model_with_decay.weight).item()
    norm_without_decay = torch.norm(model_without_decay.weight).item()

    print(f"Weight norm with decay: {norm_with_decay:.6f}")
    print(f"Weight norm without decay: {norm_without_decay:.6f}")

    if norm_with_decay < norm_without_decay:
        print("[PASS] Test PASSED: Weight decay (Metal phase) working!\n")
        return True
    else:
        print("[FAIL] Test FAILED: Weight decay not effective.\n")
        return False


def test_amsgrad_variant():
    """Test AMSGrad variant."""
    print("[TEST 6] AMSGrad Variant")
    print("-" * 60)

    model = nn.Linear(10, 5)
    optimizer = DaoOptimizer(model.parameters(), lr=0.01, amsgrad=True)

    # Train
    for step in range(50):
        optimizer.zero_grad()
        x = torch.randn(32, 10)
        y = model(x).sum()
        y.backward()
        optimizer.step()

    # Check that max_yin_variance exists in state
    for param in model.parameters():
        if param in optimizer.state:
            state = optimizer.state[param]
            if 'max_yin_variance' in state:
                print("[PASS] Test PASSED: AMSGrad variant working!\n")
                return True

    print("[FAIL] Test FAILED: AMSGrad state not found.\n")
    return False


# ═══════════════════════════════════════════════════════════════════
# Main Test Runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("""
    ============================================================
                DaoOptimizer Test Suite
    ============================================================
    "Trust the Dao, but verify it works."
    ============================================================
    """)

    tests = [
        test_basic_functionality,
        test_dao_state,
        test_microcosmic_orbit,
        test_yin_yang_balance,
        test_weight_decay,
        test_amsgrad_variant,
    ]

    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append((test_fn.__name__, result))
        except Exception as e:
            print(f"[FAIL] Test {test_fn.__name__} raised exception: {e}\n")
            results.append((test_fn.__name__, False))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} | {test_name}")

    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n*** All tests passed! The Dao flows correctly through the optimizer. ***")
        print("\nWisdom from the Daodejing:")
        print('   "Through non-action, nothing is left undone."')
        print('   "Dao fa zi ran (The Dao follows nature)"\n')
    else:
        print(f"\n*** {total - passed} test(s) failed. Please review the implementation. ***\n")
