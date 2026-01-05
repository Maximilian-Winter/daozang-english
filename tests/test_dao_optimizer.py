"""
Unit Tests for Dao Optimizer
============================

Test suite ensuring the optimizer behaves correctly and embodies
Daoist principles properly.

Run with: pytest test_dao_optimizer.py -v
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from dao_optimizer import DaoOptimizer, DaoScheduler, diagnose_qi_flow


class TestDaoOptimizerBasics:
    """Test basic functionality and initialization."""

    def test_initialization(self):
        """Test that optimizer initializes correctly."""
        params = [nn.Parameter(torch.randn(10, 10))]
        optimizer = DaoOptimizer(params, lr=1e-3)

        assert optimizer.defaults['lr'] == 1e-3
        assert optimizer.defaults['tian_beta'] == 0.9
        assert optimizer.defaults['di_beta'] == 0.999
        assert optimizer.global_step == 0

    def test_invalid_lr(self):
        """Test that invalid learning rates raise errors."""
        params = [nn.Parameter(torch.randn(10, 10))]

        with pytest.raises(ValueError):
            DaoOptimizer(params, lr=-1.0)

    def test_invalid_beta(self):
        """Test that invalid beta coefficients raise errors."""
        params = [nn.Parameter(torch.randn(10, 10))]

        with pytest.raises(ValueError):
            DaoOptimizer(params, tian_beta=1.5)

        with pytest.raises(ValueError):
            DaoOptimizer(params, di_beta=-0.1)

    def test_state_initialization(self):
        """Test that optimizer state is initialized on first step."""
        param = nn.Parameter(torch.randn(5, 5))
        param.grad = torch.randn(5, 5)

        optimizer = DaoOptimizer([param], lr=1e-3)
        optimizer.step()

        state = optimizer.state[param]
        assert 'step' in state
        assert 'tian_m' in state  # Heaven's momentum
        assert 'di_v' in state    # Earth's second moment
        assert state['step'] == 1


class TestWuXingPhases:
    """Test the Five Elements (Wu Xing) cycle."""

    def test_phase_progression(self):
        """Test that phases progress correctly."""
        params = [nn.Parameter(torch.randn(10, 10))]
        optimizer = DaoOptimizer(params, lr=1e-3, wuxing_cycle=500)

        # Test phase names
        phases = []
        for _ in range(5):
            phase, _ = optimizer._get_wuxing_phase(
                optimizer.global_step,
                optimizer.param_groups[0]['wuxing_cycle']
            )
            phases.append(phase)
            optimizer.global_step += 100

        # Should cycle through: Wood, Fire, Earth, Metal, Water
        assert phases == ['Wood', 'Fire', 'Earth', 'Metal', 'Water']

    def test_phase_progress(self):
        """Test that phase progress is calculated correctly."""
        params = [nn.Parameter(torch.randn(10, 10))]
        optimizer = DaoOptimizer(params, lr=1e-3, wuxing_cycle=1000)

        # At step 0, should be start of Wood phase
        phase, progress = optimizer._get_wuxing_phase(0, 1000)
        assert phase == 'Wood'
        assert progress == 0.0

        # At step 100 (middle of Wood phase), progress should be 0.5
        phase, progress = optimizer._get_wuxing_phase(100, 1000)
        assert phase == 'Wood'
        assert abs(progress - 0.5) < 0.01

    def test_cycle_completion(self):
        """Test that cycle completes and restarts."""
        params = [nn.Parameter(torch.randn(10, 10))]
        optimizer = DaoOptimizer(params, lr=1e-3, wuxing_cycle=100)

        # At step 0 and step 100, should both be Wood
        phase0, _ = optimizer._get_wuxing_phase(0, 100)
        phase100, _ = optimizer._get_wuxing_phase(100, 100)

        assert phase0 == phase100 == 'Wood'


class TestYinYangBalance:
    """Test the Yin-Yang balance mechanism."""

    def test_extreme_yin(self):
        """Test behavior with extreme Yin (pure exploitation)."""
        param = nn.Parameter(torch.ones(5, 5))
        param.grad = torch.ones(5, 5)

        # Pure Yin: all exploitation, no exploration
        optimizer = DaoOptimizer([param], lr=1e-2, yin_yang_balance=1.0)
        initial_param = param.clone()

        optimizer.step()

        # Should move in gradient direction
        assert torch.all(param < initial_param)

    def test_extreme_yang(self):
        """Test behavior with extreme Yang (pure exploration)."""
        param = nn.Parameter(torch.ones(5, 5))
        param.grad = torch.ones(5, 5)

        # Pure Yang: all exploration
        optimizer = DaoOptimizer([param], lr=1e-2, yin_yang_balance=0.0)
        initial_param = param.clone()

        optimizer.step()

        # Should still move (via momentum)
        assert not torch.all(param == initial_param)


class TestQiFlow:
    """Test the Qi (vital energy) flow mechanism."""

    def test_qi_computation(self):
        """Test that Qi is computed correctly."""
        param = nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn(10, 10)

        optimizer = DaoOptimizer([param], lr=1e-3, adaptive_qi=True)

        # Take a few steps to build up momentum
        for _ in range(5):
            param.grad = torch.randn(10, 10)
            optimizer.step()

        state = optimizer.state[param]
        assert 'tian_m' in state
        assert torch.any(state['tian_m'] != 0)  # Momentum should be non-zero

    def test_qi_adaptation(self):
        """Test that Qi adapts to gradient alignment."""
        param = nn.Parameter(torch.ones(5, 5) * 10.0)

        optimizer = DaoOptimizer([param], lr=1e-2, adaptive_qi=True)

        # Consistent gradient direction (good alignment)
        for _ in range(10):
            param.grad = torch.ones(5, 5)
            optimizer.step()

        state1 = optimizer.state[param]
        qi_norm_aligned = state1['tian_m'].norm()

        # Reset
        param2 = nn.Parameter(torch.ones(5, 5) * 10.0)
        optimizer2 = DaoOptimizer([param2], lr=1e-2, adaptive_qi=True)

        # Inconsistent gradient direction (poor alignment)
        for i in range(10):
            param2.grad = torch.ones(5, 5) * ((-1) ** i)  # Alternating direction
            optimizer2.step()

        state2 = optimizer2.state[param2]
        qi_norm_conflicted = state2['tian_m'].norm()

        # With good alignment, Qi should flow more strongly
        assert qi_norm_aligned > qi_norm_conflicted


class TestConvergence:
    """Test that optimizer actually converges on simple problems."""

    def test_quadratic_convergence(self):
        """Test convergence on simple quadratic: f(x) = x^2."""
        param = nn.Parameter(torch.tensor([5.0]))
        optimizer = DaoOptimizer([param], lr=1e-1)

        for _ in range(100):
            optimizer.zero_grad()
            loss = param ** 2
            loss.backward()
            optimizer.step()

        # Should converge close to 0
        assert abs(param.item()) < 0.1

    def test_neural_net_convergence(self):
        """Test convergence on simple neural network."""
        # Simple linear regression problem
        X = torch.randn(100, 5)
        y = X @ torch.randn(5, 1) + torch.randn(100, 1) * 0.1

        model = nn.Linear(5, 1)
        optimizer = DaoOptimizer(model.parameters(), lr=1e-2)

        initial_loss = None
        final_loss = None

        for epoch in range(100):
            optimizer.zero_grad()
            pred = model(X)
            loss = ((pred - y) ** 2).mean()

            if epoch == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.1


class TestScheduler:
    """Test the Dao Scheduler (seasonal learning rate)."""

    def test_scheduler_initialization(self):
        """Test scheduler initializes correctly."""
        params = [nn.Parameter(torch.randn(10, 10))]
        optimizer = DaoOptimizer(params, lr=1e-3)
        scheduler = DaoScheduler(optimizer, total_steps=1000)

        assert scheduler.total_steps == 1000
        assert scheduler.current_step == 0

    def test_warmup_phase(self):
        """Test that warmup increases learning rate."""
        params = [nn.Parameter(torch.randn(10, 10))]
        optimizer = DaoOptimizer(params, lr=1e-3)
        scheduler = DaoScheduler(optimizer, total_steps=1000, warmup_steps=100)

        initial_lr = optimizer.param_groups[0]['lr']

        # Step through warmup
        for _ in range(50):
            scheduler.step()

        mid_warmup_lr = optimizer.param_groups[0]['lr']

        # Learning rate should increase during warmup
        assert mid_warmup_lr > initial_lr

    def test_seasonal_cycle(self):
        """Test that learning rate follows seasonal cycle."""
        params = [nn.Parameter(torch.randn(10, 10))]
        optimizer = DaoOptimizer(params, lr=1.0)  # Base LR = 1 for easy testing
        scheduler = DaoScheduler(optimizer, total_steps=1000, warmup_steps=100)

        lrs = []
        for _ in range(1000):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # After warmup, LR should decrease
        assert lrs[-1] < lrs[200]


class TestDiagnostics:
    """Test diagnostic utilities."""

    def test_diagnose_qi_flow(self):
        """Test Qi flow diagnostics."""
        param = nn.Parameter(torch.randn(5, 5))
        param.grad = torch.randn(5, 5)

        optimizer = DaoOptimizer([param], lr=1e-3)
        optimizer.step()

        diag = diagnose_qi_flow(optimizer)

        assert 'phase' in diag
        assert 'phase_progress' in diag
        assert 'step' in diag
        assert 'param_groups' in diag
        assert len(diag['param_groups']) > 0

    def test_get_current_phase(self):
        """Test getting current Wu Xing phase."""
        params = [nn.Parameter(torch.randn(10, 10))]
        optimizer = DaoOptimizer(params, lr=1e-3)

        phase, progress = optimizer.get_current_phase()

        assert phase in ['Wood', 'Fire', 'Earth', 'Metal', 'Water']
        assert 0 <= progress <= 1


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_zero_gradient(self):
        """Test behavior with zero gradients."""
        param = nn.Parameter(torch.randn(5, 5))
        param.grad = torch.zeros(5, 5)

        optimizer = DaoOptimizer([param], lr=1e-3)
        initial_param = param.clone()

        optimizer.step()

        # Should not move with zero gradient
        assert torch.allclose(param, initial_param)

    def test_none_gradient(self):
        """Test behavior when gradient is None."""
        param = nn.Parameter(torch.randn(5, 5))
        param.grad = None

        optimizer = DaoOptimizer([param], lr=1e-3)
        initial_param = param.clone()

        optimizer.step()

        # Should not crash, and param should not change
        assert torch.allclose(param, initial_param)

    def test_multiple_param_groups(self):
        """Test with multiple parameter groups."""
        params1 = [nn.Parameter(torch.randn(5, 5))]
        params2 = [nn.Parameter(torch.randn(3, 3))]

        optimizer = DaoOptimizer([
            {'params': params1, 'lr': 1e-3},
            {'params': params2, 'lr': 1e-2}
        ])

        # Set gradients
        params1[0].grad = torch.randn(5, 5)
        params2[0].grad = torch.randn(3, 3)

        optimizer.step()

        # Both groups should have been updated
        assert optimizer.state[params1[0]]['step'] == 1
        assert optimizer.state[params2[0]]['step'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
