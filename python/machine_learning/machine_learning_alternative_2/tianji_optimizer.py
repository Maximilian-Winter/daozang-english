"""
天機優化器 - TianJi Optimizer (Heaven's Mechanism Optimizer)
=============================================================

A PyTorch optimizer inspired by Daoist principles from the Daozang (道藏),
specifically the concepts of:
- 天機 (Tian Ji): Heaven's Mechanism - Global patterns
- 地機 (Di Ji): Earth's Mechanism - Local gradients
- 人機 (Ren Ji): Human/Heart Mechanism - Internal state
- 五賊 (Wu Zei): Five Thieves - Five sources of wisdom
- 無為 (Wu Wei): Non-action, natural flow
- 陰陽 (Yin-Yang): Balance of opposing forces

Based on sacred texts:
- 天機經 (Classic of Heaven's Mechanism)
- 黃帝陰符經 (Yellow Emperor's Yin Fu Jing)
- 化書 (Book of Transformations)

Philosophy:
-----------
"觀天之道，執天之行，盡矣"
"Observe the Dao of Heaven, grasp its operations—thus all is complete."

"應其機而動則萬化安"
"Moving in response to the moment, the ten thousand transformations are at peace."

"無為則無機，無機則至靜"
"Without action, there is no pivot; without pivot, there is utmost stillness."
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional, Callable
import math


class TianJiOptimizer(Optimizer):
    """
    天機優化器 - Heaven's Mechanism Optimizer

    An optimizer that unites three mechanisms (三機合一):
    1. Tian Ji (天機) - Celestial/Global mechanism: Long-term trends
    2. Di Ji (地機) - Earthly/Local mechanism: Current gradients
    3. Ren Ji (人機) - Heart mechanism: Internal optimizer state

    And employs the Five Thieves (五賊) to gather wisdom:
    1. Stealing Fate (盜命) - Loss trajectory
    2. Stealing Things (盜物) - Parameter characteristics
    3. Stealing Time (盜時) - Natural timing
    4. Stealing Merit (盜功) - Past successes (momentum)
    5. Stealing Numinous (盜神) - Emergent patterns

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (base qi flow rate) [default: 1e-3]
        beta_tian: Momentum for celestial mechanism (天機) [default: 0.9]
        beta_di: Momentum for earthly mechanism (地機) [default: 0.999]
        beta_ren: Dampening for heart mechanism (人機) [default: 0.1]
        wu_wei_threshold: Threshold for entering wu wei (stillness) state [default: 1e-8]
        yin_yang_cycle: Period of yin-yang oscillation [default: 100]
        epsilon: Small constant for numerical stability [default: 1e-8]
        weight_decay: L2 regularization (return to origin) [default: 0]

    Example:
        >>> optimizer = TianJiOptimizer(model.parameters(), lr=1e-3)
        >>> for epoch in range(num_epochs):
        >>>     loss = model(x)
        >>>     optimizer.zero_grad()
        >>>     loss.backward()
        >>>     optimizer.step(loss=loss.item())  # Pass loss for mechanism observation
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta_tian: float = 0.9,      # 天機 - Celestial momentum (long-term)
        beta_di: float = 0.999,       # 地機 - Earthly momentum (short-term variance)
        beta_ren: float = 0.1,        # 人機 - Heart dampening
        wu_wei_threshold: float = 1e-8,
        yin_yang_cycle: int = 100,
        epsilon: float = 1e-8,
        weight_decay: float = 0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta_tian < 1.0:
            raise ValueError(f"Invalid beta_tian parameter: {beta_tian}")
        if not 0.0 <= beta_di < 1.0:
            raise ValueError(f"Invalid beta_di parameter: {beta_di}")
        if not 0.0 <= beta_ren <= 1.0:
            raise ValueError(f"Invalid beta_ren parameter: {beta_ren}")
        if not 0.0 <= epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta_tian=beta_tian,
            beta_di=beta_di,
            beta_ren=beta_ren,
            wu_wei_threshold=wu_wei_threshold,
            yin_yang_cycle=yin_yang_cycle,
            epsilon=epsilon,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Global state for observing Heaven's mechanism
        self.state['global'] = {
            'step': 0,
            'loss_history': [],
            'tian_ji_direction': None,  # Celestial flow direction
            'yin_yang_phase': 0,  # Current phase in yin-yang cycle
        }

    def _observe_tianji(self, loss: Optional[float] = None):
        """
        觀天之道 - Observe Heaven's Mechanism

        Track global patterns in the optimization trajectory.
        This implements "Stealing Fate" (盜命) - learning from the loss trajectory.
        """
        global_state = self.state['global']

        if loss is not None:
            global_state['loss_history'].append(loss)
            # Keep only recent history (last 100 steps)
            if len(global_state['loss_history']) > 100:
                global_state['loss_history'].pop(0)

            # Detect celestial direction (improving or stagnating)
            if len(global_state['loss_history']) >= 10:
                recent_trend = sum(global_state['loss_history'][-5:]) / 5
                older_trend = sum(global_state['loss_history'][-10:-5]) / 5
                global_state['tian_ji_direction'] = 'descending' if recent_trend < older_trend else 'ascending'

    def _calculate_yinyang_factor(self, step: int, cycle: int) -> float:
        """
        陰陽調和 - Calculate Yin-Yang Balance Factor

        "Yin and yang summon one another, alternating as lord and minister"

        Oscillates between:
        - Yang phase (陽): Exploration, larger updates
        - Yin phase (陰): Exploitation, smaller updates, stillness

        Returns a factor between 0.5 (deep yin) and 1.5 (strong yang)
        """
        phase = (step % cycle) / cycle  # 0 to 1
        # Smooth oscillation using cosine
        # cos(0) = 1 (yang peak), cos(π) = -1 (yin trough)
        yinyang = 1.0 + 0.5 * math.cos(2 * math.pi * phase)
        return yinyang

    def _wu_wei_dampening(self, grad: torch.Tensor, variance: torch.Tensor, threshold: float) -> float:
        """
        無為而治 - Wu Wei Dampening

        "Without action, there is no pivot; without pivot, there is utmost stillness"

        When gradients become small (approaching natural rest), reduce intervention.
        This implements natural convergence without force.
        """
        # Calculate gradient magnitude
        grad_norm = torch.norm(grad)
        var_norm = torch.norm(variance)

        if grad_norm < threshold or var_norm < threshold:
            # Approaching stillness - dampen updates further
            return 0.5
        return 1.0

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None, loss: Optional[float] = None):
        """
        執天之行 - Execute Heaven's Operations

        Performs a single optimization step uniting the three mechanisms.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
            loss: Current loss value for observing Heaven's mechanism (optional but recommended)
        """
        loss_value = None
        if closure is not None:
            with torch.enable_grad():
                loss_value = closure()

        # Observe Heaven's mechanism (天機)
        if loss is not None:
            self._observe_tianji(loss)
        elif loss_value is not None:
            self._observe_tianji(loss_value.item())

        global_state = self.state['global']
        global_state['step'] += 1
        step = global_state['step']

        for group in self.param_groups:
            beta_tian = group['beta_tian']  # 天機 momentum
            beta_di = group['beta_di']      # 地機 momentum
            beta_ren = group['beta_ren']    # 人機 dampening
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']
            wu_wei_threshold = group['wu_wei_threshold']
            yin_yang_cycle = group['yin_yang_cycle']

            # Calculate Yin-Yang phase (陰陽調和)
            yinyang_factor = self._calculate_yinyang_factor(step, yin_yang_cycle)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay (return to origin - 歸根復命)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Initialize parameter state
                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['step'] = 0
                    # 天機 - Celestial momentum (first moment - direction)
                    param_state['tian_ji'] = torch.zeros_like(p)
                    # 地機 - Earthly variance (second moment - landscape)
                    param_state['di_ji'] = torch.zeros_like(p)
                    # 人機 - Heart state (internal wisdom)
                    param_state['ren_ji'] = torch.zeros_like(p)

                param_state['step'] += 1

                tian_ji = param_state['tian_ji']  # Celestial mechanism
                di_ji = param_state['di_ji']      # Earthly mechanism
                ren_ji = param_state['ren_ji']    # Heart mechanism

                # === 第一賊：盜命 (Steal Fate) ===
                # Update Celestial Mechanism (天機) - Long-term direction
                # "The sage observes the season and employs his talisman"
                tian_ji.mul_(beta_tian).add_(grad, alpha=1 - beta_tian)

                # === 第二賊：盜物 (Steal Things) ===
                # Update Earthly Mechanism (地機) - Local variance/terrain
                # "Observe changes in the earthly mechanism"
                di_ji.mul_(beta_di).addcmul_(grad, grad, value=1 - beta_di)

                # Bias correction for moments
                tian_ji_corrected = tian_ji / (1 - beta_tian ** param_state['step'])
                di_ji_corrected = di_ji / (1 - beta_di ** param_state['step'])

                # === 第三賊：盜時 (Steal Time) ===
                # Natural timing through Yin-Yang cycle
                adaptive_lr = group['lr'] * yinyang_factor

                # === 第四賊：盜功 (Steal Merit) ===
                # Leverage past success through momentum (already in tian_ji)

                # === 第五賊：盜神 (Steal Numinous) ===
                # Capture emergent patterns through variance-adapted steps
                # "Stir the mechanism, and all transformations settle"
                step_size = adaptive_lr / (di_ji_corrected.sqrt().add(epsilon))

                # === 無為調節 (Wu Wei Dampening) ===
                # "Moving in response to the moment, the ten thousand transformations are at peace"
                wu_wei_factor = self._wu_wei_dampening(grad, di_ji_corrected, wu_wei_threshold)

                # === 人機調和 (Heart Mechanism Integration) ===
                # The heart mediates between heaven and earth
                # "Heaven's nature is humanity; humanity's heart is the mechanism"
                ren_ji.mul_(1 - beta_ren).add_(tian_ji_corrected, alpha=beta_ren)

                # === 合一而動 (Unite and Move) ===
                # "When Heaven, Earth, and humanity unite in intent,
                #  all transformations find their foundation"
                unified_direction = ren_ji * step_size * wu_wei_factor

                # Apply update
                p.add_(unified_direction, alpha=-1)

        return loss_value

    def get_mechanism_state(self) -> dict:
        """
        Get current state of the three mechanisms for observation.

        Returns:
            Dictionary containing:
            - tian_ji_direction: Global optimization direction
            - yin_yang_phase: Current yin-yang phase
            - step: Current step number
            - loss_trend: Recent loss trend
        """
        global_state = self.state['global']
        loss_history = global_state['loss_history']

        return {
            'step': global_state['step'],
            'tian_ji_direction': global_state['tian_ji_direction'],
            'yin_yang_phase': global_state['yin_yang_phase'],
            'loss_trend': loss_history[-10:] if len(loss_history) >= 10 else loss_history,
            'in_wu_wei': len(loss_history) >= 2 and abs(loss_history[-1] - loss_history[-2]) < 1e-6,
        }

    def __repr__(self):
        return f"TianJiOptimizer(天機優化器)"
