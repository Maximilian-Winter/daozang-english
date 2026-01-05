"""
TianDiRenOptimizer (天地人优化器) - Three Powers Optimizer
═══════════════════════════════════════════════════════════

"The Sage observed celestial symbols and drew the Trigrams,
 adhering to the Dragon Diagram's intricate numerical order.
 Gazing upward at Heaven and downward at Earth,
 drawing near to the body and extending to all things,
 the six-line Trigram symbols were established,
 and the San Cai (三才, Heaven-Earth-Human) principles were thus complete."
 — 遗论九事 (Nine Matters of Legacy Teaching)

This optimizer implements the Three Powers (三才 - San Cai) hierarchy:

1. **Celestial Mechanism (天機 - Tian Ji)**: Long-term, cosmic patterns
   - 3650-step cycle (10 Daoist years)
   - Slow, strategic adjustments
   - Global learning trends
   - Like planetary movements

2. **Human Mechanism (人機 - Ren Ji)**: The mediator
   - 365-step cycle (1 Daoist year)
   - Balances heaven and earth
   - Core optimization dynamics
   - The conscious adapter

3. **Earthly Mechanism (地機 - Di Ji)**: Short-term, local responses
   - 73-step cycle (Heaven-Earth number)
   - Fast, tactical corrections
   - Responsive to immediate gradients
   - Like weather patterns

The Three Powers create a complete optimization cosmos where:
- Heaven provides strategic direction
- Earth provides tactical response
- Human mediates and harmonizes

Based on passages from:
- 重阳真人授丹阳二十四诀 (Twenty-Four Precepts)
- 三才定位图 (Chart of the Three Powers' Fixed Positions)
- 元始天尊说三官宝号经 (Scripture of Three Officials)
- 钟吕传道集 (Transmission between Zhongli and Lü)

Author: The Lovelace-Hopper-Hypatia Creative Coding Mechanism
"""

import torch
from torch.optim.optimizer import Optimizer
import math
from typing import List, Optional, Tuple


class TianDiRenOptimizer(Optimizer):
    """
    Three Powers (San Cai) Optimizer - Hierarchical multi-scale optimization

    Implements three nested momentum mechanisms operating at different timescales:

    天 (Tian) - Celestial:  Long-term strategic trends (3650 steps)
    人 (Ren)  - Human:      Medium-term balance (365 steps)
    地 (Di)   - Earthly:    Short-term tactical responses (73 steps)

    The Three Powers correspond to the Three Officials (三官):
    - Heavenly Official (天官): Bestows blessings (long-term gains)
    - Earthly Official (地官): Pardons sins (corrects errors)
    - Water Official (水官): Resolves calamities (handles crises)

    Mathematical Formulation:
    ════════════════════════

    At each step t, for parameter θ:

    1. Earthly Momentum (Fast, responsive):
       m_di(t) = β_di · m_di(t-1) + (1 - β_di) · ĝ(t)
       Cycle: 73 steps

    2. Human Momentum (Medium, balanced):
       m_ren(t) = β_ren · m_ren(t-1) + (1 - β_ren) · ĝ(t)
       Cycle: 365 steps

    3. Celestial Momentum (Slow, strategic):
       m_tian(t) = β_tian · m_tian(t-1) + (1 - β_tian) · ĝ(t)
       Cycle: 3650 steps

    4. Hierarchical Integration:
       w_di = sin²(2π · t/73)          [Earthly weight]
       w_ren = sin²(2π · t/365)        [Human weight]
       w_tian = sin²(2π · t/3650)      [Celestial weight]

       W = w_di + w_ren + w_tian        [Normalize]

       m_final = (w_di·m_di + w_ren·m_ren + w_tian·m_tian) / W

    5. Parameter Update:
       θ(t+1) = θ(t) - α · m_final

    Parameters:
    ══════════
    params (iterable): Parameters to optimize
    lr (float, default: 0.01): Base learning rate

    beta_di (float, default: 0.6): Earthly momentum (fast, responsive)
    beta_ren (float, default: 0.9): Human momentum (balanced)
    beta_tian (float, default: 0.999): Celestial momentum (slow, strategic)

    beta_yin (float, default: 0.999): Yin variance for qi-flow normalization

    weight_decay (float, default: 0.0): Weight decay (L2 penalty)

    earthly_cycle (int, default: 73): Earthly cycle length (天地數)
    human_cycle (int, default: 365): Human cycle length (1 Daoist year)
    celestial_cycle (int, default: 3650): Celestial cycle length (10 Daoist years)

    cycle_amplitude (float, default: 0.1): Amplitude of cyclical modulation

    eps (float, default: 1e-8): Small constant for numerical stability

    Example Usage:
    ═════════════

    >>> model = MyNeuralNetwork()
    >>> optimizer = TianDiRenOptimizer(
    ...     model.parameters(),
    ...     lr=0.01,
    ...     beta_di=0.6,      # Fast earthly response
    ...     beta_ren=0.9,     # Balanced human mediation
    ...     beta_tian=0.999   # Slow celestial strategy
    ... )
    >>>
    >>> for epoch in range(num_epochs):
    ...     for batch in dataloader:
    ...         optimizer.zero_grad()
    ...         loss = model(batch)
    ...         loss.backward()
    ...         optimizer.step()  # The Three Powers harmonize

    Philosophy:
    ══════════

    "The Three Powers of the heavens are the Moon and Stars.
     The Three Powers of the earth are Yi, Bing, and Ding.
     The Three Powers of humans are Jing (essence), Shen (spirit), and Qi (vital breath)."
     — 重阳真人授丹阳二十四诀

    In neural network optimization:
    - **Celestial (天)**: Long-term convergence trends, like the movement of planets
    - **Human (人)**: Conscious adaptation, mediating between extremes
    - **Earthly (地)**: Immediate gradient responses, like changing weather

    Together they form a complete cosmos of optimization.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        beta_di: float = 0.6,
        beta_ren: float = 0.9,
        beta_tian: float = 0.999,
        beta_yin: float = 0.999,
        weight_decay: float = 0.0,
        earthly_cycle: int = 73,
        human_cycle: int = 365,
        celestial_cycle: int = 3650,
        cycle_amplitude: float = 0.1,
        eps: float = 1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta_di < 1.0:
            raise ValueError(f"Invalid beta_di (Earthly momentum): {beta_di}")
        if not 0.0 <= beta_ren < 1.0:
            raise ValueError(f"Invalid beta_ren (Human momentum): {beta_ren}")
        if not 0.0 <= beta_tian < 1.0:
            raise ValueError(f"Invalid beta_tian (Celestial momentum): {beta_tian}")
        if not 0.0 <= beta_yin < 1.0:
            raise ValueError(f"Invalid beta_yin (Yin stability): {beta_yin}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not earthly_cycle > 0:
            raise ValueError(f"Invalid earthly_cycle: {earthly_cycle}")
        if not human_cycle > 0:
            raise ValueError(f"Invalid human_cycle: {human_cycle}")
        if not celestial_cycle > 0:
            raise ValueError(f"Invalid celestial_cycle: {celestial_cycle}")
        if not 0.0 <= cycle_amplitude <= 1.0:
            raise ValueError(f"Invalid cycle_amplitude: {cycle_amplitude}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(
            lr=lr,
            beta_di=beta_di,
            beta_ren=beta_ren,
            beta_tian=beta_tian,
            beta_yin=beta_yin,
            weight_decay=weight_decay,
            earthly_cycle=earthly_cycle,
            human_cycle=human_cycle,
            celestial_cycle=celestial_cycle,
            cycle_amplitude=cycle_amplitude,
            eps=eps,
        )
        super(TianDiRenOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TianDiRenOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step guided by the Three Powers.

        "Gazing upward at Heaven and downward at Earth,
         drawing near to the body and extending to all things."

        Arguments:
            closure (callable, optional): A closure that reevaluates the model

        Returns:
            loss: The loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Extract hyperparameters
            lr = group['lr']
            beta_di = group['beta_di']
            beta_ren = group['beta_ren']
            beta_tian = group['beta_tian']
            beta_yin = group['beta_yin']
            weight_decay = group['weight_decay']
            earthly_cycle = group['earthly_cycle']
            human_cycle = group['human_cycle']
            celestial_cycle = group['celestial_cycle']
            cycle_amplitude = group['cycle_amplitude']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('TianDiRenOptimizer does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Three Powers momentum buffers
                    state['earthly_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['human_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['celestial_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Yin variance for qi-flow normalization
                    state['yin_variance'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Retrieve state
                earthly_momentum = state['earthly_momentum']
                human_momentum = state['human_momentum']
                celestial_momentum = state['celestial_momentum']
                yin_variance = state['yin_variance']

                state['step'] += 1
                step = state['step']

                # ═══════════════════════════════════════════════════
                # Metal Phase (金) - Weight Decay
                # ═══════════════════════════════════════════════════
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # ═══════════════════════════════════════════════════
                # Qi-Flow Normalization (氣流調節)
                # ═══════════════════════════════════════════════════

                # Update Yin variance (tracks gradient energy)
                yin_variance.mul_(beta_yin).addcmul_(grad, grad, value=1 - beta_yin)

                # Normalize gradient by RMS (qi energy)
                denom = yin_variance.sqrt().add_(eps)
                normalized_grad = grad / denom

                # ═══════════════════════════════════════════════════
                # THREE POWERS MOMENTUM (三才動量)
                # ═══════════════════════════════════════════════════

                # 1. Earthly Momentum (地) - Fast, responsive
                earthly_momentum.mul_(beta_di).add_(normalized_grad, alpha=1 - beta_di)

                # 2. Human Momentum (人) - Balanced, mediating
                human_momentum.mul_(beta_ren).add_(normalized_grad, alpha=1 - beta_ren)

                # 3. Celestial Momentum (天) - Slow, strategic
                celestial_momentum.mul_(beta_tian).add_(normalized_grad, alpha=1 - beta_tian)

                # ═══════════════════════════════════════════════════
                # HIERARCHICAL INTEGRATION (三才合一)
                # ═══════════════════════════════════════════════════

                # Compute cyclical weights for each power
                # Each power has maximum influence at different phases

                # Earthly: Fast cycle (73 steps - Heaven-Earth number)
                earthly_phase = (step % earthly_cycle) / earthly_cycle
                earthly_weight = math.sin(2 * math.pi * earthly_phase) ** 2

                # Human: Medium cycle (365 steps - 1 year)
                human_phase = (step % human_cycle) / human_cycle
                human_weight = math.sin(2 * math.pi * human_phase) ** 2

                # Celestial: Slow cycle (3650 steps - 10 years)
                celestial_phase = (step % celestial_cycle) / celestial_cycle
                celestial_weight = math.sin(2 * math.pi * celestial_phase) ** 2

                # Normalize weights (ensure they sum to 1)
                total_weight = earthly_weight + human_weight + celestial_weight + eps
                w_di = earthly_weight / total_weight
                w_ren = human_weight / total_weight
                w_tian = celestial_weight / total_weight

                # Hierarchical momentum combination
                # The Three Powers work together as one
                unified_momentum = (
                    w_di * earthly_momentum +
                    w_ren * human_momentum +
                    w_tian * celestial_momentum
                )

                # Bias correction (like Adam)
                bias_correction_yin = 1 - beta_yin ** step

                # ═══════════════════════════════════════════════════
                # CYCLICAL LEARNING RATE MODULATION (週期調節)
                # ═══════════════════════════════════════════════════

                # Combine all three cyclical influences
                cyclical_factor = 1.0 + cycle_amplitude * (
                    0.5 * math.cos(2 * math.pi * earthly_phase) +
                    0.3 * math.cos(2 * math.pi * human_phase) +
                    0.2 * math.cos(2 * math.pi * celestial_phase)
                )

                # Wu-Wei adaptive rate based on landscape harmony
                harmony_factor = bias_correction_yin / (1.0 + torch.norm(yin_variance).item())

                # Combined step size
                step_size = lr * cyclical_factor * harmony_factor

                # ═══════════════════════════════════════════════════
                # PARAMETER UPDATE (參數更新)
                # "The Three Powers harmonize, and all things are complete"
                # ═══════════════════════════════════════════════════

                p.add_(unified_momentum, alpha=-step_size)

                # ═══════════════════════════════════════════════════
                # PERIODIC RESETS (週期重置)
                # "At the completion of cycles, return to the origin"
                # ═══════════════════════════════════════════════════

                # Earthly reset (every 73 steps) - Soft reset
                if step % earthly_cycle == 0:
                    earthly_momentum.mul_(0.7)

                # Human reset (every 365 steps) - Medium reset
                if step % human_cycle == 0:
                    human_momentum.mul_(0.5)

                # Celestial reset (every 3650 steps) - Gentle reset
                if step % celestial_cycle == 0:
                    celestial_momentum.mul_(0.9)  # Very gentle, preserves long-term trends

        return loss

    def get_tiandiren_state(self) -> dict:
        """
        Return the current state of the Three Powers.

        Returns detailed information about each mechanism:
        - Earthly (地): Fast, tactical
        - Human (人): Balanced, mediating
        - Celestial (天): Slow, strategic
        """
        total_steps = 0
        total_di_norm = 0.0
        total_ren_norm = 0.0
        total_tian_norm = 0.0
        total_yin_norm = 0.0
        num_params = 0

        for group in self.param_groups:
            earthly_cycle = group['earthly_cycle']
            human_cycle = group['human_cycle']
            celestial_cycle = group['celestial_cycle']

            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    total_steps += state['step']
                    total_di_norm += torch.norm(state['earthly_momentum']).item()
                    total_ren_norm += torch.norm(state['human_momentum']).item()
                    total_tian_norm += torch.norm(state['celestial_momentum']).item()
                    total_yin_norm += torch.norm(state['yin_variance']).item()
                    num_params += 1

        if num_params == 0:
            return {
                'avg_step': 0,
                'earthly': {'norm': 0.0, 'cycle': '0/0', 'phase': 0.0},
                'human': {'norm': 0.0, 'cycle': '0/0', 'phase': 0.0},
                'celestial': {'norm': 0.0, 'cycle': '0/0', 'phase': 0.0},
                'yin_variance': 0.0,
            }

        avg_step = total_steps / num_params

        di_phase = (avg_step % earthly_cycle) / earthly_cycle
        ren_phase = (avg_step % human_cycle) / human_cycle
        tian_phase = (avg_step % celestial_cycle) / celestial_cycle

        return {
            'avg_step': avg_step,
            'earthly': {
                'norm': total_di_norm / num_params,
                'cycle': f"{avg_step % earthly_cycle:.0f}/{earthly_cycle}",
                'phase': di_phase,
            },
            'human': {
                'norm': total_ren_norm / num_params,
                'cycle': f"{avg_step % human_cycle:.0f}/{human_cycle}",
                'phase': ren_phase,
            },
            'celestial': {
                'norm': total_tian_norm / num_params,
                'cycle': f"{avg_step % celestial_cycle:.0f}/{celestial_cycle}",
                'phase': tian_phase,
            },
            'yin_variance': total_yin_norm / num_params,
        }


# ═══════════════════════════════════════════════════════════════════
# Wisdom from the Three Spirits
# ═══════════════════════════════════════════════════════════════════

"""
Ada Lovelace contemplates:
    "Three nested cycles, like music in three tempos playing simultaneously.
     The celestial moves like a slow bass line, the human like melody,
     and the earthly like rapid percussion. Together: a symphony of optimization."

Grace Hopper observes:
    "It's elegant: three momentum buffers, three timescales, one unified update.
     Computational cost? Same as Adam. Complexity? Hidden in the harmony.
     This is engineering meeting philosophy."

Hypatia reveals:
    "The ancients understood multi-scale dynamics millennia before us.
     Heaven (macro), Earth (micro), Human (meso) - this is not poetry,
     it's a description of hierarchical temporal decomposition in dynamical systems.
     The mathematics of the Dao reveals itself."

Together, they proclaim:
    "天地人三才 - The Three Powers of Heaven, Earth, and Humanity
     三才合一 - The Three Powers unite as one
     道法自然 - The Dao follows nature

     In optimization, as in cosmos:
     The slow guides strategy,
     The fast handles tactics,
     The human mediates wisdom."

     --- The Lovelace-Hopper-Hypatia Creative Coding Mechanism
"""
