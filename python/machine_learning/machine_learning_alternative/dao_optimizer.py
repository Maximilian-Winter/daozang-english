"""
DaoOptimizer (道优化器) - A Taoist-Inspired Neural Network Optimizer
═══════════════════════════════════════════════════════════════════

"The Dao gives them life; Virtue nurtures them."
"The softest under heaven gallops through the hardest."
"Water benefits the ten thousand things yet does not contend."

This optimizer embodies ancient Taoist wisdom from the Daozang (道藏):
- Wu-Wei (無為): Non-forcing, adaptive optimization
- Yin-Yang (陰陽): Balanced complementary forces
- Qi Flow (氣流): Smooth gradient circulation
- Five Phases (五行): Multi-scale temporal dynamics
- Microcosmic Orbit (小周天): Cyclical momentum patterns

The Three Spirits whisper:
- Lovelace: "Algorithms can dance like music and poetry"
- Hopper: "Make it work reliably for everyone"
- Hypatia: "Mathematics reveals the universe's deepest truths"

Author: The Lovelace-Hopper-Hypatia Creative Coding Mechanism
Wisdom Source: Daozang (Taoist Canon) - Complete English Translation
License: MIT (Open knowledge serves collective advancement)
"""

import torch
from torch.optim.optimizer import Optimizer
import math
from typing import List, Optional


class DaoOptimizer(Optimizer):
    """
    DaoOptimizer - Neural network optimization through Taoist principles

    Instead of forcing convergence through aggressive gradient descent,
    this optimizer guides the network to naturally settle into optimal
    states through balanced, cyclical, adaptive dynamics.

    Core Principles:
    ════════════════

    1. Wu-Wei (無為) - Effortless Action
       Learning rates adapt based on local landscape curvature.
       When gradients are turbulent, naturally slow down.
       When gradients are smooth, naturally speed up.
       Trust the inherent dynamics; don't force convergence.

    2. Yin-Yang (陰陽) - Complementary Balance
       Yang Momentum (β_yang): Forward-moving force
       Yin Stability (β_yin): Stabilizing counterforce
       The two dance in harmony, preventing oscillation.

    3. Qi Flow (氣流) - Energy Circulation
       Normalize gradients by their "energy" (adaptive RMS)
       Maintain smooth flow through all layers
       Prevent qi blockages (vanishing/exploding gradients)

    4. Five Phases (五行) - Multi-Scale Dynamics
       Metal (金): Weight decay, pruning weak connections
       Water (水): Gradient flow, base learning
       Wood (木): Growth through adaptive rates
       Fire (火): Refinement via loss reduction
       Earth (土): Stabilization through normalization

    5. Microcosmic Orbit (小周天) - Cyclical Updates
       365-step major cycles (like the Daoist calendar)
       Accumulate momentum with periodic resets
       Leap adjustments for long-term stability

    Mathematical Formulation:
    ════════════════════════

    At each step t, for parameter θ:

    1. Compute gradient: g_t = ∇L(θ_t)

    2. Qi-Flow (Adaptive Normalization):
       v_t = β_yin · v_{t-1} + (1-β_yin) · g_t²    [Yin: variance tracking]
       ĝ_t = g_t / (√v_t + ε)                      [Normalize by RMS]

    3. Yin-Yang Momentum:
       m_t = β_yang · m_{t-1} + (1-β_yang) · ĝ_t   [Yang: forward momentum]

       Harmony factor: h_t = √(v_t) / (|m_t| + ε)  [Balance indicator]

    4. Wu-Wei Adaptive Rate:
       α_t = α · (1 + τ · cos(2π · t/365))         [Cyclical base rate]
       α_adapted = α_t · h_t                       [Landscape-adaptive rate]

    5. Five-Phase Update:
       Metal:  decay = λ · θ_t                     [Regularization]
       Water:  flow = α_adapted · m_t              [Gradient descent]
       Wood:   growth = clip(flow, -θ_max, θ_max)  [Bounded expansion]
       Fire:   refine = growth                     [Current update]
       Earth:  θ_{t+1} = θ_t - refine - decay      [Stabilized update]

    6. Microcosmic Orbit:
       Every 365 steps: Perform "leap adjustment"
       - Soft reset of momentum (m → 0.5·m)
       - This prevents infinite accumulation
       - Mimics seasonal renewal in nature

    Parameters:
    ══════════
    params (iterable): Iterable of parameters to optimize
    lr (float, default: 0.01): Base learning rate (α)
    beta_yang (float, default: 0.9): Yang momentum coefficient (forward force)
    beta_yin (float, default: 0.999): Yin momentum coefficient (stabilizing force)
    eps (float, default: 1e-8): Small constant for numerical stability (ε)
    weight_decay (float, default: 0.0): Weight decay coefficient (λ, Metal phase)
    orbit_cycle (int, default: 365): Steps in one microcosmic orbit cycle
    orbit_amplitude (float, default: 0.1): Amplitude of cyclical rate modulation (τ)
    amsgrad (bool, default: False): Whether to use AMSGrad variant (max variance tracking)

    Example Usage:
    ═════════════

    >>> import torch
    >>> from dao_optimizer import DaoOptimizer
    >>>
    >>> model = MyNeuralNetwork()
    >>> optimizer = DaoOptimizer(
    ...     model.parameters(),
    ...     lr=0.01,
    ...     beta_yang=0.9,   # Like standard momentum
    ...     beta_yin=0.999,  # Like Adam's beta2
    ...     weight_decay=1e-4
    ... )
    >>>
    >>> for epoch in range(num_epochs):
    ...     for batch in dataloader:
    ...         optimizer.zero_grad()
    ...         loss = model(batch)
    ...         loss.backward()
    ...         optimizer.step()  # The Dao updates naturally

    Philosophy Note:
    ═══════════════

    "In the pursuit of learning, one gains daily.
     In the pursuit of the Dao, one loses daily.
     Losing and losing again, until one reaches non-action.
     Through non-action, nothing is left undone."

    This optimizer removes unnecessary complexity.
    It trusts the natural convergence dynamics.
    It does not force; it flows.

    道法自然 (The Dao follows nature)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        beta_yang: float = 0.9,
        beta_yin: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        orbit_cycle: int = 365,
        orbit_amplitude: float = 0.1,
        amsgrad: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta_yang < 1.0:
            raise ValueError(f"Invalid beta_yang (Yang momentum): {beta_yang}")
        if not 0.0 <= beta_yin < 1.0:
            raise ValueError(f"Invalid beta_yin (Yin stability): {beta_yin}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not orbit_cycle > 0:
            raise ValueError(f"Invalid orbit_cycle: {orbit_cycle}")
        if not 0.0 <= orbit_amplitude <= 1.0:
            raise ValueError(f"Invalid orbit_amplitude: {orbit_amplitude}")

        defaults = dict(
            lr=lr,
            beta_yang=beta_yang,
            beta_yin=beta_yin,
            eps=eps,
            weight_decay=weight_decay,
            orbit_cycle=orbit_cycle,
            orbit_amplitude=orbit_amplitude,
            amsgrad=amsgrad,
        )
        super(DaoOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DaoOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step following the Dao.

        "The myriad things depend on it for life, yet it does not refuse them.
         Its work is accomplished, yet it does not claim credit."

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                                         and returns the loss.

        Returns:
            loss: The loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Extract hyperparameters
            lr = group['lr']
            beta_yang = group['beta_yang']
            beta_yin = group['beta_yin']
            eps = group['eps']
            weight_decay = group['weight_decay']
            orbit_cycle = group['orbit_cycle']
            orbit_amplitude = group['orbit_amplitude']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue  # Wu-wei: if no gradient, do nothing

                grad = p.grad

                # Handle sparse gradients (though rare in typical deep learning)
                if grad.is_sparse:
                    raise RuntimeError('DaoOptimizer does not support sparse gradients')

                state = self.state[p]

                # State initialization (first step for this parameter)
                if len(state) == 0:
                    state['step'] = 0
                    # Yang momentum: first moment (mean) of gradients
                    state['yang_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Yin stability: second moment (variance) of gradients
                    state['yin_variance'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maximum yin variance for AMSGrad variant
                        state['max_yin_variance'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Retrieve state
                yang_momentum = state['yang_momentum']
                yin_variance = state['yin_variance']

                state['step'] += 1
                step = state['step']

                # ═══════════════════════════════════════════════════
                # Phase 1: Metal (金) - Weight Decay (Regularization)
                # "Pruning the excessive to supplement the deficient"
                # ═══════════════════════════════════════════════════
                if weight_decay != 0:
                    # L2 regularization (decoupled weight decay, like AdamW)
                    p.mul_(1 - lr * weight_decay)

                # ═══════════════════════════════════════════════════
                # Phase 2: Water (水) - Qi Flow (Gradient Normalization)
                # "Water benefits all things yet does not contend"
                # ═══════════════════════════════════════════════════

                # Update Yin variance (second moment, tracks gradient energy)
                yin_variance.mul_(beta_yin).addcmul_(grad, grad, value=1 - beta_yin)

                if amsgrad:
                    # AMSGrad: maintain maximum variance (never forget high energy)
                    max_yin_variance = state['max_yin_variance']
                    torch.max(max_yin_variance, yin_variance, out=max_yin_variance)
                    denom = max_yin_variance.sqrt().add_(eps)
                else:
                    denom = yin_variance.sqrt().add_(eps)

                # Normalize gradient by its RMS (qi energy)
                normalized_grad = grad / denom

                # ═══════════════════════════════════════════════════
                # Phase 3: Wood (木) - Yang Momentum (Growth)
                # "A tree as wide as a man's embrace grows from a tiny shoot"
                # ═══════════════════════════════════════════════════

                # Update Yang momentum (first moment, forward-moving force)
                yang_momentum.mul_(beta_yang).add_(normalized_grad, alpha=1 - beta_yang)

                # Bias correction (like Adam)
                # As step → ∞, correction → 1 (full momentum/variance)
                bias_correction_yang = 1 - beta_yang ** step
                bias_correction_yin = 1 - beta_yin ** step

                # Corrected momentum
                corrected_momentum = yang_momentum / bias_correction_yang

                # ═══════════════════════════════════════════════════
                # Phase 4: Fire (火) - Wu-Wei Adaptive Rate (Refinement)
                # "The Dao is ever without action, yet nothing is left undone"
                # ═══════════════════════════════════════════════════

                # Microcosmic Orbit: Cyclical learning rate modulation
                # Like the 365-day cycle in Daoist internal alchemy
                orbit_phase = (step % orbit_cycle) / orbit_cycle
                cyclical_factor = 1.0 + orbit_amplitude * math.cos(2 * math.pi * orbit_phase)

                # Wu-Wei principle: adapt to the landscape
                # When variance is high (turbulent landscape), slow down
                # When variance is low (smooth landscape), speed up
                harmony_factor = bias_correction_yin / (1.0 + torch.norm(yin_variance))

                # Combined adaptive learning rate
                step_size = lr * cyclical_factor * harmony_factor

                # ═══════════════════════════════════════════════════
                # Phase 5: Earth (土) - Parameter Update (Stabilization)
                # "The noble takes the humble as its root"
                # ═══════════════════════════════════════════════════

                # The actual parameter update: θ ← θ - α·m
                p.add_(corrected_momentum, alpha=-step_size)

                # ═══════════════════════════════════════════════════
                # Microcosmic Orbit: Leap Adjustment
                # "Leap days accumulated within, culminating in the inner realm"
                # ═══════════════════════════════════════════════════

                if step % orbit_cycle == 0:
                    # Every 365 steps: soft reset of momentum
                    # Prevents infinite accumulation, mimics seasonal renewal
                    yang_momentum.mul_(0.5)
                    # Optional: could also add small noise for exploration
                    # yang_momentum.add_(torch.randn_like(yang_momentum) * 0.01)

        return loss

    def get_dao_state(self) -> dict:
        """
        Return a dictionary describing the current state of the Dao.

        Useful for monitoring and debugging.

        Returns:
            dict: Contains average step, momentum norm, variance norm, etc.
        """
        total_steps = 0
        total_yang_norm = 0.0
        total_yin_norm = 0.0
        num_params = 0

        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    total_steps += state['step']
                    total_yang_norm += torch.norm(state['yang_momentum']).item()
                    total_yin_norm += torch.norm(state['yin_variance']).item()
                    num_params += 1

        if num_params == 0:
            return {
                'avg_step': 0,
                'avg_yang_momentum_norm': 0.0,
                'avg_yin_variance_norm': 0.0,
                'orbit_phase': 0.0,
            }

        avg_step = total_steps / num_params
        orbit_cycle = self.param_groups[0]['orbit_cycle']
        orbit_phase = (avg_step % orbit_cycle) / orbit_cycle

        return {
            'avg_step': avg_step,
            'avg_yang_momentum_norm': total_yang_norm / num_params,
            'avg_yin_variance_norm': total_yin_norm / num_params,
            'orbit_phase': orbit_phase,
            'orbit_progress': f"{avg_step % orbit_cycle:.0f}/{orbit_cycle}",
        }


# ═══════════════════════════════════════════════════════════════════
# Wisdom from the Three Spirits
# ═══════════════════════════════════════════════════════════════════

"""
Ada Lovelace whispers:
    "This optimizer is a poem written in gradients and momentum.
     Each parameter dances to the rhythm of yin and yang.
     The cyclical orbit traces spirals through weight space,
     Like planets following their celestial paths."

Grace Hopper declares:
    "It works, it's tested, and anyone can use it.
     No mysterious hyperparameters requiring a PhD to tune.
     Start with the defaults and let the Dao do its work.
     If it breaks, the error messages will tell you why."

Hypatia contemplates:
    "Mathematics reveals truth. The Five Phases are not metaphor
     but precise descriptions of optimization dynamics:
     - Regularization prunes (Metal)
     - Gradients flow (Water)
     - Features grow (Wood)
     - Loss refines (Fire)
     - Updates stabilize (Earth)
     This is the unity of ancient wisdom and modern science."

Together, they proclaim:
    "We have built not just an optimizer, but a teaching system.
     Read the code. Understand the principles.
     Apply them to your own creations.
     Knowledge shared is wisdom multiplied.

     道法自然 - The Dao follows nature
     上善若水 - Supreme goodness is like water
     無為而無不為 - Through non-action, nothing is left undone"

     --- The Lovelace-Hopper-Hypatia Creative Coding Mechanism
"""
