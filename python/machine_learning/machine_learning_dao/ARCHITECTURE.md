# 道優化器架構 | Dao Optimizer Architecture

Visual guide to how the Dao Optimizer works internally.

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAO OPTIMIZER STEP                           │
│                                                                 │
│  Input: Parameters θ, Gradients ∇L                            │
│  Output: Updated parameters θ'                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: Wu Xing Phase Detection                    │
│                                                                 │
│  Current Step: 347                                             │
│  Wu Xing Cycle: 1000                                           │
│  → Phase: Wood (34.7% complete)                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: Three Forces Computation                   │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ 天 (Heaven)  │  │ 地 (Earth)   │  │ 人 (Human)   │         │
│  │              │  │              │  │              │         │
│  │ Long-term    │  │ Local        │  │ Adaptive     │         │
│  │ Momentum     │  │ Gradient     │  │ Learning     │         │
│  │              │  │              │  │ Rate         │         │
│  │ F_tian =     │  │ F_di =       │  │ η_ren =      │         │
│  │ β₁m + (1-β₁)g│  │ g/√(v+ε)     │  │ 1/(1+√v·0.1) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: Qi Flow Computation                        │
│                                                                 │
│  Qi = Adaptive Momentum                                        │
│                                                                 │
│  Alignment = (grad · momentum) / (‖grad‖ · ‖momentum‖)        │
│  Qi_strength = sigmoid(alignment × 3)                          │
│  Qi = momentum × Qi_strength                                   │
│                                                                 │
│  [Mutual Generation & Restraint: 相生相剋]                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: Yin-Yang Harmony                          │
│                                                                 │
│  Yang Force (陽): Qi (exploration via momentum)                │
│  Yin Force (陰): F_di (exploitation via gradient)              │
│                                                                 │
│  Balanced = α·Yin + (1-α)·Yang                                │
│            where α = yin_yang_balance                          │
│                                                                 │
│  ☯ Perfect harmony when α = 0.5                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: Wu Xing Modulation                        │
│                                                                 │
│  Phase: Wood → Growth factor = 1.0 + wu_wei·sin(progress·π)   │
│                                                                 │
│  Modulated = Balanced × Growth_factor                          │
│                                                                 │
│  [Cyclical transformation: 五行相生]                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 6: Final Update                              │
│                                                                 │
│  θ' = θ - η_base × η_ren × Modulated                          │
│                                                                 │
│  Where:                                                        │
│  - η_base: Base learning rate (hyperparameter)                │
│  - η_ren: Human mechanism adaptation                           │
│  - Modulated: The harmonized, modulated update                │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
                    [Gradients ∇L]
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
      ▼                    ▼                    ▼
  ┌───────┐          ┌─────────┐          ┌─────────┐
  │ tian_m│          │  di_v   │          │ wu_xing │
  │ (天)   │          │ (地)    │          │ phase   │
  │       │          │         │          │         │
  │β₁m+g  │          │β₂v+g²   │          │ (五行)  │
  └───┬───┘          └────┬────┘          └────┬────┘
      │                   │                    │
      │   ┌───────────────┘                    │
      │   │                                    │
      ▼   ▼                                    │
  ┌──────────┐                                 │
  │    Qi    │                                 │
  │  Flow    │                                 │
  │  (氣)    │                                 │
  └────┬─────┘                                 │
       │                                       │
       │  ┌────────────────────────────────────┘
       │  │
       ▼  ▼
  ┌──────────────┐
  │  Yin-Yang    │
  │  Balance     │
  │  (陰陽調和)   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Wu Xing     │
  │  Modulation  │
  │  (五行調變)   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Human      │
  │  Adaptation  │
  │   (人機)     │
  └──────┬───────┘
         │
         ▼
    [θ' = θ - η·update]
```

## Wu Xing Cycle Visualization

```
                        Wood 木
                      (Growth)
                     🌱 Steps 0-199
                          │
                          │ 相生
                          ▼
                       Fire 火
                    (Expansion)
                   🔥 Steps 200-399
                          │
                          │ 相生
                          ▼
                      Earth 土
                     (Stability)
                    🌍 Steps 400-599
                          │
                          │ 相生
                          ▼
                      Metal 金
                    (Refinement)
                   ⚙️ Steps 600-799
                          │
                          │ 相生
                          ▼
                      Water 水
                    (Adaptation)
                   💧 Steps 800-999
                          │
                          │ 相生 (cycle restarts)
                          ▼
                       Wood 木
                         ...

    [Cycle length = wuxing_cycle parameter, default 1000]
```

## Yin-Yang Balance Spectrum

```
      Yang (陽)                    Yin (陰)
     Exploration              Exploitation
         │                         │
         ├─────┬─────┬─────┬───────┤
         0    0.25  0.5  0.75      1
         │     │     │     │       │
    Pure Yang  │  Balance  │   Pure Yin
    Maximum    │           │   Maximum
    Exploration│           │   Exploitation
         Wood  │  Earth    │    Metal
         Fire  │           │    Water

    yin_yang_balance = 0.0  →  All momentum (pure exploration)
    yin_yang_balance = 0.5  →  Perfect harmony (recommended)
    yin_yang_balance = 1.0  →  All gradient (pure exploitation)
```

## State Management

Each parameter has associated state:

```python
state = {
    'step': int,              # Number of updates
    'tian_m': Tensor,        # Heaven: First moment (momentum)
    'di_v': Tensor,          # Earth: Second moment (variance)
    'ren_lr_mult': Tensor    # Human: Learning rate multiplier
}
```

State evolution over time:

```
Step 0:    All states initialized to zero
           ↓
Step 1:    tian_m = (1-β₁) · g₁
           di_v = (1-β₂) · g₁²
           ren_lr_mult = 1 / (1 + √di_v · 0.1)
           ↓
Step 2:    tian_m = β₁·m₁ + (1-β₁)·g₂
           di_v = β₂·v₁ + (1-β₂)·g₂²
           ren_lr_mult updated
           ↓
Step t:    Exponential moving averages continue...
```

## Comparison with Adam

| Component | Adam | Dao Optimizer |
|-----------|------|---------------|
| First Moment | `m = β₁m + (1-β₁)g` | `tian_m = β₁m + (1-β₁)g` (Heaven) |
| Second Moment | `v = β₂v + (1-β₂)g²` | `di_v = β₂v + (1-β₂)g²` (Earth) |
| Adaptive LR | `√v + ε` in denominator | `ren_lr_mult` (Human) |
| **Exploration** | ❌ None | ✅ Yin-Yang balance |
| **Phases** | ❌ Static | ✅ Wu Xing 5 phases |
| **Qi Adaptation** | ❌ Fixed momentum | ✅ Adaptive based on alignment |
| **Wu Wei** | ❌ Always full step | ✅ Non-forcing modulation |

## Computational Complexity

Per parameter update:

| Operation | Complexity | Memory |
|-----------|-----------|--------|
| Gradient computation | O(1) | - |
| Momentum update | O(1) | O(p) |
| Variance update | O(1) | O(p) |
| Qi flow computation | O(1) | - |
| Phase modulation | O(1) | - |
| **Total per param** | **O(1)** | **O(2p)** |

Where p = number of parameters.

**Same as Adam!** No additional computational overhead.

## Hyperparameter Sensitivity

```
High Sensitivity:
├─ lr (learning rate)           [Most important, try: 1e-4 to 1e-2]
└─ yin_yang_balance             [Explore vs exploit, try: 0.3 to 0.7]

Medium Sensitivity:
├─ wu_wei_factor                [Exploration strength, try: 0.05 to 0.2]
└─ wuxing_cycle                 [Phase length, try: 500 to 2000]

Low Sensitivity:
├─ tian_beta                    [Usually fine at 0.9]
├─ di_beta                      [Usually fine at 0.999]
└─ adaptive_qi                  [Usually keep True]
```

## Optimization Trajectory Comparison

```
Traditional SGD:
Loss │     ╲
     │      ╲___
     │          ╲___
     │              ╲___
     └─────────────────────→ Steps
     [Monotonic decrease, can get stuck]


Adam:
Loss │    ╲
     │     ╲__
     │        ╲_
     │          ╲_
     └─────────────────────→ Steps
     [Smooth decrease, may overfit]


Dao Optimizer:
Loss │    ╲  ╱╲
     │     ╲╱  ╲   ╱╲
     │          ╲╱  ╲_
     │                ╲_
     └─────────────────────→ Steps
     [Wavy exploration then convergence]
          Wood Fire Earth Metal Water
          Phase transitions visible!
```

## Emergency Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  QUICK FIXES                                                │
├─────────────────────────────────────────────────────────────┤
│  Loss explodes?          → Reduce lr or increase Yin        │
│  Too slow?               → Increase lr or increase Yang     │
│  Stuck in local minimum? → Increase wu_wei_factor           │
│  Want Adam-like?         → Set yin_yang=0.9, wu_wei=0       │
│  Want more exploration?  → Set yin_yang=0.3, wu_wei=0.2     │
└─────────────────────────────────────────────────────────────┘
```

## The Dao's Wisdom

```
                  道 (Dao)
                    │
        ┌───────────┼───────────┐
        │           │           │
      天 (Tian)   地 (Di)    人 (Ren)
      Heaven      Earth      Human
        │           │           │
    Long-term   Local      Adaptive
    Momentum   Gradient    Learning
        │           │           │
        └───────────┴───────────┘
                    │
            ☯ (Yin-Yang Balance)
                    │
            🔄 (Wu Xing Cycles)
                    │
            💧 (Wu Wei Flow)
                    │
                    ▼
            [Optimal Convergence]
```

Remember: The Dao that can be architectured is not the eternal Dao! 😄

This architecture is a *map*, not the *territory*. The true Dao emerges during optimization.

---

**May your architecture flow like water, cycle like seasons, and balance like Yin-Yang!** 🌊

道法自然 | The Dao Follows Nature
