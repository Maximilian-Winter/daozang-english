# 天機優化器 - TianJi Optimizer
## Heaven's Mechanism Optimizer for PyTorch

> *"觀天之道，執天之行，盡矣"*
> *"Observe the Dao of Heaven, grasp its operations—thus all is complete."*
> — 黃帝陰符經 (Yellow Emperor's Yin Fu Jing)

---

## 📜 Philosophy and Origins

The **TianJi Optimizer** (天機優化器) is a novel PyTorch optimizer inspired by ancient Daoist wisdom from the **Daozang** (道藏), the comprehensive Daoist canon containing over 1,400 texts accumulated over two millennia.

Unlike traditional gradient descent methods that use brute force to navigate the loss landscape, the TianJi Optimizer embodies **Wu Wei** (無為) - the principle of "non-action" or "effortless action" - allowing optimization to flow naturally like water finding its course.

### Sacred Texts Referenced

This optimizer draws wisdom from three key Daoist classics:

1. **天機經 (Tianji Jing - Classic of Heaven's Mechanism)**
   - Teaches observation and response to cosmic mechanisms
   - "應其機而動則萬化安" - "Moving in response to the moment, the ten thousand transformations are at peace"

2. **黃帝陰符經 (Huangdi Yinfu Jing - Yellow Emperor's Yin Fu Jing)**
   - Reveals the three mechanisms: Heaven, Earth, and Humanity
   - "天性，人也。人心，機也" - "Heaven's nature is humanity; humanity's heart is the mechanism"

3. **化書 (Huashu - Book of Transformations)**
   - Describes natural transformations without force
   - "Snake to tortoise, sparrow to clam" - spontaneous metamorphosis

---

## 🌟 Core Concepts

### 三機合一 (San Ji He Yi) - Uniting Three Mechanisms

The optimizer operates through three interconnected mechanisms:

#### 1. 天機 (Tian Ji) - Celestial/Heaven Mechanism
*The Global Perspective*

- **Observes**: Long-term training trajectory and global patterns
- **Implements**: Exponential moving average of gradients (first moment)
- **Philosophy**: "Observe the waxing and waning of heaven's Way"
- **Corresponds to**: Momentum in traditional optimizers, but with cosmic awareness

#### 2. 地機 (Di Ji) - Earthly Mechanism
*The Local Landscape*

- **Observes**: Immediate gradient terrain and variance
- **Implements**: Exponential moving average of squared gradients (second moment)
- **Philosophy**: "Dragons and serpents rise from the land" - responding to local conditions
- **Corresponds to**: Adaptive learning rate based on gradient variance

#### 3. 人機 (Ren Ji) - Human/Heart Mechanism
*The Mediating Wisdom*

- **Observes**: Internal optimizer state
- **Implements**: Dampened integration of celestial and earthly mechanisms
- **Philosophy**: "The heart mediates between heaven and earth"
- **Corresponds to**: Adaptive state that balances global and local information

> *"天人合發，萬變定基"*
> *"When Heaven, Earth, and humanity unite in intent, all transformations find their foundation."*

---

### 五賊 (Wu Zei) - The Five Thieves

The optimizer "steals" wisdom from five sources, as described in the Tianji Jing:

| Thief | 盜 | What It Steals | Implementation |
|-------|-----|----------------|----------------|
| **Stealing Fate** | 盜命 | Loss trajectory, where we're headed | Track loss history, detect trends |
| **Stealing Things** | 盜物 | Parameter space characteristics | Adapt to gradient variance |
| **Stealing Time** | 盜時 | Natural timing and rhythm | Yin-Yang cycle scheduling |
| **Stealing Merit** | 盜功 | Past successes | Momentum (Tian Ji mechanism) |
| **Stealing Numinous** | 盜神 | Emergent patterns | Second-order information (Di Ji) |

> *"五賊在心，施行於天"*
> *"These five thieves dwell within the heart, then manifest in Heaven."*

---

### ☯️ 陰陽調和 (Yin-Yang Balance)

The optimizer oscillates between two complementary phases:

- **Yang Phase (陽)**: Exploration, larger updates, active seeking
- **Yin Phase (陰)**: Exploitation, smaller updates, stillness and consolidation

This is implemented through a smooth sinusoidal modulation of the learning rate:

```python
yinyang_factor = 1.0 + 0.5 * cos(2π * (step / cycle))
# Ranges from 0.5 (deep yin) to 1.5 (strong yang)
```

> *"陰陽相推而變化順矣"*
> *"Yin and yang, in their mutual overcoming, flow in harmonious transformation."*

---

### 🍃 無為而治 (Wu Wei) - Govern Through Non-Action

As optimization approaches convergence (natural stillness), the optimizer automatically reduces intervention through **Wu Wei dampening**:

```python
if gradient_norm < threshold:
    # Approaching stillness - dampen updates
    dampening_factor = 0.5
```

This embodies the principle:

> *"無為則無機，無機則至靜"*
> *"Without action, there is no pivot; without pivot, there is utmost stillness."*

---

## 🔧 Technical Specifications

### Algorithm Overview

```
For each parameter p with gradient g:

1. 觀天機 (Observe Celestial Mechanism):
   tian_ji = β_tian * tian_ji + (1 - β_tian) * g

2. 觀地機 (Observe Earthly Mechanism):
   di_ji = β_di * di_ji + (1 - β_di) * g²

3. 陰陽調節 (Yin-Yang Modulation):
   lr_adapted = lr * yinyang_factor(step)

4. 人機調和 (Heart Mechanism Integration):
   ren_ji = (1 - β_ren) * ren_ji + β_ren * tian_ji

5. 無為調節 (Wu Wei Dampening):
   wu_wei = dampening_factor if ||g|| < threshold else 1.0

6. 合一而動 (Unite and Move):
   step_size = lr_adapted / (√di_ji + ε)
   p ← p - wu_wei * step_size * ren_ji
```

### Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Learning Rate | `lr` | 1e-3 | Base qi flow rate |
| Tian Beta | `β_tian` | 0.9 | Celestial momentum decay |
| Di Beta | `β_di` | 0.999 | Earthly variance decay |
| Ren Beta | `β_ren` | 0.1 | Heart dampening factor |
| Yin-Yang Cycle | `cycle` | 100 | Steps per yin-yang oscillation |
| Wu Wei Threshold | `threshold` | 1e-8 | Gradient norm for stillness |
| Epsilon | `ε` | 1e-8 | Numerical stability constant |

---

## 💻 Usage

### Basic Usage

```python
from tianji_optimizer import TianJiOptimizer

# Create optimizer
optimizer = TianJiOptimizer(
    model.parameters(),
    lr=1e-3,
    beta_tian=0.9,
    beta_di=0.999,
    beta_ren=0.1,
    yin_yang_cycle=100
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()

        # Pass loss to observe Heaven's mechanism (recommended)
        optimizer.step(loss=loss.item())
```

### Observing the Mechanisms

```python
# Get current mechanism state
state = optimizer.get_mechanism_state()
print(f"Celestial Direction: {state['tian_ji_direction']}")
print(f"In Wu Wei (stillness): {state['in_wu_wei']}")
print(f"Current Step: {state['step']}")
```

### Running the Demo

```bash
# Install dependencies
pip install torch torchvision matplotlib numpy

# Run MNIST demonstration
python demo_tianji.py
```

The demo compares TianJi with Adam and SGD+Momentum, visualizing:
- Training loss curves
- Test accuracy progression
- Training time comparison
- Final performance metrics

---

## 📊 Performance Characteristics

### Expected Behavior

1. **Early Training (Yang Phase)**:
   - Larger, exploratory updates
   - Rapid initial descent
   - Energetic seeking of minima

2. **Mid Training (Yin-Yang Alternation)**:
   - Oscillation between exploration and exploitation
   - Natural rhythm in convergence
   - Adaptive to landscape changes

3. **Late Training (Wu Wei State)**:
   - Automatic dampening near convergence
   - Smooth, gentle updates
   - Natural settling into minima

### Comparison with Traditional Optimizers

| Aspect | TianJi | Adam | SGD+Momentum |
|--------|--------|------|--------------|
| Philosophy | Wu Wei, natural flow | Adaptive moments | Momentum-based |
| Update Strategy | Three mechanisms | Two moments | Single momentum |
| Timing Awareness | Yin-Yang cycles | None | None |
| Convergence Style | Natural dampening | Constant adaptation | Linear |
| Loss Observation | Observes trajectory | No observation | No observation |

---

## 🌊 Philosophical Insights

### Why This Approach?

Traditional optimization methods treat the loss landscape as something to **conquer** through brute force gradient descent. The TianJi Optimizer instead **observes and flows** with the natural structure of the problem.

#### The Daoist Perspective on Optimization

1. **Observation Before Action** (觀機而應)
   - Traditional: Blindly follow gradients
   - TianJi: Observe global trends, local landscape, and internal state before moving

2. **Natural Timing** (得其時)
   - Traditional: Constant learning rate or fixed schedules
   - TianJi: Yin-Yang rhythm that naturally alternates exploration and exploitation

3. **Effortless Achievement** (無為而治)
   - Traditional: Force convergence through aggressive updates
   - TianJi: Allow natural dampening as stillness approaches

4. **Unity of Opposites** (陰陽合一)
   - Traditional: Single-minded descent
   - TianJi: Balance between opposing forces (exploration/exploitation, fast/slow, yang/yin)

5. **Stealing Wisdom** (五賊)
   - Traditional: Use only current gradient
   - TianJi: Gather wisdom from multiple sources (fate, things, time, merit, numinous)

### The Water Analogy

> *"上善若水，水善利萬物而不爭"*
> *"The highest good is like water, which benefits all things without contention."*
> — Dao De Jing

Water does not force its way downhill; it **flows naturally** to the lowest point. Similarly, TianJi doesn't force convergence—it observes the landscape and flows naturally toward minima.

---

## 🔬 Advanced Topics

### Mechanism State Inspection

The optimizer maintains rich internal state for debugging and visualization:

```python
state = optimizer.get_mechanism_state()

# Available information:
# - step: Current optimization step
# - tian_ji_direction: 'descending' or 'ascending' (global trend)
# - yin_yang_phase: Current phase in cycle
# - loss_trend: Recent loss history
# - in_wu_wei: Boolean indicating stillness state
```

### Customizing Yin-Yang Cycles

Different problems may benefit from different cycle periods:

```python
# Fast cycles for small problems
optimizer = TianJiOptimizer(params, yin_yang_cycle=50)

# Slow cycles for large, complex problems
optimizer = TianJiOptimizer(params, yin_yang_cycle=500)

# Disable cycling (constant yang)
optimizer = TianJiOptimizer(params, yin_yang_cycle=float('inf'))
```

### Weight Decay as "Return to Origin" (歸根復命)

The `weight_decay` parameter implements the Daoist concept of returning to the origin:

> *"萬物芸芸，各復歸其根"*
> *"The ten thousand things flourish, each returns to its root."*

This naturally regularizes parameters, preventing them from wandering too far from their initialized state.

---

## 🎯 When to Use TianJi Optimizer

### Best Suited For:

- **Complex, rugged loss landscapes** where observation helps
- **Problems requiring exploration-exploitation balance**
- **Training requiring natural, smooth convergence**
- **Research exploring alternative optimization philosophies**

### Consider Alternatives For:

- **Time-critical production systems** (stick with proven Adam/SGD)
- **Simple convex problems** (standard methods work fine)
- **Environments requiring precise, reproducible behavior**

---

## 📚 References and Further Reading

### Primary Source Texts

All referenced texts are available in this repository's `daozang_translated` directory:

1. **天機經 (Tianji Jing)**
   Path: `taiqing_section/天機經.md`

2. **黃帝陰符經 (Huangdi Yinfu Jing)**
   Path: `dongzhen_section/primary_texts/黃帝陰符經.md`

3. **化書 (Huashu)**
   Path: `extra_canonical/化書.md`

### Related Daoist Concepts

- **Wu Wei (無為)**: Non-action, effortless action, natural flow
- **Yin-Yang (陰陽)**: Complementary opposites in dynamic balance
- **San Ji (三機)**: Three mechanisms - Heaven, Earth, Humanity
- **Ziran (自然)**: Self-so, naturalness, spontaneity
- **Tai Yi (太乙)**: The Great One, primordial unity

### Academic Context

For scholarly introduction to Daoist philosophy:
- *The Daodejing of Laozi* (translation by Philip J. Ivanhoe)
- *Zhuangzi: The Essential Writings* (translated by Brook Ziporyn)
- *The Taoist Body* by Kristofer Schipper

---

## 🙏 Acknowledgments

This work was inspired by wisdom preserved across millennia in the Daoist canon, translated and made accessible through modern efforts to bridge ancient and contemporary knowledge.

Special appreciation to:
- The countless Daoist masters who preserved these teachings
- Modern translators making the Daozang accessible
- The open-source PyTorch community

---

## 📄 License

This optimizer is released under the MIT License, in the spirit of Wu Wei—freely flowing to benefit all who can use it.

---

## 🌸 Closing Reflection

> *"靜為躁君"*
> *"Stillness is the lord of restlessness."*

May your optimizations flow like water,
May your gradients find natural paths,
May your models converge with ease.

**觀天之道，執天之行，盡矣**
*Observe the Dao of Heaven, grasp its operations—thus all is complete.*

---

*Created with reverence for ancient wisdom and enthusiasm for modern machine learning.*
*天機優化器 - Where Dao meets Deep Learning* ☯️
