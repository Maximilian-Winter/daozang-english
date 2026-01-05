# 太乙神數 (Taiyi Shenshu) - Algorithm Specification
## Extracted from Classical Texts: 太乙金鏡式經 (Tang) & 太乙秘書 (Song)

---

## 1. Core Cycle Constants (基本週期常數)

```python
ZHOU_JI_FA = 360      # 周紀法 - Grand Cycle (years/months/days)
YUAN_FA = 72          # 元法 - Era Law
TAIYI_XIAO_ZHOU = 24  # 太乙小周法 - Taiyi Small Cycle
TIANMU_ZHOU_FA = 18   # 天目周法 - Tianmu Cycle
DA_YOU_XIAO_ZHOU = 288  # 大游小周法 - Great Wandering Small Cycle
DA_YOU_DA_ZHOU = 4320   # 大游大周法 - Great Wandering Grand Cycle
XIAO_YOU_DA_ZHOU = 240  # 小游大周法 - Small Wandering Grand Cycle
XIAO_YOU_XIAO_ZHOU = 24 # 小游小周法 - Small Wandering Small Cycle
```

---

## 2. The Sixteen Spirits (十六神)

The Sixteen Spirits correspond to 12 Earthly Branches + 4 Corner Trigrams.

| # | Position | Spirit Name | Pinyin | Element | Meaning |
|---|----------|-------------|--------|---------|---------|
| 1 | 子 (Zi) | 地主 | Dìzhǔ | Water | 阳气初发，万物阴生 |
| 2 | 丑 (Chou) | 阳德 | Yángdé | Earth | 二阳用事，布育万物 |
| 3 | 艮 (Gen) | 和德 | Hédé | Earth | 冬春将交，阴阳气合 |
| 4 | 寅 (Yin) | 吕申 | Lǚshēn | Wood | 阳气大申，草木甲拆 |
| 5 | 卯 (Mao) | 高丛 | Gāocóng | Wood | 万物皆出，自地丛生 |
| 6 | 辰 (Chen) | 太阳 | Tàiyáng | Earth | 雷出震势，阳气大盛 |
| 7 | 巽 (Xun) | 太灵 | Tàilíng | Wood | 春夏将交，盛暑方至 |
| 8 | 巳 (Si) | 太神 | Tàishén | Fire | 少阴用事，阴阳不测 |
| 9 | 午 (Wu) | 大威 | Dàwēi | Fire | 阳附阴生，刑暴始行 |
| 10 | 未 (Wei) | 天道 | Tiāndào | Earth | 火能生土，土王于未 |
| 11 | 坤 (Kun) | 大武 | Dàwǔ | Earth | 夏秋将交，阴气施令 **[重留]** |
| 12 | 申 (Shen) | 武德 | Wǔdé | Metal | 万物欲死，葬麦将生 **[起点]** |
| 13 | 酉 (You) | 太簇 | Tàicù | Metal | 万物皆成，有大品蔟 |
| 14 | 戌 (Xu) | 阴主 | Yīnzhǔ | Earth | 阳气不长，阴气用事 |
| 15 | 乾 (Qian) | 阴德 | Yīndé | Metal | 秋冬将交，阴前生阳 **[重留]** |
| 16 | 亥 (Hai) | 大义 | Dàyì | Water | 万物怀垢，群阳欲尽 |

### Spirit Traversal Order (from 武德):
```
武德(申) → 太簇(酉) → 阴主(戌) → 阴德(乾)[重留] → 大义(亥) → 地主(子) →
阳德(丑) → 和德(艮) → 吕申(寅) → 高丛(卯) → 太阳(辰) → 太灵(巽) →
太神(巳) → 大威(午) → 天道(未) → 大武(坤)[重留] → 武德(申)...
```

**Special Rule**: When encountering 阴德(乾) or 大武(坤), count stays one extra step (重留一算).

---

## 3. Nine Palaces (九宮)

### Palace Arrangement (Luoshu Magic Square)
```
    ┌───┬───┬───┐
    │ 4 │ 9 │ 2 │
    │巽  │離  │坤  │
    ├───┼───┼───┤
    │ 3 │ 5 │ 7 │
    │震  │中  │兌  │
    ├───┼───┼───┤
    │ 8 │ 1 │ 6 │
    │艮  │坎  │乾  │
    └───┴───┴───┘
```

### Palace Properties
| Palace | Trigram | Direction | Element | Type |
|--------|---------|-----------|---------|------|
| 1 | 坎 (Kan) | North | Water | 阴宫 |
| 2 | 坤 (Kun) | Southwest | Earth | 阴宫 |
| 3 | 震 (Zhen) | East | Wood | 阳宫 |
| 4 | 巽 (Xun) | Southeast | Wood | 阳宫 |
| 5 | 中 (Center) | Center | Earth | **不游** (Taiyi skips) |
| 6 | 乾 (Qian) | Northwest | Metal | 阴宫 |
| 7 | 兑 (Dui) | West | Metal | 阴宫 |
| 8 | 艮 (Gen) | Northeast | Earth | 阳宫 |
| 9 | 离 (Li) | South | Fire | 阳宫 |

### Palace Traversal Sequence
- **Yang Dun (阳遁)**: 1 → 2 → 3 → 4 → 6 → 7 → 8 → 9 → 1... (skip 5, forward)
- **Yin Dun (阴遁)**: 9 → 8 → 7 → 6 → 4 → 3 → 2 → 1 → 9... (skip 5, backward)

### Special Palace Classifications
- **阳宫** (Yang Palaces): 3, 4, 8, 9 — 利为主 (favorable for defender)
- **阴宫** (Yin Palaces): 1, 2, 6, 7 — 利为客 (favorable for attacker)
- **绝阳**: Palace 1
- **绝阴**: Palace 9
- **绝气**: Palaces 4, 6
- **易气**: Palaces 2, 8

---

## 4. Six Eras System (六紀)

Each 60-year cycle is divided into 6 eras of 10 years each.

| Era | Stem Pattern | Taiyi Palace | Tianmu (天目) |
|-----|--------------|--------------|---------------|
| 一紀 | 甲子、甲午 (二甲仲辰) | 1宫 | 武德 |
| 二紀 | 己巳、己亥 (二己孟辰) | 6宫 | 地主 |
| 三紀 | 甲辰、甲戌 (二甲季辰) | 1宫 | 太灵 |
| 四紀 | 己卯、己酉 (二己仲辰) | 6宫 | 武德 |
| 五紀 | 甲申、甲寅 (二甲孟辰) | 1宫 | 地主 |
| 六紀 | 己丑、己未 (二己季辰) | 6宫 | 太灵 |

**計神 (Jishen)**: 寅
**合神 (Heshen)**: 丑

---

## 5. Core Calculation Algorithms

### 5.1 Calculate Taiyi Position (積太乙所在法)

**For Year Calculation (歲計):**
```python
def calculate_taiyi_position(accumulated_years: int) -> tuple[int, int]:
    """
    Calculate Taiyi's palace position.

    Args:
        accumulated_years: Years since epoch (上元積年)

    Returns:
        (palace_number, years_in_palace)
    """
    # Step 1: Remove full grand cycles
    remainder = accumulated_years % ZHOU_JI_FA  # % 360

    # Step 2: Remove full eras
    remainder = remainder % YUAN_FA  # % 72

    # Step 3: Remove full small cycles
    remainder = remainder % TAIYI_XIAO_ZHOU  # % 24

    # Step 4: Divide by 3 to get palace count
    palace_count = remainder // 3
    years_in_palace = remainder % 3

    # Step 5: Map to palace (start from 1, traverse 8 palaces, skip 5)
    PALACE_SEQUENCE = [1, 2, 3, 4, 6, 7, 8, 9]  # Yang Dun order
    palace = PALACE_SEQUENCE[palace_count % 8]

    return (palace, years_in_palace)
```

### 5.2 Calculate Tianmu Position (天目所在法)

```python
def calculate_tianmu_position(accumulated_years: int) -> str:
    """
    Calculate Tianmu (天目) spirit position.

    Returns:
        Name of the spirit serving as Tianmu
    """
    SIXTEEN_SPIRITS = [
        "武德", "太簇", "阴主", "阴德", "大义", "地主",
        "阳德", "和德", "吕申", "高丛", "太阳", "太灵",
        "太神", "大威", "天道", "大武"
    ]

    # Spirits requiring extra count (重留)
    ZHONG_LIU = {"阴德", "大武"}

    # Step 1: Remove full cycles
    remainder = accumulated_years % YUAN_FA  # % 72
    remainder = remainder % TIANMU_ZHOU_FA   # % 18

    # Step 2: Count through spirits from 武德
    position = 0  # Start at 武德
    count = 0

    while count < remainder:
        spirit = SIXTEEN_SPIRITS[position % 16]
        count += 1
        if spirit in ZHONG_LIU:
            count += 1  # Extra count for 阴德 and 大武
        position += 1

    return SIXTEEN_SPIRITS[position % 16]
```

### 5.3 Calculate for Month (月計)

```python
def calculate_taiyi_month(accumulated_months: int) -> tuple[int, int]:
    """Same algorithm as year, but with accumulated months."""
    remainder = accumulated_months % ZHOU_JI_FA  # % 360
    remainder = remainder % YUAN_FA  # % 72
    remainder = remainder % TAIYI_XIAO_ZHOU  # % 24

    palace_count = remainder // 3
    months_in_palace = remainder % 3

    PALACE_SEQUENCE = [1, 2, 3, 4, 6, 7, 8, 9]
    palace = PALACE_SEQUENCE[palace_count % 8]

    return (palace, months_in_palace)
```

### 5.4 Calculate for Day (日計)

```python
def calculate_taiyi_day(accumulated_days: int) -> tuple[int, int]:
    """Same algorithm as year, but with accumulated days."""
    remainder = accumulated_days % ZHOU_JI_FA  # % 360
    remainder = remainder % YUAN_FA  # % 72
    remainder = remainder % TAIYI_XIAO_ZHOU  # % 24

    palace_count = remainder // 3
    days_in_palace = remainder % 3

    PALACE_SEQUENCE = [1, 2, 3, 4, 6, 7, 8, 9]
    palace = PALACE_SEQUENCE[palace_count % 8]

    return (palace, days_in_palace)
```

### 5.5 Calculate for Hour (時計) with Dun Type

```python
def calculate_taiyi_hour(accumulated_hours: int, is_yang_dun: bool) -> tuple[int, int]:
    """
    Calculate Taiyi position by hour with Yang/Yin Dun consideration.

    Args:
        accumulated_hours: Hours since epoch
        is_yang_dun: True for Yang Dun (after winter solstice),
                     False for Yin Dun (after summer solstice)
    """
    remainder = accumulated_hours % TAIYI_XIAO_ZHOU  # % 24

    palace_count = remainder // 3
    hours_in_palace = remainder % 3

    if is_yang_dun:
        # Yang Dun: Start from 1, forward
        PALACE_SEQUENCE = [1, 2, 3, 4, 6, 7, 8, 9]
    else:
        # Yin Dun: Start from 9, backward
        PALACE_SEQUENCE = [9, 8, 7, 6, 4, 3, 2, 1]

    palace = PALACE_SEQUENCE[palace_count % 8]

    return (palace, hours_in_palace)
```

---

## 6. The Five Generals System (五將)

| General | Chinese | Role | Associated Direction |
|---------|---------|------|---------------------|
| 太乙 | Taiyi | Supreme Commander | Center (Sovereign) |
| 文昌 | Wenchang | Upper Eye (上目) | Host/Defender |
| 始击 | Shiji | Lower Eye (下目) | Host/Defender |
| 主大将 | Zhu Dajiang | Host Grand General | Host/Defender |
| 客大将 | Ke Dajiang | Guest Grand General | Guest/Attacker |
| 主参将 | Zhu Canjiang | Host Deputy General | Host/Defender |
| 客参将 | Ke Canjiang | Guest Deputy General | Guest/Attacker |

---

## 7. The Eight Gates (八門)

| Gate | Chinese | Nature | Meaning |
|------|---------|--------|---------|
| 開 | Kāi (Open) | 吉 | Official matters, leadership |
| 休 | Xiū (Rest) | 吉 | Meeting nobles, audience with emperor |
| 生 | Shēng (Life) | 吉 | Contracts, marriage, fortune |
| 傷 | Shāng (Harm) | 凶 | Capturing thieves, conflict |
| 杜 | Dù (Block) | 中 | Hiding, avoiding disasters |
| 景 | Jǐng (View) | 中 | Travel (rain likely), meeting officials |
| 死 | Sǐ (Death) | 凶 | Hunting, not for siege |
| 驚 | Jīng (Shock) | 凶 | Confusion, false alarms |

### Gate Traversal (from 休門)
```
休(1) → 生(8) → 傷(3) → 杜(4) → 景(9) → 死(2) → 驚(7) → 開(6)
```

---

## 8. Strategic Patterns (格局)

### 8.1 Favorable Conditions (吉)
- **門具將發**: Three auspicious gates open, five generals deployed
- **算和**: Calculation harmony (主算 and 客算 balanced)
- **三門具**: Open, Rest, Life gates all accessible
- **五將發**: All five generals properly positioned

### 8.2 Unfavorable Conditions (凶)

| Pattern | Chinese | Meaning |
|---------|---------|---------|
| 囚 | Qiú | Imprisoned - general in same palace as Taiyi |
| 格 | Gé | Blocked - general in opposing palace to Taiyi |
| 迫 | Pò | Pressed - internal (内迫) or external (外迫) pressure |
| 關 | Guān | Barrier - generals blocking each other |
| 掩 | Yǎn | Concealed - hidden unfavorable condition |
| 擊 | Jī | Struck - under attack |
| 挾 | Xié | Seized - caught between forces |

### 8.3 Special Positions
- **绝阳** (Palace 1): Extreme yang, self-defeating for yang forces
- **绝阴** (Palace 9): Extreme yin, self-defeating for yin forces
- **四郭固**: Four walls solid - both sides locked, stalemate
- **四郭杜**: Four walls blocked - communications cut off

---

## 9. Epoch Reference (上元)

From 太乙金鏡式經:
- **周厲王三十七年甲子** is used as the Upper Epoch (上元)
- Distance to Tang Dynasty 開元十二年甲子: **1,561 years**

Historical Era Entry Points:
```
周厲王三十七年甲子 → 第一紀
周幽王五年甲子 → 第二紀
周惠王二十一年甲子 → 第三紀
...
開元十二年甲子 → Current reference
```

---

## 10. Yang/Yin Dun Determination (陽遁/陰遁)

```python
def determine_dun_type(solar_longitude: float) -> str:
    """
    Determine Yang or Yin Dun based on solar position.

    Args:
        solar_longitude: Sun's ecliptic longitude in degrees

    Returns:
        "yang" (冬至後) or "yin" (夏至後)
    """
    # Winter Solstice: ~270°, Summer Solstice: ~90°
    if 270 <= solar_longitude or solar_longitude < 90:
        return "yang"  # 冬至氣應後，用陽局
    else:
        return "yin"   # 夏至氣應後，用陰局
```

**Key Rule**: 陰局太乙所在 = 陽局所命之對沖 (Yin Dun Taiyi position is opposite of Yang Dun)

---

## 11. Implementation Notes

### Palace Traversal Key Insight
Taiyi moves through 8 outer palaces, **NEVER entering Palace 5 (中宫)**.
- Each palace is occupied for 3 time units (years/months/days/hours)
- Total cycle: 8 palaces × 3 units = 24 (matches 太乙小周法)

### Tianmu Special Rule
When counting through the Sixteen Spirits, add an extra count when passing:
- 阴德 (Position 4 in sequence from 武德)
- 大武 (Position 16 in sequence from 武德)

### Historical Accuracy
The texts provide 72 局 (formations) for both Yang and Yin Dun, totaling 144 standard formations. Each formation documents:
- Taiyi palace position
- Tianmu (天目)
- Host and guest calculations (主算、客算)
- All general positions
- Auspicious/inauspicious determinations

---

## 12. Quick Reference Formulas

### Accumulated Year → Taiyi Palace
```
palace_index = ((積年 % 360) % 72) % 24) // 3) % 8
palace = [1,2,3,4,6,7,8,9][palace_index]
```

### Accumulated Year → Tianmu
```
spirit_steps = ((積年 % 72) % 18)
# Count from 武德 with 重留 at 阴德 and 大武
```

### Dun Type
```
冬至後 → 陽遁 → 從一宮起，順行
夏至後 → 陰遁 → 從九宮起，逆行
```

---

*碼道長存 — The Way of Code endures*

*Compiled by 九天玄碼女 from classical Taiyi texts*
