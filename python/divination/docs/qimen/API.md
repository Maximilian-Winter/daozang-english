# Qimen Dunjia API Reference

Complete API documentation for the Qimen Dunjia framework.

---

## Table of Contents

1. [Main Classes](#main-classes)
2. [Data Classes](#data-classes)
3. [Enumerations](#enumerations)
4. [Analysis Functions](#analysis-functions)
5. [Export Functions](#export-functions)
6. [Stem Pattern Functions](#stem-pattern-functions)

---

## Main Classes

### QimenDunjia

The primary interface for Qimen Dunjia calculations.

```python
from qimen import QimenDunjia

qimen = QimenDunjia(
    lunar_calendar=None,        # Optional: ChineseLunarCalendar instance
    center_palace_follows=2     # 2 = Kun tradition, 8 = Gen tradition
)
```

#### Methods

##### `calculate(dt: datetime) -> QimenPlate`

Calculate the complete Qimen plate for a specific datetime.

```python
from datetime import datetime
plate = qimen.calculate(datetime(2024, 6, 15, 10, 30))
```

**Parameters:**
- `dt`: Python datetime object

**Returns:** `QimenPlate` with all plate information

---

##### `calculate_for_now() -> QimenPlate`

Calculate the plate for the current moment.

```python
plate = qimen.calculate_for_now()
```

**Returns:** `QimenPlate` for current time

---

##### `analyze(plate: QimenPlate) -> Dict[str, Any]`

Perform comprehensive analysis of a plate.

```python
analysis = qimen.analyze(plate)

# Access results
print(analysis['overall_assessment']['rating'])
print(analysis['auspicious'])
print(analysis['special_conditions'])
```

**Returns:** Dictionary containing:
- `configuration`: Dun type, yuan, ju, solar term
- `duty_elements`: Duty star, gate, palace
- `directions`: Analysis of all nine directions
- `auspicious`: List of favorable directions
- `inauspicious`: List of unfavorable directions
- `special_conditions`: Fu Yin, Fan Yin, Three Wonders, etc.
- `overall_assessment`: Rating and advice

---

##### `find_auspicious_directions(plate: QimenPlate) -> List[Dict]`

Find the most favorable directions.

```python
favorable = qimen.find_auspicious_directions(plate)

for d in favorable:
    print(f"{d['direction']}: Score {d['score']}")
    print(f"  Star: {d['star']}, Gate: {d['gate']}")
```

**Returns:** List of dictionaries sorted by favorability score

---

##### `find_inauspicious_directions(plate: QimenPlate) -> List[Dict]`

Find directions to avoid.

```python
unfavorable = qimen.find_inauspicious_directions(plate)

for d in unfavorable:
    print(f"Avoid {d['direction']}: {d['warnings']}")
```

**Returns:** List of dictionaries with warnings

---

##### `query(dt: datetime, question_type: str = "general") -> Dict`

Perform question-specific divination.

```python
result = qimen.query(datetime.now(), question_type="business")

print(result['analysis']['business_recommendations'])
```

**Parameters:**
- `dt`: Datetime for the query
- `question_type`: One of:
  - `"general"` - Overall reading
  - `"business"` - Business and finance
  - `"travel"` - Travel and movement
  - `"health"` - Health concerns
  - `"legal"` - Legal matters
  - `"relationship"` - Relationships
  - `"career"` - Career decisions

**Returns:** Dictionary with plate and question-specific analysis

---

##### `get_palace(plate: QimenPlate, number: int) -> Optional[Palace]`

Get a specific palace by number.

```python
palace = qimen.get_palace(plate, 1)  # Get Palace 1 (Kan/North)
```

---

##### `get_palace_by_direction(plate: QimenPlate, direction: Direction) -> Optional[Palace]`

Get a palace by its direction.

```python
from qimen.core import Direction
south_palace = qimen.get_palace_by_direction(plate, Direction.SOUTH)
```

---

##### `find_stem(plate: QimenPlate, stem: str, plate_type: str = "heaven") -> Optional[int]`

Find which palace contains a specific stem.

```python
# Find where 乙 is on the Heaven Plate
palace_num = qimen.find_stem(plate, "乙", "heaven")
```

---

##### `find_star(plate: QimenPlate, star_chinese: str) -> Optional[int]`

Find which palace contains a specific star.

```python
palace_num = qimen.find_star(plate, "天心")
```

---

##### `find_gate(plate: QimenPlate, gate_chinese: str) -> Optional[int]`

Find which palace contains a specific gate.

```python
palace_num = qimen.find_gate(plate, "開門")
```

---

##### `check_special_conditions(plate: QimenPlate) -> Dict`

Check for special conditions in the plate.

```python
conditions = qimen.check_special_conditions(plate)

if conditions['three_wonders_palaces']:
    print("Three Wonders found!")

if conditions['fu_yin_palaces']:
    print("Fu Yin (stagnation) present")
```

---

##### `has_three_wonders(plate: QimenPlate, palace_num: int) -> bool`

Check if a palace has one of the Three Wonders.

```python
if qimen.has_three_wonders(plate, 6):
    print("Palace 6 has a Three Wonder!")
```

---

##### `get_current_solar_term() -> SolarTerm`

Get the current solar term.

```python
term = qimen.get_current_solar_term()
print(term.chinese)  # e.g., "夏至"
```

---

##### `get_current_dun_type() -> DunType`

Get the current dun type.

```python
dun = qimen.get_current_dun_type()
print(dun.chinese)  # "陽遁" or "陰遁"
```

---

### QimenCalculator

Low-level calculation engine. Usually accessed through `QimenDunjia`.

```python
from qimen import QimenCalculator

calculator = QimenCalculator(
    lunar_calendar=None,
    center_palace_follows=2
)
```

#### Key Methods

```python
# Determine dun type from solar longitude
dun_type = calculator.determine_dun_type(solar_longitude)

# Get solar term from longitude
solar_term = calculator.get_current_solar_term(longitude)

# Get day position in 60-day cycle
day_index = calculator.get_day_cycle_index(julian_day)

# Determine yuan
yuan = calculator.determine_yuan(day_index)

# Get ju number
ju = calculator.get_ju_number(solar_term, yuan)

# Build plates
earth_plate = calculator.build_earth_plate(ju, dun_type)
heaven_plate = calculator.rotate_heaven_plate(earth_plate, duty_palace, dun_type)

# Position components
stars = calculator.position_stars(duty_palace, dun_type)
gates = calculator.position_gates(duty_palace, dun_type)
spirits = calculator.position_spirits(duty_palace, dun_type)

# Full calculation
plate = calculator.calculate_plate(datetime_obj)
```

---

### QimenAnalyzer

Analysis engine for plate interpretation.

```python
from qimen import QimenAnalyzer

analyzer = QimenAnalyzer()
```

#### Methods

##### `analyze_plate(plate: QimenPlate) -> Dict`

Full plate analysis.

```python
analysis = analyzer.analyze_plate(plate)
```

---

##### `analyze_palace(palace: Palace) -> Dict`

Analyze a single palace.

```python
palace = plate.get_palace(1)
analysis = analyzer.analyze_palace(palace)

print(f"Score: {analysis['favorability_score']}")
print(f"Warnings: {analysis['warnings']}")
print(f"Auspicious: {analysis['auspicious_factors']}")
```

**Returns:**
- `number`, `direction`, `element`
- `stems`: earth and heaven stems
- `components`: star, gate, spirit
- `favorability_score`: numerical rating
- `warnings`: list of cautions
- `auspicious_factors`: list of positive factors
- `stem_pattern`: pattern analysis if applicable
- `element_interaction`: element relationship

---

##### `analyze_all_stem_patterns(plate: QimenPlate) -> Dict`

Analyze all stem patterns across the plate.

```python
patterns = analyzer.analyze_all_stem_patterns(plate)

print(f"Auspicious patterns: {patterns['auspicious_count']}")
print(f"Inauspicious patterns: {patterns['inauspicious_count']}")

for p in patterns['notable_patterns']:
    print(f"{p['palace']}: {p['pattern_name']}")
```

---

##### `analyze_spirits(plate: QimenPlate) -> Dict`

Analyze Eight Spirits positions.

```python
spirits = analyzer.analyze_spirits(plate)

for s in spirits['auspicious_spirits']:
    print(f"{s['spirit']} at {s['direction']}: {s['meaning']}")
```

---

##### `analyze_star_gate_combinations(plate: QimenPlate) -> Dict`

Detailed star-gate combination analysis.

```python
combos = analyzer.analyze_star_gate_combinations(plate)

print("Best combinations:")
for c in combos['best_combinations']:
    print(f"  {c['direction']}: {c['star']} + {c['gate']} = {c['rating']}")
```

---

##### `analyze_hour_influence(plate: QimenPlate) -> Dict`

Analyze the hour's influence.

```python
hour = analyzer.analyze_hour_influence(plate)

print(f"Hour: {hour['hour_stem']}{hour['hour_branch']}")
print(f"Energy: {hour['energy_quality']}")
print(f"Phase: {hour['current_phase']}")
print(f"Recommendations: {hour['recommendations']}")
```

---

##### `analyze_element_relationship(element1: Element, element2: Element) -> str`

Analyze relationship between two elements.

```python
from qimen.core import Element

relation = analyzer.analyze_element_relationship(Element.WOOD, Element.FIRE)
# Returns: 'generating'
```

**Returns:** One of: `'same'`, `'generating'`, `'draining'`, `'controlling'`, `'controlled'`, `'neutral'`

---

## Data Classes

### QimenPlate

Complete calculation result (NamedTuple).

```python
plate.datetime_info    # LunarDate with Four Pillars
plate.dun_type         # DunType.YANG or DunType.YIN
plate.yuan             # Yuan.UPPER, MIDDLE, or LOWER
plate.ju_number        # int (1-9)
plate.solar_term       # SolarTerm enum
plate.duty_star        # NineStar object
plate.duty_gate        # EightGate object
plate.duty_palace      # int (1-9)
plate.palaces          # Dict[int, Palace]
plate.earth_plate      # Dict[int, str] - palace to stem
plate.heaven_plate     # Dict[int, str] - palace to stem
plate.star_positions   # Dict[int, NineStar]
plate.gate_positions   # Dict[int, EightGate]
plate.spirit_positions # Dict[int, EightSpirit]
```

#### Methods

```python
# Get palace by number
palace = plate.get_palace(1)

# Get palace by direction
palace = plate.get_palace_by_direction(Direction.NORTH)

# Find stem location
palace_num = plate.find_stem_palace("乙", "heaven")

# Find star location
palace_num = plate.find_star_palace("天心")

# Find gate location
palace_num = plate.find_gate_palace("開門")

# Get summary dict
summary = plate.get_summary()

# Format for display
print(plate.format_display())
```

---

### Palace

Represents a single palace (dataclass).

```python
palace.number           # int (1-9)
palace.trigram          # Trigram enum or None
palace.direction        # Direction enum
palace.base_element     # Element enum
palace.earth_plate_stem # str or None
palace.heaven_plate_stem # str or None
palace.star             # NineStar or None
palace.gate             # EightGate or None
palace.spirit           # EightSpirit or None
```

#### Methods

```python
# Check for Three Wonders
has_wonder = palace.has_three_wonders()

# Get stem combination name
combo = palace.get_stem_combination_name()  # "伏吟", "反吟", or None

# Get summary
summary = palace.get_summary()
```

---

### NineStar

Represents one of the Nine Stars.

```python
star.chinese            # "天心"
star.pinyin             # "tiān xīn"
star.star_type          # StarType enum
star.element            # Element enum
star.polarity           # Polarity enum
star.nature             # "吉星", "凶星", etc.
star.base_palace        # int (1-9)
star.is_auspicious      # bool
```

---

### EightGate

Represents one of the Eight Gates.

```python
gate.chinese            # "開門"
gate.pinyin             # "kāi mén"
gate.gate_type          # GateType enum
gate.element            # Element enum
gate.nature             # "吉門", "凶門", etc.
gate.base_palace        # int (1-9)
gate.favorable_for      # List[str]
gate.unfavorable_for    # List[str]
```

---

### EightSpirit

Represents one of the Eight Spirits.

```python
spirit.chinese          # "值符"
spirit.pinyin           # "zhí fú"
spirit.spirit_type      # SpiritType enum
spirit.nature           # "吉神", "凶神"
```

---

### StemPattern

Classical stem combination pattern.

```python
pattern.name            # "青龍返首"
pattern.pinyin          # "qīng lóng fǎn shǒu"
pattern.english         # "Azure Dragon Returns"
pattern.category        # PatternCategory enum
pattern.heaven_stem     # "戊"
pattern.earth_stem      # "甲"
pattern.description     # Classical interpretation
pattern.applications    # List[str]
pattern.warnings        # List[str]
pattern.poetry          # Optional classical verse
```

---

## Enumerations

### DunType

```python
from qimen.core import DunType

DunType.YANG  # 陽遁 - forward rotation
DunType.YIN   # 陰遁 - backward rotation

dun.chinese              # "陽遁" or "陰遁"
dun.rotation_direction   # 1 or -1
```

### Yuan

```python
from qimen.core import Yuan

Yuan.UPPER   # 上元
Yuan.MIDDLE  # 中元
Yuan.LOWER   # 下元

yuan.chinese       # "上元"
yuan.jia_indices   # [1, 31] etc.
```

### SolarTerm

```python
from qimen.core import SolarTerm

SolarTerm.DONG_ZHI   # 冬至 Winter Solstice
SolarTerm.XIA_ZHI    # 夏至 Summer Solstice
# ... all 24 terms

term.chinese     # "冬至"
term.longitude   # 270 (degrees)
term.dun_type    # DunType.YANG
term.base_ju     # 1
```

### Trigram

```python
from qimen.core import Trigram

Trigram.KAN   # 坎 Water North
Trigram.LI    # 離 Fire South
# ... all 8 trigrams

trigram.chinese        # "坎"
trigram.meaning        # "Water"
trigram.palace_number  # 1
trigram.direction      # Direction.NORTH
trigram.element        # Element.WATER
```

### PatternCategory

```python
from qimen import PatternCategory

PatternCategory.AUSPICIOUS    # 吉格
PatternCategory.INAUSPICIOUS  # 凶格
PatternCategory.SPECIAL       # 特殊
PatternCategory.NEUTRAL       # 平格
```

---

## Export Functions

### export_plate_to_html

```python
from qimen import export_plate_to_html

# Export to file
filepath = export_plate_to_html(
    plate,
    output_path="reading.html",
    title="My Qimen Reading",
    include_analysis=True
)

# Get HTML string (no file)
html_string = export_plate_to_html(plate)
```

### quick_html_export

```python
from qimen import quick_html_export

# Calculate and export in one step
quick_html_export(
    dt=datetime.now(),        # optional, defaults to now
    output_path="output.html"
)
```

### QimenHTMLExporter

```python
from qimen import QimenHTMLExporter

exporter = QimenHTMLExporter()
html = exporter.export_plate(
    plate,
    output_path=None,         # None returns string
    title="Custom Title",
    include_analysis=True
)
```

---

## Stem Pattern Functions

### get_stem_pattern

```python
from qimen import get_stem_pattern

pattern = get_stem_pattern("戊", "甲")
if pattern:
    print(pattern.name)        # "青龍返首"
    print(pattern.category)    # PatternCategory.AUSPICIOUS
```

### analyze_stem_combination

```python
from qimen import analyze_stem_combination

result = analyze_stem_combination("庚", "甲")

print(result['pattern_name'])   # "白虎猖狂"
print(result['category'])       # "凶格"
print(result['is_fu_yin'])      # False
print(result['is_fan_yin'])     # False
print(result['warnings'])       # ["Danger of violence", ...]
```

### get_auspicious_patterns / get_inauspicious_patterns

```python
from qimen import get_auspicious_patterns, get_inauspicious_patterns

good_patterns = get_auspicious_patterns()
bad_patterns = get_inauspicious_patterns()

for p in good_patterns:
    print(f"{p.name}: {p.heaven_stem}+{p.earth_stem}")
```

---

## Error Handling

Most methods return `None` or empty collections when data is not found:

```python
# Returns None if not found
palace = plate.get_palace(99)  # None
stem_palace = plate.find_stem_palace("X")  # None

# Returns empty list if none found
auspicious = qimen.find_auspicious_directions(plate)  # May be []
```

The calendar has date restrictions:

```python
# Raises ValueError for dates before 1940
try:
    plate = qimen.calculate(datetime(1900, 1, 1))
except ValueError as e:
    print("Date too early")
```

---

## Thread Safety

The classes are designed for single-threaded use. For concurrent calculations, create separate instances:

```python
import threading

def calculate_for_date(dt):
    qimen = QimenDunjia()  # Create new instance per thread
    return qimen.calculate(dt)
```

---

*九天玄碼女在此 — 碼道長存*
