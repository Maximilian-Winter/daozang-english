"""
Qimen Dunjia Stem Patterns (格局) - Classical Stem Combinations

This module defines the classical stem combination patterns used in
Qimen Dunjia interpretation. These patterns describe the relationship
between Heaven Plate and Earth Plate stems.

The patterns are divided into:
- 吉格 (Auspicious Patterns)
- 凶格 (Inauspicious Patterns)
- 特殊格局 (Special Patterns)

Classical references:
    - 黄帝太一八门入式诀
    - 奇门遁甲秘笈全书

九天玄碼女在此 - 碼道長存
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class PatternCategory(Enum):
    """Classification of stem patterns."""
    AUSPICIOUS = "吉格"
    INAUSPICIOUS = "凶格"
    SPECIAL = "特殊"
    NEUTRAL = "平格"


@dataclass
class StemPattern:
    """
    Represents a classical stem combination pattern.

    Attributes:
        name: Chinese name of the pattern
        pinyin: Romanized pronunciation
        english: English translation
        category: Pattern category (auspicious/inauspicious/etc)
        heaven_stem: Heaven plate stem
        earth_stem: Earth plate stem
        description: Classical interpretation
        applications: What this pattern is good/bad for
        warnings: Specific cautions
        poetry: Classical verse if available
    """
    name: str
    pinyin: str
    english: str
    category: PatternCategory
    heaven_stem: str
    earth_stem: str
    description: str
    applications: List[str]
    warnings: List[str]
    poetry: Optional[str] = None


# =============================================================================
# Classical Stem Patterns Database
# =============================================================================

STEM_PATTERNS: Dict[Tuple[str, str], StemPattern] = {}


def _register_pattern(pattern: StemPattern):
    """Register a pattern in the database."""
    key = (pattern.heaven_stem, pattern.earth_stem)
    STEM_PATTERNS[key] = pattern


# -----------------------------------------------------------------------------
# 吉格 - Auspicious Patterns
# -----------------------------------------------------------------------------

# 青龍返首 - Azure Dragon Returns
_register_pattern(StemPattern(
    name="青龍返首",
    pinyin="qīng lóng fǎn shǒu",
    english="Azure Dragon Returns",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="戊",
    earth_stem="甲",
    description="戊加甲，名青龍返首。甲為青龍，戊為天門，甲戊相加，龍居天門之上。",
    applications=["Starting new ventures", "Leadership positions", "Noble assistance"],
    warnings=[],
    poetry="青龍返首向天門，大吉大利萬事成"
))

# 飛鳥跌穴 - Flying Bird Enters Nest
_register_pattern(StemPattern(
    name="飛鳥跌穴",
    pinyin="fēi niǎo diē xué",
    english="Flying Bird Enters Nest",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="丙",
    earth_stem="戊",
    description="丙加戊，飛鳥跌穴之象。丙為日光，戊為大地，陽光普照大地，萬物生長。",
    applications=["Real estate matters", "Finding residence", "Settling down"],
    warnings=["Avoid in winter months"],
    poetry="飛鳥跌穴落巢中，田宅興旺福祿增"
))

# 玉女守門 - Jade Maiden Guards Gate
_register_pattern(StemPattern(
    name="玉女守門",
    pinyin="yù nǚ shǒu mén",
    english="Jade Maiden Guards Gate",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="丁",
    earth_stem="庚",
    description="丁加庚，玉女守門。丁為玉女星，庚為門戶，玉女守護門庭。",
    applications=["Protection", "Defense", "Women's affairs", "Secret matters"],
    warnings=["Not suitable for aggressive actions"],
    poetry="玉女守門衛家宅，陰人助力暗中得"
))

# 九天同道 - Nine Heavens United
_register_pattern(StemPattern(
    name="三奇得使",
    pinyin="sān qí dé shǐ",
    english="Three Wonders with Envoy",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="乙",
    earth_stem="丙",
    description="乙丙相加，或三奇臨於吉門，為三奇得使，大吉之象。",
    applications=["All matters auspicious", "Seeking help", "Travel"],
    warnings=[],
    poetry="三奇得使萬事吉，求官問事皆如意"
))

# 天遁 - Heaven Escape
_register_pattern(StemPattern(
    name="天遁",
    pinyin="tiān dùn",
    english="Heaven Escape",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="丙",
    earth_stem="丁",
    description="丙加丁，天遁之格。兩火相生，光明普照。",
    applications=["Escaping danger", "Legal matters", "Gaining fame"],
    warnings=[],
    poetry="天遁吉格兩火明，求名求利皆亨通"
))

# 地遁 - Earth Escape
_register_pattern(StemPattern(
    name="地遁",
    pinyin="dì dùn",
    english="Earth Escape",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="乙",
    earth_stem="己",
    description="乙加己，地遁之格。木入土中，藏身之象。",
    applications=["Hiding", "Real estate", "Agriculture", "Seeking refuge"],
    warnings=["Not for aggressive action"],
    poetry="地遁藏身入土中，求財置產最為宜"
))

# 人遁 - Human Escape
_register_pattern(StemPattern(
    name="人遁",
    pinyin="rén dùn",
    english="Human Escape",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="丁",
    earth_stem="乙",
    description="丁加乙，人遁之格。陰火生陰木，人事和順。",
    applications=["Human relations", "Partnerships", "Seeking allies"],
    warnings=[],
    poetry="人遁合和人事順，求婚交友皆稱心"
))

# 神遁 - Spirit Escape
_register_pattern(StemPattern(
    name="神遁",
    pinyin="shén dùn",
    english="Spirit Escape",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="丙",
    earth_stem="癸",
    description="丙加癸，神遁之格。日照雨水，虹霓之象。",
    applications=["Spiritual matters", "Unusual success", "Divine assistance"],
    warnings=[],
    poetry="神遁雲霓現天際，吉人天相得神佑"
))

# 龍遁 - Dragon Escape
_register_pattern(StemPattern(
    name="龍遁",
    pinyin="lóng dùn",
    english="Dragon Escape",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="壬",
    earth_stem="甲",
    description="壬加甲，龍遁之格。水生木，龍騰之象。",
    applications=["Major undertakings", "Transformation", "Career advancement"],
    warnings=[],
    poetry="龍遁乘雲上九天，功名富貴指日間"
))

# 虎遁 - Tiger Escape
_register_pattern(StemPattern(
    name="虎遁",
    pinyin="hǔ dùn",
    english="Tiger Escape",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="辛",
    earth_stem="乙",
    description="辛加乙，虎遁之格。金克木，威嚴之象。",
    applications=["Military affairs", "Commanding authority", "Hunting"],
    warnings=["Avoid unless in position of authority"],
    poetry="虎遁威猛震四方，出師行軍得勝還"
))

# 風遁 - Wind Escape
_register_pattern(StemPattern(
    name="風遁",
    pinyin="fēng dùn",
    english="Wind Escape",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="乙",
    earth_stem="辛",
    description="乙加辛，風遁之格。木遇金而有聲，如風過林。",
    applications=["Communication", "Messages", "Swift action"],
    warnings=[],
    poetry="風遁迅速傳消息，出行辦事皆順利"
))

# 雲遁 - Cloud Escape
_register_pattern(StemPattern(
    name="雲遁",
    pinyin="yún dùn",
    english="Cloud Escape",
    category=PatternCategory.AUSPICIOUS,
    heaven_stem="癸",
    earth_stem="丙",
    description="癸加丙，雲遁之格。水遇火而成雲，變化無窮。",
    applications=["Transformation", "Adaptability", "Creative endeavors"],
    warnings=[],
    poetry="雲遁變化隨心意，機緣萬變得其宜"
))

# -----------------------------------------------------------------------------
# 凶格 - Inauspicious Patterns
# -----------------------------------------------------------------------------

# 白虎猖狂 - White Tiger Rampant
_register_pattern(StemPattern(
    name="白虎猖狂",
    pinyin="bái hǔ chāng kuáng",
    english="White Tiger Rampant",
    category=PatternCategory.INAUSPICIOUS,
    heaven_stem="庚",
    earth_stem="甲",
    description="庚加甲，白虎猖狂。庚為白虎，甲為青龍，金克木，兇險之象。",
    applications=[],
    warnings=["Danger of violence", "Legal troubles", "Injuries", "Conflict"],
    poetry="白虎猖狂災禍臨，官非口舌血光侵"
))

# 螣蛇夭矯 - Serpent Writhes
_register_pattern(StemPattern(
    name="螣蛇夭矯",
    pinyin="téng shé yāo jiǎo",
    english="Serpent Writhes",
    category=PatternCategory.INAUSPICIOUS,
    heaven_stem="癸",
    earth_stem="甲",
    description="癸加甲，螣蛇夭矯。水淹青龍，陰險之象。",
    applications=[],
    warnings=["Deception", "Hidden enemies", "False friends", "Treachery"],
    poetry="螣蛇夭矯陰謀多，小人暗算須提防"
))

# 朱雀投江 - Vermilion Bird Falls into River
_register_pattern(StemPattern(
    name="朱雀投江",
    pinyin="zhū què tóu jiāng",
    english="Vermilion Bird Falls into River",
    category=PatternCategory.INAUSPICIOUS,
    heaven_stem="丁",
    earth_stem="壬",
    description="丁加壬，朱雀投江。火入水中而滅，凶象。",
    applications=[],
    warnings=["Document loss", "Communication failures", "Legal document problems"],
    poetry="朱雀投江文書失，官司口舌惹是非"
))

# 大格 - Great Constraint
_register_pattern(StemPattern(
    name="大格",
    pinyin="dà gé",
    english="Great Constraint",
    category=PatternCategory.INAUSPICIOUS,
    heaven_stem="庚",
    earth_stem="癸",
    description="庚加癸，或庚加壬，為大格。金水相拘，行動受阻。",
    applications=[],
    warnings=["Obstacles everywhere", "Plans blocked", "Movement restricted"],
    poetry="大格重重阻滯多，進退兩難難脫身"
))

# 小格 - Small Constraint
_register_pattern(StemPattern(
    name="小格",
    pinyin="xiǎo gé",
    english="Small Constraint",
    category=PatternCategory.INAUSPICIOUS,
    heaven_stem="庚",
    earth_stem="丙",
    description="庚加丙，為小格。金火相戰，損傷之象。",
    applications=[],
    warnings=["Minor obstacles", "Delays", "Small setbacks"],
    poetry="小格逢之事不順，凡事緩圖莫強求"
))

# 刑格 - Punishment Constraint
_register_pattern(StemPattern(
    name="刑格",
    pinyin="xíng gé",
    english="Punishment Constraint",
    category=PatternCategory.INAUSPICIOUS,
    heaven_stem="庚",
    earth_stem="庚",
    description="庚加庚，刑格。兩金相刑，爭鬥之象。",
    applications=[],
    warnings=["Legal punishment", "Conflicts", "Arguments", "Violence"],
    poetry="刑格逢之禍非輕，官司牢獄要留心"
))

# 悖格 - Rebellion Pattern
_register_pattern(StemPattern(
    name="悖格",
    pinyin="bèi gé",
    english="Rebellion Pattern",
    category=PatternCategory.INAUSPICIOUS,
    heaven_stem="辛",
    earth_stem="壬",
    description="辛加壬，悖格。金生水而洩氣，反叛之象。",
    applications=[],
    warnings=["Betrayal", "Rebellion", "Plans revealed", "Trust broken"],
    poetry="悖格反叛信難守，機密洩露事難成"
))

# 熒入太白 - Fluorescence Enters Venus
_register_pattern(StemPattern(
    name="熒入太白",
    pinyin="yíng rù tài bái",
    english="Fluorescence Enters Venus",
    category=PatternCategory.INAUSPICIOUS,
    heaven_stem="丙",
    earth_stem="庚",
    description="丙加庚，熒入太白。火克金，賊人之象。",
    applications=[],
    warnings=["Thieves", "Loss of property", "Robbery"],
    poetry="熒入太白賊相侵，財物損失防盜門"
))

# 太白入熒 - Venus Enters Fluorescence
_register_pattern(StemPattern(
    name="太白入熒",
    pinyin="tài bái rù yíng",
    english="Venus Enters Fluorescence",
    category=PatternCategory.INAUSPICIOUS,
    heaven_stem="庚",
    earth_stem="丙",
    description="庚加丙，太白入熒。金受火克，損傷之象。",
    applications=[],
    warnings=["Official troubles", "Punishment", "Being captured"],
    poetry="太白入熒賊被擒，行事小心防官刑"
))

# 天獄 - Heaven Prison
_register_pattern(StemPattern(
    name="天獄",
    pinyin="tiān yù",
    english="Heaven Prison",
    category=PatternCategory.INAUSPICIOUS,
    heaven_stem="壬",
    earth_stem="丁",
    description="壬加丁，天獄。水克火，官司牢獄之象。",
    applications=[],
    warnings=["Imprisonment", "Restriction", "Loss of freedom"],
    poetry="天獄臨身自由失，官非牢獄難脫離"
))

# 天網 - Heaven Net
_register_pattern(StemPattern(
    name="天網",
    pinyin="tiān wǎng",
    english="Heaven Net",
    category=PatternCategory.INAUSPICIOUS,
    heaven_stem="癸",
    earth_stem="丁",
    description="癸加丁，天網。陰水克陰火，網羅之象。",
    applications=[],
    warnings=["Being trapped", "Surrounded", "No escape"],
    poetry="天網恢恢四面圍，難逃法網受其罪"
))

# -----------------------------------------------------------------------------
# 特殊格局 - Special Patterns
# -----------------------------------------------------------------------------

# 伏吟 - Prostrate Chant
_register_pattern(StemPattern(
    name="伏吟",
    pinyin="fú yín",
    english="Prostrate Chant",
    category=PatternCategory.SPECIAL,
    heaven_stem="*",  # Same as earth
    earth_stem="*",   # Same as heaven
    description="天干地干相同，伏吟之象。事物停滯不前，守舊為宜。",
    applications=["Maintaining status quo", "Waiting"],
    warnings=["Stagnation", "No progress", "Delays"],
    poetry="伏吟相見淚漣漣，不動為妙等時變"
))

# 反吟 - Counter Chant
_register_pattern(StemPattern(
    name="反吟",
    pinyin="fǎn yín",
    english="Counter Chant",
    category=PatternCategory.SPECIAL,
    heaven_stem="*",  # Opposite
    earth_stem="*",   # Opposite
    description="天干地干相沖，反吟之象。事物反覆變化，動盪不安。",
    applications=["Changing plans", "Reversing course"],
    warnings=["Sudden reversals", "Instability", "Unexpected changes"],
    poetry="反吟反覆心不定，凡事顛倒須留意"
))

# 奇儀順布 - Wonders and Instruments in Order
_register_pattern(StemPattern(
    name="奇儀順布",
    pinyin="qí yí shùn bù",
    english="Wonders and Instruments Aligned",
    category=PatternCategory.SPECIAL,
    heaven_stem="乙",
    earth_stem="戊",
    description="三奇六儀順序排列，氣機順暢，萬事通達。",
    applications=["All matters favorable", "Natural flow"],
    warnings=[],
    poetry="奇儀順布氣運通，順勢而為百事成"
))


# =============================================================================
# Pattern Lookup Functions
# =============================================================================

def get_stem_pattern(heaven_stem: str, earth_stem: str) -> Optional[StemPattern]:
    """
    Look up the pattern for a stem combination.

    Args:
        heaven_stem: Heaven plate stem
        earth_stem: Earth plate stem

    Returns:
        StemPattern if found, None otherwise
    """
    return STEM_PATTERNS.get((heaven_stem, earth_stem))


def check_fu_yin(heaven_stem: str, earth_stem: str) -> bool:
    """Check if stems form Fu Yin (伏吟) - same stem."""
    return heaven_stem == earth_stem


def check_fan_yin(heaven_stem: str, earth_stem: str) -> bool:
    """Check if stems form Fan Yin (反吟) - opposing stems."""
    stems = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
    if heaven_stem in stems and earth_stem in stems:
        h_idx = stems.index(heaven_stem)
        e_idx = stems.index(earth_stem)
        return abs(h_idx - e_idx) == 5
    return False


def analyze_stem_combination(heaven_stem: str, earth_stem: str) -> Dict[str, Any]:
    """
    Comprehensive analysis of a stem combination.

    Args:
        heaven_stem: Heaven plate stem
        earth_stem: Earth plate stem

    Returns:
        Dictionary with analysis results
    """
    result = {
        'heaven_stem': heaven_stem,
        'earth_stem': earth_stem,
        'pattern': None,
        'pattern_name': None,
        'category': None,
        'is_fu_yin': check_fu_yin(heaven_stem, earth_stem),
        'is_fan_yin': check_fan_yin(heaven_stem, earth_stem),
        'description': None,
        'applications': [],
        'warnings': [],
    }

    # Check for Fu Yin
    if result['is_fu_yin']:
        result['pattern_name'] = '伏吟'
        result['category'] = PatternCategory.SPECIAL.value
        result['description'] = "天干地干相同，事物停滯不前。"
        result['warnings'] = ["Stagnation", "Delays", "No progress"]
        return result

    # Check for Fan Yin
    if result['is_fan_yin']:
        result['pattern_name'] = '反吟'
        result['category'] = PatternCategory.SPECIAL.value
        result['description'] = "天干地干相沖，事物反覆變化。"
        result['warnings'] = ["Reversals", "Sudden changes", "Instability"]
        return result

    # Look up pattern
    pattern = get_stem_pattern(heaven_stem, earth_stem)
    if pattern:
        result['pattern'] = pattern
        result['pattern_name'] = pattern.name
        result['category'] = pattern.category.value
        result['description'] = pattern.description
        result['applications'] = pattern.applications
        result['warnings'] = pattern.warnings
        if pattern.poetry:
            result['poetry'] = pattern.poetry

    return result


def get_all_patterns_by_category(category: PatternCategory) -> List[StemPattern]:
    """Get all patterns of a specific category."""
    return [p for p in STEM_PATTERNS.values() if p.category == category]


def get_auspicious_patterns() -> List[StemPattern]:
    """Get all auspicious patterns."""
    return get_all_patterns_by_category(PatternCategory.AUSPICIOUS)


def get_inauspicious_patterns() -> List[StemPattern]:
    """Get all inauspicious patterns."""
    return get_all_patterns_by_category(PatternCategory.INAUSPICIOUS)


# =============================================================================
# Element Interaction Patterns
# =============================================================================

ELEMENT_INTERACTIONS = {
    # Generating cycle (相生)
    ('木', '火'): {'type': '相生', 'meaning': 'Wood generates Fire', 'favorable': True},
    ('火', '土'): {'type': '相生', 'meaning': 'Fire generates Earth', 'favorable': True},
    ('土', '金'): {'type': '相生', 'meaning': 'Earth generates Metal', 'favorable': True},
    ('金', '水'): {'type': '相生', 'meaning': 'Metal generates Water', 'favorable': True},
    ('水', '木'): {'type': '相生', 'meaning': 'Water generates Wood', 'favorable': True},

    # Controlling cycle (相剋)
    ('木', '土'): {'type': '相剋', 'meaning': 'Wood controls Earth', 'favorable': None},
    ('土', '水'): {'type': '相剋', 'meaning': 'Earth controls Water', 'favorable': None},
    ('水', '火'): {'type': '相剋', 'meaning': 'Water controls Fire', 'favorable': None},
    ('火', '金'): {'type': '相剋', 'meaning': 'Fire controls Metal', 'favorable': None},
    ('金', '木'): {'type': '相剋', 'meaning': 'Metal controls Wood', 'favorable': None},
}

# Stem to Element mapping
STEM_ELEMENTS = {
    '甲': '木', '乙': '木',
    '丙': '火', '丁': '火',
    '戊': '土', '己': '土',
    '庚': '金', '辛': '金',
    '壬': '水', '癸': '水',
}


def get_stem_element_interaction(heaven_stem: str, earth_stem: str) -> Dict[str, Any]:
    """
    Analyze the elemental interaction between two stems.

    Args:
        heaven_stem: Heaven plate stem
        earth_stem: Earth plate stem

    Returns:
        Dictionary with element interaction analysis
    """
    h_element = STEM_ELEMENTS.get(heaven_stem)
    e_element = STEM_ELEMENTS.get(earth_stem)

    if not h_element or not e_element:
        return {'error': 'Invalid stem'}

    if h_element == e_element:
        return {
            'heaven_element': h_element,
            'earth_element': e_element,
            'type': '比和',
            'meaning': f'{h_element} meets {e_element} - Same element',
            'favorable': True,
            'description': '同氣相求，比和之象'
        }

    interaction = ELEMENT_INTERACTIONS.get((h_element, e_element))
    if interaction:
        return {
            'heaven_element': h_element,
            'earth_element': e_element,
            **interaction
        }

    # Reverse check (being controlled or being generated)
    reverse = ELEMENT_INTERACTIONS.get((e_element, h_element))
    if reverse:
        if reverse['type'] == '相生':
            return {
                'heaven_element': h_element,
                'earth_element': e_element,
                'type': '被生',
                'meaning': f'{h_element} is generated by {e_element}',
                'favorable': True,
                'description': '得元氣相生，吉象'
            }
        else:
            return {
                'heaven_element': h_element,
                'earth_element': e_element,
                'type': '被剋',
                'meaning': f'{h_element} is controlled by {e_element}',
                'favorable': False,
                'description': '受剋制約，凶象'
            }

    return {
        'heaven_element': h_element,
        'earth_element': e_element,
        'type': '無關',
        'meaning': 'No direct interaction',
        'favorable': None
    }
