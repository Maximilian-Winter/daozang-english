"""
Da Liu Ren Constants (大六壬常數)

Classical mappings and reference data for Liu Ren divination.
Primary source: 六壬大全 (Ming Dynasty, 郭載騋)

Constants include:
- Twelve Branches and their properties
- Monthly General (月將) mappings to solar terms
- Noble Person (貴人) mappings for day/night
- Day Stem residences (干寄宮)
- Five Element relationships
- Branch clash and harmony relationships
"""

from enum import Enum, IntEnum
from typing import Dict, List, Tuple

# =============================================================================
# Core Sequences (基本序列)
# =============================================================================

# The 12 Earthly Branches in order
TWELVE_BRANCHES: List[str] = [
    '子', '丑', '寅', '卯', '辰', '巳',
    '午', '未', '申', '酉', '戌', '亥'
]

# Branch to index mapping (0-based)
BRANCH_TO_INDEX: Dict[str, int] = {
    branch: i for i, branch in enumerate(TWELVE_BRANCHES)
}

# The 10 Heavenly Stems in order
TEN_STEMS: List[str] = [
    '甲', '乙', '丙', '丁', '戊',
    '己', '庚', '辛', '壬', '癸'
]

# Stem to index mapping (0-based)
STEM_TO_INDEX: Dict[str, int] = {
    stem: i for i, stem in enumerate(TEN_STEMS)
}

# Yang stems (odd index: 甲丙戊庚壬)
YANG_STEMS: List[str] = ['甲', '丙', '戊', '庚', '壬']

# Yin stems (even index: 乙丁己辛癸)
YIN_STEMS: List[str] = ['乙', '丁', '己', '辛', '癸']


# =============================================================================
# Monthly General Mapping (月將配置)
# =============================================================================

# The 12 Monthly Generals (十二月將) with their associated branches
# Format: (Chinese name, Pinyin, Associated Branch, Description)
MONTHLY_GENERALS: List[Tuple[str, str, str, str]] = [
    ('神后', 'shénhòu', '子', '正月將，主水'),
    ('大吉', 'dàjí', '丑', '二月將，主土'),
    ('功曹', 'gōngcáo', '寅', '三月將，主木'),
    ('太衝', 'tàichōng', '卯', '四月將，主木'),
    ('天罡', 'tiāngāng', '辰', '五月將，主土'),
    ('太乙', 'tàiyǐ', '巳', '六月將，主火'),
    ('勝光', 'shèngguāng', '午', '七月將，主火'),
    ('小吉', 'xiǎojí', '未', '八月將，主土'),
    ('傳送', 'chuánsòng', '申', '九月將，主金'),
    ('從魁', 'cóngkuí', '酉', '十月將，主金'),
    ('河魁', 'hékuí', '戌', '十一月將，主土'),
    ('登明', 'dēngmíng', '亥', '十二月將，主水'),
]

# Monthly General by branch lookup
GENERAL_BY_BRANCH: Dict[str, str] = {
    general[2]: general[0] for general in MONTHLY_GENERALS
}

# Branch by General name lookup
BRANCH_BY_GENERAL: Dict[str, str] = {
    general[0]: general[2] for general in MONTHLY_GENERALS
}

# Solar Term to Monthly General mapping
# 月將以中氣換將: Changes at Zhong Qi (middle solar terms)
# Each solar term pair uses the same monthly general
# Format: solar_term_chinese -> (general_name, branch)
SOLAR_TERM_TO_GENERAL: Dict[str, Tuple[str, str]] = {
    # 雨水 to 春分 uses 亥將(登明)
    '雨水': ('登明', '亥'),
    '驚蟄': ('登明', '亥'),
    # 春分 to 穀雨 uses 戌將(河魁)
    '春分': ('河魁', '戌'),
    '清明': ('河魁', '戌'),
    # 穀雨 to 小滿 uses 酉將(從魁)
    '穀雨': ('從魁', '酉'),
    '立夏': ('從魁', '酉'),
    # 小滿 to 夏至 uses 申將(傳送)
    '小滿': ('傳送', '申'),
    '芒種': ('傳送', '申'),
    # 夏至 to 大暑 uses 未將(小吉)
    '夏至': ('小吉', '未'),
    '小暑': ('小吉', '未'),
    # 大暑 to 處暑 uses 午將(勝光)
    '大暑': ('勝光', '午'),
    '立秋': ('勝光', '午'),
    # 處暑 to 秋分 uses 巳將(太乙)
    '處暑': ('太乙', '巳'),
    '白露': ('太乙', '巳'),
    # 秋分 to 霜降 uses 辰將(天罡)
    '秋分': ('天罡', '辰'),
    '寒露': ('天罡', '辰'),
    # 霜降 to 小雪 uses 卯將(太衝)
    '霜降': ('太衝', '卯'),
    '立冬': ('太衝', '卯'),
    # 小雪 to 冬至 uses 寅將(功曹)
    '小雪': ('功曹', '寅'),
    '大雪': ('功曹', '寅'),
    # 冬至 to 大寒 uses 丑將(大吉)
    '冬至': ('大吉', '丑'),
    '小寒': ('大吉', '丑'),
    # 大寒 to 雨水 uses 子將(神后)
    '大寒': ('神后', '子'),
    '立春': ('神后', '子'),
}

# Simplified lookup: just the solar terms that trigger general changes (中氣)
ZHONG_QI_GENERALS: Dict[str, str] = {
    '雨水': '亥',     # 登明
    '春分': '戌',     # 河魁
    '穀雨': '酉',     # 從魁
    '小滿': '申',     # 傳送
    '夏至': '未',     # 小吉
    '大暑': '午',     # 勝光
    '處暑': '巳',     # 太乙
    '秋分': '辰',     # 天罡
    '霜降': '卯',     # 太衝
    '小雪': '寅',     # 功曹
    '冬至': '丑',     # 大吉
    '大寒': '子',     # 神后
}


# =============================================================================
# Noble Person Mapping (貴人配置)
# =============================================================================

# 甲戊庚牛羊，乙己鼠猴鄉，丙丁豬雞位，壬癸兔蛇藏，六辛逢虎馬，此是貴人方
# Day stem determines the position of the Noble Person (天乙貴人)

# Noble Person position for DAYTIME (晝貴人)
NOBLE_PERSON_DAY: Dict[str, str] = {
    '甲': '丑',  # 甲日晝貴在丑
    '戊': '丑',  # 戊日晝貴在丑
    '庚': '丑',  # 庚日晝貴在丑 (牛)
    '乙': '子',  # 乙日晝貴在子 (鼠)
    '己': '子',  # 己日晝貴在子
    '丙': '亥',  # 丙日晝貴在亥 (豬)
    '丁': '酉',  # 丁日晝貴在酉 (雞)
    '壬': '卯',  # 壬日晝貴在卯 (兔)
    '癸': '巳',  # 癸日晝貴在巳 (蛇)
    '辛': '午',  # 辛日晝貴在午 (馬)
}

# Noble Person position for NIGHTTIME (夜貴人)
NOBLE_PERSON_NIGHT: Dict[str, str] = {
    '甲': '未',  # 甲日夜貴在未
    '戊': '未',  # 戊日夜貴在未
    '庚': '未',  # 庚日夜貴在未 (羊)
    '乙': '申',  # 乙日夜貴在申 (猴)
    '己': '申',  # 己日夜貴在申
    '丙': '酉',  # 丙日夜貴在酉
    '丁': '亥',  # 丁日夜貴在亥
    '壬': '巳',  # 壬日夜貴在巳
    '癸': '卯',  # 癸日夜貴在卯
    '辛': '寅',  # 辛日夜貴在寅 (虎)
}


# =============================================================================
# Day Stem Residence (日干寄宮)
# =============================================================================

# Each Heavenly Stem "resides" at a specific Earthly Branch
# Used for calculating the First and Second Lessons
STEM_RESIDENCE: Dict[str, str] = {
    '甲': '寅',  # 甲寄寅
    '乙': '辰',  # 乙寄辰
    '丙': '巳',  # 丙寄巳
    '丁': '未',  # 丁寄未
    '戊': '巳',  # 戊寄巳 (與丙同宮)
    '己': '未',  # 己寄未 (與丁同宮)
    '庚': '申',  # 庚寄申
    '辛': '戌',  # 辛寄戌
    '壬': '亥',  # 壬寄亥
    '癸': '丑',  # 癸寄丑
}


# =============================================================================
# Five Element Relationships (五行關係)
# =============================================================================

# Branch to Element mapping
BRANCH_ELEMENT: Dict[str, str] = {
    '子': '水', '丑': '土', '寅': '木', '卯': '木',
    '辰': '土', '巳': '火', '午': '火', '未': '土',
    '申': '金', '酉': '金', '戌': '土', '亥': '水',
}

# Stem to Element mapping
STEM_ELEMENT: Dict[str, str] = {
    '甲': '木', '乙': '木',
    '丙': '火', '丁': '火',
    '戊': '土', '己': '土',
    '庚': '金', '辛': '金',
    '壬': '水', '癸': '水',
}

# Element generating cycle (相生)
ELEMENT_GENERATES: Dict[str, str] = {
    '木': '火',  # Wood generates Fire
    '火': '土',  # Fire generates Earth
    '土': '金',  # Earth generates Metal
    '金': '水',  # Metal generates Water
    '水': '木',  # Water generates Wood
}

# Element controlling cycle (相克)
ELEMENT_CONTROLS: Dict[str, str] = {
    '木': '土',  # Wood controls Earth
    '土': '水',  # Earth controls Water
    '水': '火',  # Water controls Fire
    '火': '金',  # Fire controls Metal
    '金': '木',  # Metal controls Wood
}


# =============================================================================
# Branch Relationships (地支關係)
# =============================================================================

# Six Clashes (六衝) - Opposing branches
SIX_CLASHES: Dict[str, str] = {
    '子': '午', '午': '子',
    '丑': '未', '未': '丑',
    '寅': '申', '申': '寅',
    '卯': '酉', '酉': '卯',
    '辰': '戌', '戌': '辰',
    '巳': '亥', '亥': '巳',
}

# Six Harmonies (六合) - Harmonious branch pairs
SIX_HARMONIES: Dict[str, str] = {
    '子': '丑', '丑': '子',
    '寅': '亥', '亥': '寅',
    '卯': '戌', '戌': '卯',
    '辰': '酉', '酉': '辰',
    '巳': '申', '申': '巳',
    '午': '未', '未': '午',
}

# Three Harmonies (三合) - Triangular branch combinations forming elements
THREE_HARMONIES: Dict[str, Tuple[str, str, str]] = {
    '水': ('申', '子', '辰'),  # Water frame
    '木': ('亥', '卯', '未'),  # Wood frame
    '火': ('寅', '午', '戌'),  # Fire frame
    '金': ('巳', '酉', '丑'),  # Metal frame
}


# =============================================================================
# Twelve Heavenly Generals (十二天將)
# =============================================================================

class GeneralNature(Enum):
    """Nature of Heavenly General - Auspicious or Inauspicious"""
    AUSPICIOUS = "吉"
    INAUSPICIOUS = "凶"


# The 12 Heavenly Generals with their properties
# Format: (Chinese, Pinyin, Base Branch, Element, Nature, Domain, Description)
TWELVE_GENERALS_DATA: List[Tuple[str, str, str, str, str, str, str]] = [
    ('貴人', 'guìrén', '丑', '土', '吉', '尊貴、貴人相助',
     '天乙貴人，主尊貴、喜慶、貴人扶助'),
    ('騰蛇', 'téngshé', '巳', '火', '凶', '虛驚、怪異、夢寐',
     '主虛驚怪異、火災口舌、憂疑不定'),
    ('朱雀', 'zhūquè', '午', '火', '凶', '口舌、文書、信息',
     '主口舌是非、文書信件、消息傳遞'),
    ('六合', 'liùhé', '卯', '木', '吉', '婚姻、和合、交易',
     '主婚姻、媒妁、交易和合、中人'),
    ('勾陳', 'gōuchén', '辰', '土', '凶', '田土、牢獄、爭訟',
     '主田土、牢獄、訴訟、爭鬥'),
    ('青龍', 'qīnglóng', '寅', '木', '吉', '喜慶、財帛、貴人',
     '主喜慶、財帛、進益、婚姻'),
    ('天后', 'tiānhòu', '未', '土', '吉', '婦女、陰私、暗昧',
     '主婦女、陰私、暗昧、後宮'),
    ('太陰', 'tàiyīn', '酉', '金', '吉', '陰私、藏匿、女性',
     '主陰私、藏匿、女性、暗中相助'),
    ('玄武', 'xuánwǔ', '戌', '土', '凶', '盜賊、失亡、曖昧',
     '主盜賊、失亡、暗昧、欺詐'),  # Note: 戌 not 子 in some traditions
    ('太常', 'tàicháng', '申', '金', '吉', '飲食、衣冠、禮儀',  # Note: 申 not 戌
     '主飲食、宴會、衣裳、禮儀'),
    ('白虎', 'báihǔ', '子', '金', '凶', '喪亡、凶災、血光',  # Note: 子 for some
     '主喪亡、凶災、血光、行兇'),
    ('天空', 'tiānkōng', '亥', '水', '凶', '空亡、欺詐、虛假',
     '主欺詐、虛假、空亡、不實'),
]

# Quick lookup by general name
GENERAL_DATA_BY_NAME: Dict[str, Tuple] = {
    data[0]: data for data in TWELVE_GENERALS_DATA
}

# General sequence for rotation (starting from Noble Person going forward)
GENERAL_FORWARD_SEQUENCE: List[str] = [
    '貴人', '騰蛇', '朱雀', '六合', '勾陳', '青龍'
]

# General sequence for rotation (starting from Noble Person going backward)
GENERAL_BACKWARD_SEQUENCE: List[str] = [
    '貴人', '天后', '太陰', '玄武', '太常', '白虎', '天空'
]

# Complete general sequence in traditional order
GENERAL_FULL_SEQUENCE: List[str] = [
    '貴人', '騰蛇', '朱雀', '六合', '勾陳', '青龍',
    '天空', '白虎', '太常', '玄武', '太陰', '天后'
]


# =============================================================================
# Direction Mappings (方位配置)
# =============================================================================

# Branch to Direction mapping
BRANCH_DIRECTION: Dict[str, str] = {
    '子': '北',
    '丑': '東北',
    '寅': '東北',
    '卯': '東',
    '辰': '東南',
    '巳': '東南',
    '午': '南',
    '未': '西南',
    '申': '西南',
    '酉': '西',
    '戌': '西北',
    '亥': '西北',
}


# =============================================================================
# Special Positions (特殊宮位)
# =============================================================================

# Four Gates (四門) for special analysis
FOUR_GATES: Dict[str, Tuple[str, str]] = {
    '天門': ('戌', '亥'),   # Heaven's Gate - Northwest
    '地戶': ('辰', '巳'),   # Earth's Gate - Southeast
    '人門': ('未', '申'),   # Human Gate - Southwest
    '鬼路': ('丑', '寅'),   # Ghost Path - Northeast
}


# =============================================================================
# Hour Branch Mapping (時支配置)
# =============================================================================

# Standard hour to branch mapping
# Each branch covers a 2-hour period
HOUR_TO_BRANCH: Dict[int, str] = {
    23: '子', 0: '子', 1: '丑', 2: '丑',
    3: '寅', 4: '寅', 5: '卯', 6: '卯',
    7: '辰', 8: '辰', 9: '巳', 10: '巳',
    11: '午', 12: '午', 13: '未', 14: '未',
    15: '申', 16: '申', 17: '酉', 18: '酉',
    19: '戌', 20: '戌', 21: '亥', 22: '亥',
}


# =============================================================================
# Utility Functions (工具函數)
# =============================================================================

def get_branch_index(branch: str) -> int:
    """Get the index (0-11) of a branch."""
    return BRANCH_TO_INDEX.get(branch, 0)


def get_branch_by_index(index: int) -> str:
    """Get the branch at a given index (wraps around)."""
    return TWELVE_BRANCHES[index % 12]


def get_offset_branch(branch: str, offset: int) -> str:
    """
    Get the branch that is offset positions away.
    Positive offset moves forward, negative moves backward.
    """
    current_index = get_branch_index(branch)
    new_index = (current_index + offset) % 12
    return TWELVE_BRANCHES[new_index]


def get_branch_distance(from_branch: str, to_branch: str) -> int:
    """
    Calculate the forward distance from one branch to another.
    Returns a value from 0 to 11.
    """
    from_idx = get_branch_index(from_branch)
    to_idx = get_branch_index(to_branch)
    return (to_idx - from_idx) % 12


def is_yang_stem(stem: str) -> bool:
    """Check if a stem is Yang (陽干)."""
    return stem in YANG_STEMS


def is_yin_stem(stem: str) -> bool:
    """Check if a stem is Yin (陰干)."""
    return stem in YIN_STEMS


def get_clash_branch(branch: str) -> str:
    """Get the branch that clashes (衝) with the given branch."""
    return SIX_CLASHES.get(branch, '')


def get_harmony_branch(branch: str) -> str:
    """Get the branch that harmonizes (合) with the given branch."""
    return SIX_HARMONIES.get(branch, '')


def get_element(branch: str) -> str:
    """Get the element of a branch."""
    return BRANCH_ELEMENT.get(branch, '')


def does_control(controller: str, controlled: str) -> bool:
    """
    Check if controller's element controls controlled's element.
    Both arguments should be branch characters.
    """
    ctrl_element = BRANCH_ELEMENT.get(controller, '')
    ctrd_element = BRANCH_ELEMENT.get(controlled, '')
    if not ctrl_element or not ctrd_element:
        return False
    return ELEMENT_CONTROLS.get(ctrl_element) == ctrd_element


def does_generate(generator: str, generated: str) -> bool:
    """
    Check if generator's element generates generated's element.
    Both arguments should be branch characters.
    """
    gen_element = BRANCH_ELEMENT.get(generator, '')
    gend_element = BRANCH_ELEMENT.get(generated, '')
    if not gen_element or not gend_element:
        return False
    return ELEMENT_GENERATES.get(gen_element) == gend_element
