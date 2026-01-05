"""
Qimen Dunjia Constants and Mappings

This module contains all the classical formulas, sequences, and mappings
used in Qimen Dunjia calculations.

Sources:
    - 黄帝太一八门逆顺生死诀
    - 黄帝太一八门入式诀
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.core import Yuan, SolarTerm, DunType

# =============================================================================
# Six Yi (六儀) and Three Wonders (三奇)
# =============================================================================

# Six Yi - the six stems under which Jia hides
SIX_YI = ['戊', '己', '庚', '辛', '壬', '癸']

# Three Wonders - the three auspicious stems
THREE_WONDERS = ['乙', '丙', '丁']

# Complete Earth Plate sequence (六儀三奇排列)
# This is the order in which stems are placed on the Earth Plate
EARTH_PLATE_SEQUENCE = ['戊', '己', '庚', '辛', '壬', '癸', '丁', '丙', '乙']

# =============================================================================
# Jia Concealment (甲遁)
# =============================================================================

# Jia hides under the Six Yi stems
# Each 甲 of the six Jia (六甲) hides under a specific Yi stem
JIA_CONCEALMENT = {
    '戊': '甲子',  # Jia-Zi hides under Wu (甲子旬)
    '己': '甲戌',  # Jia-Xu hides under Ji (甲戌旬)
    '庚': '甲申',  # Jia-Shen hides under Geng (甲申旬)
    '辛': '甲午',  # Jia-Wu hides under Xin (甲午旬)
    '壬': '甲辰',  # Jia-Chen hides under Ren (甲辰旬)
    '癸': '甲寅',  # Jia-Yin hides under Gui (甲寅旬)
}

# Reverse lookup: which stem does each Jia hide under
JIA_TO_YI = {
    '甲子': '戊',
    '甲戌': '己',
    '甲申': '庚',
    '甲午': '辛',
    '甲辰': '壬',
    '甲寅': '癸',
}

# =============================================================================
# Palace Traversal Sequences
# =============================================================================

# Yang Dun sequence: forward traversal through palaces (excludes center 5)
# Order: 坎(1) → 艮(8) → 震(3) → 巽(4) → 離(9) → 坤(2) → 兌(7) → 乾(6)
YANG_DUN_SEQUENCE = [1, 8, 3, 4, 9, 2, 7, 6]

# Yin Dun sequence: backward traversal through palaces
# Order: 坎(1) → 乾(6) → 兌(7) → 坤(2) → 離(9) → 巽(4) → 震(3) → 艮(8)
YIN_DUN_SEQUENCE = [1, 6, 7, 2, 9, 4, 3, 8]

# =============================================================================
# Solar Term to Ju Mapping (節氣定局)
# =============================================================================

# From 黄帝太一八门逆顺生死诀:
# 一宮冬至，三宮春分，六宮立冬，八宮立春
# 九宮夏至，七宮秋分，四宮立夏，二宮立秋

SOLAR_TERM_JU_MAPPING = {
    # Yang Dun period
    SolarTerm.DONG_ZHI: {Yuan.UPPER: 1, Yuan.MIDDLE: 7, Yuan.LOWER: 4},     # 冬至
    SolarTerm.XIAO_HAN: {Yuan.UPPER: 2, Yuan.MIDDLE: 8, Yuan.LOWER: 5},     # 小寒
    SolarTerm.DA_HAN: {Yuan.UPPER: 3, Yuan.MIDDLE: 9, Yuan.LOWER: 6},       # 大寒
    SolarTerm.LI_CHUN: {Yuan.UPPER: 8, Yuan.MIDDLE: 5, Yuan.LOWER: 2},      # 立春
    SolarTerm.YU_SHUI: {Yuan.UPPER: 9, Yuan.MIDDLE: 6, Yuan.LOWER: 3},      # 雨水
    SolarTerm.JING_ZHE: {Yuan.UPPER: 1, Yuan.MIDDLE: 7, Yuan.LOWER: 4},     # 惊蛰
    SolarTerm.CHUN_FEN: {Yuan.UPPER: 3, Yuan.MIDDLE: 9, Yuan.LOWER: 6},     # 春分
    SolarTerm.QING_MING: {Yuan.UPPER: 4, Yuan.MIDDLE: 1, Yuan.LOWER: 7},    # 清明
    SolarTerm.GU_YU: {Yuan.UPPER: 5, Yuan.MIDDLE: 2, Yuan.LOWER: 8},        # 谷雨
    SolarTerm.LI_XIA: {Yuan.UPPER: 4, Yuan.MIDDLE: 1, Yuan.LOWER: 7},       # 立夏
    SolarTerm.XIAO_MAN: {Yuan.UPPER: 5, Yuan.MIDDLE: 2, Yuan.LOWER: 8},     # 小满
    SolarTerm.MANG_ZHONG: {Yuan.UPPER: 6, Yuan.MIDDLE: 3, Yuan.LOWER: 9},   # 芒种

    # Yin Dun period
    SolarTerm.XIA_ZHI: {Yuan.UPPER: 9, Yuan.MIDDLE: 3, Yuan.LOWER: 6},      # 夏至
    SolarTerm.XIAO_SHU: {Yuan.UPPER: 8, Yuan.MIDDLE: 2, Yuan.LOWER: 5},     # 小暑
    SolarTerm.DA_SHU: {Yuan.UPPER: 7, Yuan.MIDDLE: 1, Yuan.LOWER: 4},       # 大暑
    SolarTerm.LI_QIU: {Yuan.UPPER: 2, Yuan.MIDDLE: 5, Yuan.LOWER: 8},       # 立秋
    SolarTerm.CHU_SHU: {Yuan.UPPER: 1, Yuan.MIDDLE: 4, Yuan.LOWER: 7},      # 处暑
    SolarTerm.BAI_LU: {Yuan.UPPER: 9, Yuan.MIDDLE: 3, Yuan.LOWER: 6},       # 白露
    SolarTerm.QIU_FEN: {Yuan.UPPER: 7, Yuan.MIDDLE: 1, Yuan.LOWER: 4},      # 秋分
    SolarTerm.HAN_LU: {Yuan.UPPER: 6, Yuan.MIDDLE: 9, Yuan.LOWER: 3},       # 寒露
    SolarTerm.SHUANG_JIANG: {Yuan.UPPER: 5, Yuan.MIDDLE: 8, Yuan.LOWER: 2}, # 霜降
    SolarTerm.LI_DONG: {Yuan.UPPER: 6, Yuan.MIDDLE: 9, Yuan.LOWER: 3},      # 立冬
    SolarTerm.XIAO_XUE: {Yuan.UPPER: 5, Yuan.MIDDLE: 8, Yuan.LOWER: 2},     # 小雪
    SolarTerm.DA_XUE: {Yuan.UPPER: 4, Yuan.MIDDLE: 7, Yuan.LOWER: 1},       # 大雪
}

# =============================================================================
# Day Cycle to Yuan Mapping
# =============================================================================

# The sexagenary cycle position that starts each Yuan
# Upper Yuan: Days 1-5 after 甲子(1) or 甲午(31)
# Middle Yuan: Days 1-5 after 甲寅(51) or 甲申(21)
# Lower Yuan: Days 1-5 after 甲辰(41) or 甲戌(11)

DAY_CYCLE_TO_YUAN = {
    1: Yuan.UPPER,    # 甲子
    31: Yuan.UPPER,   # 甲午
    51: Yuan.MIDDLE,  # 甲寅
    21: Yuan.MIDDLE,  # 甲申
    41: Yuan.LOWER,   # 甲辰
    11: Yuan.LOWER,   # 甲戌
}

# All Jia day indices in the 60-day cycle
JIA_DAY_INDICES = [1, 11, 21, 31, 41, 51]

# =============================================================================
# Classical Formulas (古訣)
# =============================================================================

# From 黄帝太一八门逆顺生死诀:
# 陽遁: 甲子一休，甲戌三傷，甲申六開，甲午八生，甲辰一休，甲寅三寅
# 陰遁: 甲子九景，甲戌七驚，甲申四杜，甲午二死，甲辰九景，甲寅七驚

YANG_DUN_JIA_GATE = {
    '甲子': (1, '休'),  # Palace 1, Xiu Gate
    '甲戌': (3, '伤'),  # Palace 3, Shang Gate
    '甲申': (6, '开'),  # Palace 6, Kai Gate
    '甲午': (8, '生'),  # Palace 8, Sheng Gate
    '甲辰': (4, '杜'),  # Palace 4, Du Gate
    '甲寅': (2, '死'),  # Palace 2, Si Gate
}

YIN_DUN_JIA_GATE = {
    '甲子': (9, '景'),  # Palace 9, Jing Gate
    '甲戌': (7, '惊'),  # Palace 7, Jing Gate
    '甲申': (4, '杜'),  # Palace 4, Du Gate
    '甲午': (2, '死'),  # Palace 2, Si Gate
    '甲辰': (6, '开'),  # Palace 6, Kai Gate
    '甲寅': (8, '生'),  # Palace 8, Sheng Gate
}

# =============================================================================
# Nine Palace Grid Positions
# =============================================================================

# Later Heaven arrangement as (row, column) positions
# Grid layout:
#   巽(4)  離(9)  坤(2)
#   震(3)  中(5)  兌(7)
#   艮(8)  坎(1)  乾(6)

PALACE_GRID_POSITIONS = {
    1: (2, 1),  # 坎 - Bottom center
    2: (0, 2),  # 坤 - Top right
    3: (1, 0),  # 震 - Middle left
    4: (0, 0),  # 巽 - Top left
    5: (1, 1),  # 中 - Center
    6: (2, 2),  # 乾 - Bottom right
    7: (1, 2),  # 兌 - Middle right
    8: (2, 0),  # 艮 - Bottom left
    9: (0, 1),  # 離 - Top center
}

# Reverse lookup: grid position to palace number
GRID_TO_PALACE = {v: k for k, v in PALACE_GRID_POSITIONS.items()}

# =============================================================================
# Star-Gate-Spirit Base Positions
# =============================================================================

# Original palace positions for each element in their base state
STAR_BASE_PALACES = {
    '天蓬': 1,  # Tianpeng at Palace 1 (Kan)
    '天芮': 2,  # Tianrui at Palace 2 (Kun)
    '天冲': 3,  # Tianchong at Palace 3 (Zhen)
    '天辅': 4,  # Tianfu at Palace 4 (Xun)
    '天禽': 5,  # Tianqin at Palace 5 (Center)
    '天心': 6,  # Tianxin at Palace 6 (Qian)
    '天柱': 7,  # Tianzhu at Palace 7 (Dui)
    '天任': 8,  # Tianren at Palace 8 (Gen)
    '天英': 9,  # Tianying at Palace 9 (Li)
}

GATE_BASE_PALACES = {
    '休门': 1,  # Xiu Gate at Palace 1 (Kan)
    '死门': 2,  # Si Gate at Palace 2 (Kun)
    '伤门': 3,  # Shang Gate at Palace 3 (Zhen)
    '杜门': 4,  # Du Gate at Palace 4 (Xun)
    # Palace 5 (Center) has no gate
    '开门': 6,  # Kai Gate at Palace 6 (Qian)
    '惊门': 7,  # Jing Gate at Palace 7 (Dui)
    '生门': 8,  # Sheng Gate at Palace 8 (Gen)
    '景门': 9,  # Jing Gate at Palace 9 (Li)
}

# Spirit sequence starting from Zhi Fu (值符)
SPIRIT_SEQUENCE = ['值符', '螣蛇', '太阴', '六合', '白虎', '玄武', '九地', '九天']

# =============================================================================
# Helper Functions
# =============================================================================

def get_sequence_for_dun(dun_type: DunType) -> list:
    """Get the appropriate palace traversal sequence for the dun type."""
    if dun_type == DunType.YANG:
        return YANG_DUN_SEQUENCE
    return YIN_DUN_SEQUENCE


def get_jia_for_day_index(day_index: int) -> str:
    """
    Get the Jia that governs a specific day in the sexagenary cycle.

    Args:
        day_index: 1-60 position in the sexagenary cycle

    Returns:
        The governing Jia (e.g., '甲子', '甲戌', etc.)
    """
    # Normalize to 1-60 range
    day_index = ((day_index - 1) % 60) + 1

    # Find which Jia period we're in
    if day_index <= 10:
        return '甲子'
    elif day_index <= 20:
        return '甲戌'
    elif day_index <= 30:
        return '甲申'
    elif day_index <= 40:
        return '甲午'
    elif day_index <= 50:
        return '甲辰'
    else:
        return '甲寅'


def get_yuan_for_day_index(day_index: int) -> Yuan:
    """
    Determine the Yuan (Upper/Middle/Lower) for a day in the sexagenary cycle.

    Each Jia period (10 days) is split into:
    - Days 1-5: One Yuan
    - Days 6-10: Same or different Yuan depending on the period

    Args:
        day_index: 1-60 position in the sexagenary cycle

    Returns:
        The Yuan for this day
    """
    # Normalize to 1-60 range
    day_index = ((day_index - 1) % 60) + 1

    # Find the nearest Jia day
    for jia_idx in sorted(JIA_DAY_INDICES, reverse=True):
        if day_index >= jia_idx:
            return DAY_CYCLE_TO_YUAN.get(jia_idx, Yuan.UPPER)

    # Wrap around to 甲寅 (51) from previous cycle
    return Yuan.MIDDLE
