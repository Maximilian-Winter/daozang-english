"""
Taiyi Shenshu Constants and Cycle Definitions

This module defines the fundamental constants, cycles, and enumerations
used in Taiyi Shenshu calculations.

Based on classical sources:
- 太乙金鏡式經: "周紀法三百六十，元法七十二，太乙小周法二十四"
- 太乙秘書: "命起武德，順行十六神，遇陰德大武，重留一算"

碼道長存 — The Way of Code endures
"""

from enum import Enum, IntEnum
from typing import List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Core Cycle Constants (基本週期常數)
# =============================================================================

class TaiyiConstants:
    """
    Fundamental constants for Taiyi Shenshu calculations.

    These values are derived from classical texts and form the mathematical
    foundation of all Taiyi computations.
    """

    # Primary Cycle Constants (主要週期常數)
    ZHOU_JI_FA = 360        # 周紀法 - Grand Cycle (years/months/days)
    YUAN_FA = 72            # 元法 - Era Law (72 years per era)
    TAIYI_XIAO_ZHOU = 24    # 太乙小周法 - Taiyi Small Cycle
    TIANMU_ZHOU_FA = 18     # 天目周法 - Tianmu (Celestial Eye) Cycle

    # Great/Small Wandering Cycles (大游/小游週期)
    DA_YOU_DA_ZHOU = 4320   # 大游大周法 - Great Wandering Grand Cycle
    DA_YOU_XIAO_ZHOU = 288  # 大游小周法 - Great Wandering Small Cycle
    DA_YOU_GONG_ZHOU = 36   # 大游宮周 - Great Wandering Palace Cycle
    XIAO_YOU_DA_ZHOU = 240  # 小游大周法 - Small Wandering Grand Cycle
    XIAO_YOU_XIAO_ZHOU = 24 # 小游小周法 - Small Wandering Small Cycle

    # Derived Constants
    JI_FA = 60              # 紀法 - Sexagenary Cycle
    PALACE_STAY = 3         # 三時一移 - 3 time units per palace
    NUM_OUTER_PALACES = 8   # 八宮 - 8 outer palaces (excluding center)
    NUM_SPIRITS = 16        # 十六神 - 16 spirits

    # Palace traversal sequence (Yang Dun - forward, skip palace 5)
    YANG_DUN_PALACE_SEQUENCE: List[int] = [1, 2, 3, 4, 6, 7, 8, 9]

    # Palace traversal sequence (Yin Dun - backward, skip palace 5)
    YIN_DUN_PALACE_SEQUENCE: List[int] = [9, 8, 7, 6, 4, 3, 2, 1]

    # Spirits requiring extra count (重留一算)
    ZHONG_LIU_SPIRITS = {"阴德", "大武"}


# =============================================================================
# Dun Type Enumeration (遁類型)
# =============================================================================

class TaiyiDunType(Enum):
    """
    Taiyi Dun Type - determines rotation direction and starting palace.

    陽遁 (Yang Dun): After Winter Solstice - starts from Palace 1, forward
    陰遁 (Yin Dun): After Summer Solstice - starts from Palace 9, backward

    The classical text states:
    "冬至氣應後，用陽局，夏至氣應後，用陰局"
    "皆以陽局所命之對沖，則陰局太乙所在也"
    """
    YANG = ("阳遁", 1, 1, TaiyiConstants.YANG_DUN_PALACE_SEQUENCE)
    YIN = ("阴遁", -1, 9, TaiyiConstants.YIN_DUN_PALACE_SEQUENCE)

    def __init__(self, chinese: str, direction: int,
                 start_palace: int, palace_sequence: List[int]):
        self.chinese = chinese
        self.direction = direction  # 1 for forward, -1 for backward
        self.start_palace = start_palace
        self.palace_sequence = palace_sequence

    @classmethod
    def from_solar_longitude(cls, longitude: float) -> 'TaiyiDunType':
        """
        Determine Dun type from solar longitude.

        Winter Solstice (冬至): ~270°
        Summer Solstice (夏至): ~90°

        Args:
            longitude: Sun's ecliptic longitude in degrees (0-360)

        Returns:
            YANG for winter→summer, YIN for summer→winter
        """
        # Yang Dun: From Winter Solstice (270°) to before Summer Solstice (90°)
        if longitude >= 270 or longitude < 90:
            return cls.YANG
        else:
            return cls.YIN


# =============================================================================
# Era System (紀元系統)
# =============================================================================

class TaiyiEra(Enum):
    """
    Six Eras (六紀) within the 60-year cycle.

    Each era is defined by specific Heavenly Stem patterns and determines
    the starting positions of Taiyi and Tianmu.

    From 太乙金鏡式經:
    "一紀：二甲仲辰。甲子、甲午。太乙在一宮，武德為天目"
    """
    ERA_1 = (1, "一紀", ["甲子", "甲午"], "二甲仲辰", 1, "武德")
    ERA_2 = (2, "二紀", ["己巳", "己亥"], "二己孟辰", 6, "地主")
    ERA_3 = (3, "三紀", ["甲辰", "甲戌"], "二甲季辰", 1, "太灵")
    ERA_4 = (4, "四紀", ["己卯", "己酉"], "二己仲辰", 6, "武德")
    ERA_5 = (5, "五紀", ["甲申", "甲寅"], "二甲孟辰", 1, "地主")
    ERA_6 = (6, "六紀", ["己丑", "己未"], "二己季辰", 6, "太灵")

    def __init__(self, number: int, chinese: str, stem_pairs: List[str],
                 pattern_name: str, taiyi_palace: int, tianmu_spirit: str):
        self.number = number
        self.chinese_name = chinese
        self.stem_pairs = stem_pairs
        self.pattern_name = pattern_name
        self.taiyi_palace = taiyi_palace
        self.tianmu_spirit = tianmu_spirit

    @classmethod
    def from_year_index(cls, year_in_cycle: int) -> 'TaiyiEra':
        """
        Determine the era from year position within 60-year cycle.

        Args:
            year_in_cycle: Year position (1-60) within sexagenary cycle

        Returns:
            The corresponding TaiyiEra
        """
        # Each era spans 10 years
        era_index = ((year_in_cycle - 1) // 10) % 6
        return list(cls)[era_index]


# =============================================================================
# Palace Type Classifications (宮位分類)
# =============================================================================

class PalaceType(Enum):
    """
    Palace classifications for strategic interpretation.

    From 太乙秘書:
    "八三四九為陽宮，二七六一為陰宮"
    "太乙在八、三、四宮者，為地內宮，助主人"
    "太乙在九、二、七、六宮者，為天外宮，助於客"
    """
    # Yin-Yang Classification
    YANG_PALACE = "阳宫"  # Palaces 3, 4, 8, 9
    YIN_PALACE = "阴宫"   # Palaces 1, 2, 6, 7
    CENTER = "中宫"       # Palace 5 (Taiyi skips this)

    # Strategic Classification
    INNER = "地内宫"      # Palaces 3, 4, 8 - favors defender (主)
    OUTER = "天外宫"      # Palaces 2, 6, 7, 9 - favors attacker (客)


@dataclass
class PalaceInfo:
    """Detailed information about a palace."""
    number: int
    trigram: str
    direction: str
    element: str
    yin_yang: PalaceType
    strategic: PalaceType

    @property
    def is_yang(self) -> bool:
        return self.yin_yang == PalaceType.YANG_PALACE

    @property
    def favors_host(self) -> bool:
        return self.strategic == PalaceType.INNER


# Palace definitions following the Luoshu arrangement
PALACE_DEFINITIONS = {
    1: PalaceInfo(1, "坎", "北", "水", PalaceType.YIN_PALACE, PalaceType.INNER),
    2: PalaceInfo(2, "坤", "西南", "土", PalaceType.YIN_PALACE, PalaceType.OUTER),
    3: PalaceInfo(3, "震", "東", "木", PalaceType.YANG_PALACE, PalaceType.INNER),
    4: PalaceInfo(4, "巽", "東南", "木", PalaceType.YANG_PALACE, PalaceType.INNER),
    5: PalaceInfo(5, "中", "中", "土", PalaceType.CENTER, PalaceType.INNER),
    6: PalaceInfo(6, "乾", "西北", "金", PalaceType.YIN_PALACE, PalaceType.OUTER),
    7: PalaceInfo(7, "兌", "西", "金", PalaceType.YIN_PALACE, PalaceType.OUTER),
    8: PalaceInfo(8, "艮", "東北", "土", PalaceType.YANG_PALACE, PalaceType.INNER),
    9: PalaceInfo(9, "離", "南", "火", PalaceType.YANG_PALACE, PalaceType.OUTER),
}


# =============================================================================
# Special Palace Positions (特殊宮位)
# =============================================================================

class SpecialPosition(Enum):
    """
    Special palace positions with particular significance.

    From 太乙秘書:
    "在一宮則為絕陽，在九宮則為絕陰"
    "在四六則為絕氣，在二八則為易氣"
    """
    JUE_YANG = (1, "绝阳", "Extreme Yang - self-defeating for yang forces")
    JUE_YIN = (9, "绝阴", "Extreme Yin - self-defeating for yin forces")
    JUE_QI_4 = (4, "绝气", "Qi Extinction")
    JUE_QI_6 = (6, "绝气", "Qi Extinction")
    YI_QI_2 = (2, "易气", "Qi Transformation")
    YI_QI_8 = (8, "易气", "Qi Transformation")

    def __init__(self, palace: int, chinese: str, meaning: str):
        self.palace = palace
        self.chinese_name = chinese
        self.meaning = meaning


def get_special_position(palace: int) -> Optional[SpecialPosition]:
    """Get the special position classification for a palace, if any."""
    for pos in SpecialPosition:
        if pos.palace == palace:
            return pos
    return None


# =============================================================================
# Calculation Helpers
# =============================================================================

def get_opposite_palace(palace: int) -> int:
    """
    Get the opposite palace (對沖) for a given palace.

    The opposition pairs are:
    1 ↔ 9, 2 ↔ 8, 3 ↔ 7, 4 ↔ 6
    Palace 5 has no opposite (it's the center).
    """
    opposites = {1: 9, 9: 1, 2: 8, 8: 2, 3: 7, 7: 3, 4: 6, 6: 4, 5: 5}
    return opposites.get(palace, palace)


def palace_to_sequence_index(palace: int, dun_type: TaiyiDunType) -> int:
    """Convert palace number to index in traversal sequence."""
    try:
        return dun_type.palace_sequence.index(palace)
    except ValueError:
        return 0  # Default to first position if not found


# End of constants module
