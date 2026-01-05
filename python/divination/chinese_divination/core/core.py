from enum import Enum


class Element(Enum):
    """Five Elements (Wu Xing) enumeration"""
    WOOD = "木"
    FIRE = "火"
    EARTH = "土"
    METAL = "金"
    WATER = "水"


class Polarity(Enum):
    """Yin-Yang polarity"""
    YANG = "阳"
    YIN = "阴"


class Season(Enum):
    """Seasonal associations"""
    SPRING = "春"
    SUMMER = "夏"
    LATE_SUMMER = "长夏"
    AUTUMN = "秋"
    WINTER = "冬"


class Direction(Enum):
    """Directional associations"""
    EAST = "东"
    SOUTH = "南"
    CENTER = "中"
    WEST = "西"
    NORTH = "北"
    # Extended directions for Nine Palaces
    NORTHEAST = "东北"
    NORTHWEST = "西北"
    SOUTHEAST = "东南"
    SOUTHWEST = "西南"


# =============================================================================
# Qimen Dunjia (奇門遁甲) Enums
# =============================================================================

class DunType(Enum):
    """
    Dun method - determines rotation direction of the plates.

    陽遁 (Yang Dun): Winter Solstice to Summer Solstice - forward rotation
    陰遁 (Yin Dun): Summer Solstice to Winter Solstice - backward rotation
    """
    YANG = ("阳遁", 1)
    YIN = ("阴遁", -1)

    def __init__(self, chinese: str, rotation_direction: int):
        self.chinese = chinese
        self.rotation_direction = rotation_direction


class Yuan(Enum):
    """
    Three Yuan periods within each solar term.
    Each solar term is divided into three 5-day periods.

    The indices indicate which 甲 (Jia) day starts each Yuan:
    - Upper Yuan: 甲子(1), 甲午(31)
    - Middle Yuan: 甲寅(51), 甲申(21)
    - Lower Yuan: 甲辰(41), 甲戌(11)
    """
    UPPER = ("上元", [1, 31])
    MIDDLE = ("中元", [51, 21])
    LOWER = ("下元", [41, 11])

    def __init__(self, chinese: str, jia_indices: list):
        self.chinese = chinese
        self.jia_indices = jia_indices


class Trigram(Enum):
    """
    Eight Trigrams (八卦) in Later Heaven (後天) arrangement.

    Each trigram has:
    - Chinese character
    - English meaning
    - Palace number in the Nine Palaces
    - Direction
    - Associated element

    Later Heaven arrangement (文王後天八卦):
        巽(4)  離(9)  坤(2)
        震(3)  中(5)  兌(7)
        艮(8)  坎(1)  乾(6)
    """
    KAN = ("坎", "Water", 1, Direction.NORTH, Element.WATER)
    KUN = ("坤", "Earth", 2, Direction.SOUTHWEST, Element.EARTH)
    ZHEN = ("震", "Thunder", 3, Direction.EAST, Element.WOOD)
    XUN = ("巽", "Wind", 4, Direction.SOUTHEAST, Element.WOOD)
    QIAN = ("乾", "Heaven", 6, Direction.NORTHWEST, Element.METAL)
    DUI = ("兌", "Lake", 7, Direction.WEST, Element.METAL)
    GEN = ("艮", "Mountain", 8, Direction.NORTHEAST, Element.EARTH)
    LI = ("離", "Fire", 9, Direction.SOUTH, Element.FIRE)

    def __init__(self, chinese: str, meaning: str, palace_number: int,
                 direction: Direction, element: Element):
        self.chinese = chinese
        self.meaning = meaning
        self.palace_number = palace_number
        self.direction = direction
        self._element = element

    @property
    def element(self) -> Element:
        return self._element


class SolarTerm(Enum):
    """
    24 Solar Terms (二十四節氣) with Qimen Dunjia associations.

    Each term has:
    - Chinese name
    - Solar longitude in degrees
    - Dun type (Yang or Yin)
    - Base Ju number for Upper Yuan

    Solar longitude progresses through 360° with each term spanning 15°.
    Winter Solstice (冬至) is at 270° and marks the start of Yang Dun.
    Summer Solstice (夏至) is at 90° and marks the start of Yin Dun.
    """
    # Yang Dun period (Winter Solstice to Summer Solstice)
    DONG_ZHI = ("冬至", 270, DunType.YANG, 1)       # Winter Solstice
    XIAO_HAN = ("小寒", 285, DunType.YANG, 2)       # Minor Cold
    DA_HAN = ("大寒", 300, DunType.YANG, 3)         # Major Cold
    LI_CHUN = ("立春", 315, DunType.YANG, 8)        # Start of Spring
    YU_SHUI = ("雨水", 330, DunType.YANG, 9)        # Rain Water
    JING_ZHE = ("惊蛰", 345, DunType.YANG, 1)       # Awakening of Insects
    CHUN_FEN = ("春分", 0, DunType.YANG, 3)         # Spring Equinox
    QING_MING = ("清明", 15, DunType.YANG, 4)       # Clear and Bright
    GU_YU = ("谷雨", 30, DunType.YANG, 5)           # Grain Rain
    LI_XIA = ("立夏", 45, DunType.YANG, 4)          # Start of Summer
    XIAO_MAN = ("小满", 60, DunType.YANG, 5)        # Grain Buds
    MANG_ZHONG = ("芒种", 75, DunType.YANG, 6)      # Grain in Ear

    # Yin Dun period (Summer Solstice to Winter Solstice)
    XIA_ZHI = ("夏至", 90, DunType.YIN, 9)          # Summer Solstice
    XIAO_SHU = ("小暑", 105, DunType.YIN, 8)        # Minor Heat
    DA_SHU = ("大暑", 120, DunType.YIN, 7)          # Major Heat
    LI_QIU = ("立秋", 135, DunType.YIN, 2)          # Start of Autumn
    CHU_SHU = ("处暑", 150, DunType.YIN, 1)         # End of Heat
    BAI_LU = ("白露", 165, DunType.YIN, 9)          # White Dew
    QIU_FEN = ("秋分", 180, DunType.YIN, 7)         # Autumn Equinox
    HAN_LU = ("寒露", 195, DunType.YIN, 6)          # Cold Dew
    SHUANG_JIANG = ("霜降", 210, DunType.YIN, 5)    # Frost Descent
    LI_DONG = ("立冬", 225, DunType.YIN, 6)         # Start of Winter
    XIAO_XUE = ("小雪", 240, DunType.YIN, 5)        # Minor Snow
    DA_XUE = ("大雪", 255, DunType.YIN, 4)          # Major Snow

    def __init__(self, chinese: str, longitude: float,
                 dun_type: DunType, base_ju: int):
        self.chinese = chinese
        self.longitude = longitude
        self.dun_type = dun_type
        self.base_ju = base_ju  # Base Ju number for upper yuan


