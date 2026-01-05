"""
Da Liu Ren Components (大六壬組件)

Core component classes for Liu Ren divination:
- TianJiang (天將): The 12 Heavenly Generals
- TwelveGenerals: Container class for all generals
- SiKe (四課): The Four Lessons structure
- SanChuan (三傳): The Three Transmissions structure

These components follow the patterns established in Qimen Dunjia
and Taiyi Shenshu implementations.
"""

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Dict, List, Optional, Tuple, Set, Any

from ..core.core import Element, Polarity, Direction


# =============================================================================
# Enumerations (枚舉)
# =============================================================================

class GeneralPosition(IntEnum):
    """
    Position enumeration for the 12 Heavenly Generals.
    Values correspond to their traditional ordering.
    """
    GUI_REN = 1      # 貴人
    TENG_SHE = 2     # 騰蛇
    ZHU_QUE = 3      # 朱雀
    LIU_HE = 4       # 六合
    GOU_CHEN = 5     # 勾陳
    QING_LONG = 6    # 青龍
    TIAN_HOU = 7     # 天后
    TAI_YIN = 8      # 太陰
    XUAN_WU = 9      # 玄武
    TAI_CHANG = 10   # 太常
    BAI_HU = 11      # 白虎
    TIAN_KONG = 12   # 天空


class GeneralNature(Enum):
    """Nature of Heavenly General - Auspicious or Inauspicious"""
    AUSPICIOUS = ("吉", "Auspicious")
    INAUSPICIOUS = ("凶", "Inauspicious")

    def __init__(self, chinese: str, english: str):
        self.chinese = chinese
        self.english = english


class KeRelation(Enum):
    """
    Relationship type in a Lesson (課).
    Used to determine which lesson becomes the transmission.
    """
    SHANG_KE_XIA = ("上克下", "Upper controls lower")
    XIA_ZEI_SHANG = ("下賊上", "Lower attacks upper")
    BI_YONG = ("比用", "Same element comparison")
    SHE_HAI = ("涉害", "Harm/damage relation")
    YAO_KE = ("遙克", "Distant control")
    NONE = ("無", "No relation")

    def __init__(self, chinese: str, english: str):
        self.chinese = chinese
        self.english = english


# =============================================================================
# TianJiang (天將) - Heavenly General
# =============================================================================

@dataclass
class TianJiang:
    """
    天將 - Heavenly General

    One of the 12 celestial generals used in Liu Ren divination.
    Each general has specific attributes and domains of influence.

    Attributes:
        position: The general's position number (1-12)
        chinese: Chinese name of the general
        pinyin: Romanized pronunciation
        base_branch: The earthly branch this general is associated with
        element: Five Element association
        polarity: Yin or Yang nature
        nature: Auspicious or Inauspicious
        domain: Areas of life this general governs
        meaning: Classical interpretation and significance
    """
    position: GeneralPosition
    chinese: str
    pinyin: str
    base_branch: str
    element: Element
    polarity: Polarity
    nature: GeneralNature
    domain: str
    meaning: str

    def __str__(self) -> str:
        return f"{self.chinese} ({self.pinyin})"

    def __repr__(self) -> str:
        return f"TianJiang({self.chinese}, {self.base_branch}, {self.nature.chinese})"

    @property
    def is_auspicious(self) -> bool:
        """Check if this general is auspicious."""
        return self.nature == GeneralNature.AUSPICIOUS

    @property
    def is_inauspicious(self) -> bool:
        """Check if this general is inauspicious."""
        return self.nature == GeneralNature.INAUSPICIOUS


# =============================================================================
# TwelveGenerals Container (十二天將容器)
# =============================================================================

class TwelveGenerals:
    """
    Container class for all 12 Heavenly Generals.

    Provides multiple lookup methods:
    - by_chinese: Lookup by Chinese name
    - by_position: Lookup by GeneralPosition enum
    - by_branch: Lookup by associated earthly branch
    - by_index: Lookup by 0-based index

    The sequence follows the traditional ordering starting from 貴人.
    """

    # Auspicious generals
    AUSPICIOUS_GENERALS: Set[str] = {'貴人', '六合', '青龍', '天后', '太陰', '太常'}

    # Inauspicious generals
    INAUSPICIOUS_GENERALS: Set[str] = {'騰蛇', '朱雀', '勾陳', '玄武', '白虎', '天空'}

    def __init__(self):
        self.generals: List[TianJiang] = self._create_generals()
        self._build_indices()

    def _create_generals(self) -> List[TianJiang]:
        """Create all 12 Heavenly Generals with their attributes."""
        return [
            TianJiang(
                position=GeneralPosition.GUI_REN,
                chinese='貴人',
                pinyin='guìrén',
                base_branch='丑',
                element=Element.EARTH,
                polarity=Polarity.YIN,
                nature=GeneralNature.AUSPICIOUS,
                domain='尊貴、貴人相助',
                meaning='天乙貴人，主尊貴、喜慶、貴人扶助。為諸神之首，最為尊貴。'
            ),
            TianJiang(
                position=GeneralPosition.TENG_SHE,
                chinese='騰蛇',
                pinyin='téngshé',
                base_branch='巳',
                element=Element.FIRE,
                polarity=Polarity.YIN,
                nature=GeneralNature.INAUSPICIOUS,
                domain='虛驚、怪異、夢寐',
                meaning='主虛驚怪異、火災口舌、憂疑不定。性情多變，主驚恐。'
            ),
            TianJiang(
                position=GeneralPosition.ZHU_QUE,
                chinese='朱雀',
                pinyin='zhūquè',
                base_branch='午',
                element=Element.FIRE,
                polarity=Polarity.YANG,
                nature=GeneralNature.INAUSPICIOUS,
                domain='口舌、文書、信息',
                meaning='主口舌是非、文書信件、消息傳遞。喜動多言，主是非。'
            ),
            TianJiang(
                position=GeneralPosition.LIU_HE,
                chinese='六合',
                pinyin='liùhé',
                base_branch='卯',
                element=Element.WOOD,
                polarity=Polarity.YIN,
                nature=GeneralNature.AUSPICIOUS,
                domain='婚姻、和合、交易',
                meaning='主婚姻、媒妁、交易和合。為和合之神，主成就。'
            ),
            TianJiang(
                position=GeneralPosition.GOU_CHEN,
                chinese='勾陳',
                pinyin='gōuchén',
                base_branch='辰',
                element=Element.EARTH,
                polarity=Polarity.YANG,
                nature=GeneralNature.INAUSPICIOUS,
                domain='田土、牢獄、爭訟',
                meaning='主田土、牢獄、訴訟。為爭鬥之神，主遲滯。'
            ),
            TianJiang(
                position=GeneralPosition.QING_LONG,
                chinese='青龍',
                pinyin='qīnglóng',
                base_branch='寅',
                element=Element.WOOD,
                polarity=Polarity.YANG,
                nature=GeneralNature.AUSPICIOUS,
                domain='喜慶、財帛、貴人',
                meaning='主喜慶、財帛、進益。為吉慶之神，主財喜。'
            ),
            TianJiang(
                position=GeneralPosition.TIAN_HOU,
                chinese='天后',
                pinyin='tiānhòu',
                base_branch='未',
                element=Element.EARTH,
                polarity=Polarity.YIN,
                nature=GeneralNature.AUSPICIOUS,
                domain='婦女、陰私、暗昧',
                meaning='主婦女、陰私、暗昧。為陰德之神，主女性。'
            ),
            TianJiang(
                position=GeneralPosition.TAI_YIN,
                chinese='太陰',
                pinyin='tàiyīn',
                base_branch='酉',
                element=Element.METAL,
                polarity=Polarity.YIN,
                nature=GeneralNature.AUSPICIOUS,
                domain='陰私、藏匿、女性',
                meaning='主陰私、藏匿、暗中相助。為陰護之神，主隱密。'
            ),
            TianJiang(
                position=GeneralPosition.XUAN_WU,
                chinese='玄武',
                pinyin='xuánwǔ',
                base_branch='子',
                element=Element.WATER,
                polarity=Polarity.YANG,
                nature=GeneralNature.INAUSPICIOUS,
                domain='盜賊、失亡、曖昧',
                meaning='主盜賊、失亡、曖昧。為陰邪之神，主欺騙。'
            ),
            TianJiang(
                position=GeneralPosition.TAI_CHANG,
                chinese='太常',
                pinyin='tàicháng',
                base_branch='戌',
                element=Element.EARTH,
                polarity=Polarity.YANG,
                nature=GeneralNature.AUSPICIOUS,
                domain='飲食、衣冠、禮儀',
                meaning='主飲食、宴會、衣裳。為禮樂之神，主文雅。'
            ),
            TianJiang(
                position=GeneralPosition.BAI_HU,
                chinese='白虎',
                pinyin='báihǔ',
                base_branch='申',
                element=Element.METAL,
                polarity=Polarity.YANG,
                nature=GeneralNature.INAUSPICIOUS,
                domain='喪亡、凶災、血光',
                meaning='主喪亡、凶災、血光。為兇煞之神，主殺伐。'
            ),
            TianJiang(
                position=GeneralPosition.TIAN_KONG,
                chinese='天空',
                pinyin='tiānkōng',
                base_branch='亥',
                element=Element.WATER,
                polarity=Polarity.YIN,
                nature=GeneralNature.INAUSPICIOUS,
                domain='空亡、欺詐、虛假',
                meaning='主欺詐、虛假、空亡。為虛空之神，主不實。'
            ),
        ]

    def _build_indices(self) -> None:
        """Build lookup indices for quick access."""
        self.by_chinese: Dict[str, TianJiang] = {
            g.chinese: g for g in self.generals
        }
        self.by_position: Dict[GeneralPosition, TianJiang] = {
            g.position: g for g in self.generals
        }
        self.by_branch: Dict[str, TianJiang] = {
            g.base_branch: g for g in self.generals
        }

    def get_by_chinese(self, name: str) -> Optional[TianJiang]:
        """Get a general by Chinese name."""
        return self.by_chinese.get(name)

    def get_by_position(self, position: GeneralPosition) -> Optional[TianJiang]:
        """Get a general by position enum."""
        return self.by_position.get(position)

    def get_by_branch(self, branch: str) -> Optional[TianJiang]:
        """Get a general by base branch."""
        return self.by_branch.get(branch)

    def get_by_index(self, index: int) -> Optional[TianJiang]:
        """Get a general by 0-based index."""
        if 0 <= index < 12:
            return self.generals[index]
        return None

    def get_auspicious(self) -> List[TianJiang]:
        """Get all auspicious generals."""
        return [g for g in self.generals if g.is_auspicious]

    def get_inauspicious(self) -> List[TianJiang]:
        """Get all inauspicious generals."""
        return [g for g in self.generals if g.is_inauspicious]

    def get_sequence_from(self, start_branch: str, forward: bool = True) -> List[TianJiang]:
        """
        Get the generals in sequence starting from a given branch position.

        Args:
            start_branch: The branch where 貴人 is placed
            forward: If True, sequence goes forward; if False, backward

        Returns:
            List of generals in rotated sequence
        """
        from .constants import BRANCH_TO_INDEX, TWELVE_BRANCHES

        start_idx = BRANCH_TO_INDEX.get(start_branch, 0)
        result = []

        for i in range(12):
            if forward:
                branch_idx = (start_idx + i) % 12
            else:
                branch_idx = (start_idx - i) % 12
            # Get the general for this position in sequence
            result.append(self.generals[i])

        return result

    def __getitem__(self, key) -> Optional[TianJiang]:
        """
        Flexible accessor supporting multiple key types.

        Usage:
            generals[0]           # by index
            generals['貴人']       # by Chinese name
            generals[GeneralPosition.GUI_REN]  # by enum
        """
        if isinstance(key, int):
            return self.get_by_index(key)
        elif isinstance(key, str):
            return self.get_by_chinese(key)
        elif isinstance(key, GeneralPosition):
            return self.get_by_position(key)
        return None

    def __iter__(self):
        return iter(self.generals)

    def __len__(self) -> int:
        return len(self.generals)


# =============================================================================
# Ke (課) - Individual Lesson
# =============================================================================

@dataclass
class Ke:
    """
    課 - A single Lesson in the Four Lessons system.

    Each lesson consists of two branches:
    - shang (上): The branch on top
    - xia (下): The branch on bottom

    The relationship between them determines克 patterns.
    """
    shang: str  # 上神 - Upper position
    xia: str    # 下神 - Lower position
    index: int  # Which lesson (1-4)

    # Optional: the general at this position
    shang_general: Optional[TianJiang] = None
    xia_general: Optional[TianJiang] = None

    def __str__(self) -> str:
        return f"第{self.index}課: {self.shang}/{self.xia}"

    def __repr__(self) -> str:
        return f"Ke({self.index}, {self.shang}/{self.xia})"

    def get_relation(self) -> KeRelation:
        """
        Determine the 克 relationship between upper and lower.

        Returns:
            KeRelation enum indicating the type of relationship
        """
        from .constants import does_control

        if does_control(self.shang, self.xia):
            return KeRelation.SHANG_KE_XIA
        elif does_control(self.xia, self.shang):
            return KeRelation.XIA_ZEI_SHANG
        else:
            return KeRelation.NONE

    @property
    def has_ke(self) -> bool:
        """Check if there is any 克 relationship in this lesson."""
        return self.get_relation() != KeRelation.NONE

    @property
    def is_zei(self) -> bool:
        """Check if this lesson has 賊 (lower attacks upper)."""
        return self.get_relation() == KeRelation.XIA_ZEI_SHANG


# =============================================================================
# SiKe (四課) - Four Lessons
# =============================================================================

@dataclass
class SiKe:
    """
    四課 - The Four Lessons

    The Four Lessons are derived from the day stem and branch
    overlaid on the Heaven Plate. They form the basis for
    deriving the Three Transmissions.

    Structure:
        第一課: Day stem's residence (寄宮) and its overlaid branch
        第二課: First lesson's upper branch and its overlaid branch
        第三課: Day branch and its overlaid branch
        第四課: Third lesson's upper branch and its overlaid branch

    Attributes:
        lessons: List of 4 Ke objects
        day_stem: The day's Heavenly Stem
        day_branch: The day's Earthly Branch
    """
    lessons: List[Ke]
    day_stem: str
    day_branch: str

    def __post_init__(self):
        if len(self.lessons) != 4:
            raise ValueError("SiKe must have exactly 4 lessons")

    def __str__(self) -> str:
        parts = [str(ke) for ke in self.lessons]
        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"SiKe({self.day_stem}{self.day_branch}, {[k.shang + '/' + k.xia for k in self.lessons]})"

    @property
    def lesson_1(self) -> Ke:
        """Get the first lesson (第一課)."""
        return self.lessons[0]

    @property
    def lesson_2(self) -> Ke:
        """Get the second lesson (第二課)."""
        return self.lessons[1]

    @property
    def lesson_3(self) -> Ke:
        """Get the third lesson (第三課)."""
        return self.lessons[2]

    @property
    def lesson_4(self) -> Ke:
        """Get the fourth lesson (第四課)."""
        return self.lessons[3]

    def get_ke_lessons(self) -> List[Ke]:
        """Get all lessons that have 克 relationships."""
        return [ke for ke in self.lessons if ke.has_ke]

    def get_zei_lessons(self) -> List[Ke]:
        """Get all lessons with 賊 (lower attacks upper)."""
        return [ke for ke in self.lessons if ke.is_zei]

    def get_shang_ke_lessons(self) -> List[Ke]:
        """Get all lessons with 上克下 (upper controls lower)."""
        return [
            ke for ke in self.lessons
            if ke.get_relation() == KeRelation.SHANG_KE_XIA
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the Four Lessons."""
        return {
            'day_stem': self.day_stem,
            'day_branch': self.day_branch,
            'lessons': [
                {
                    'index': ke.index,
                    'shang': ke.shang,
                    'xia': ke.xia,
                    'relation': ke.get_relation().chinese
                }
                for ke in self.lessons
            ],
            'ke_count': len(self.get_ke_lessons()),
            'zei_count': len(self.get_zei_lessons()),
        }


# =============================================================================
# SanChuan (三傳) - Three Transmissions
# =============================================================================

@dataclass
class SanChuan:
    """
    三傳 - The Three Transmissions

    The Three Transmissions are derived from the Four Lessons
    and represent the flow of events: beginning, development, and outcome.

    Structure:
        初傳 (chu_chuan): Initial transmission - represents the beginning
        中傳 (zhong_chuan): Middle transmission - represents development
        末傳 (mo_chuan): Final transmission - represents the outcome

    Attributes:
        chu_chuan: Initial transmission branch
        zhong_chuan: Middle transmission branch
        mo_chuan: Final transmission branch
        derivation_method: Which pattern was used to derive (e.g., 賊克, 比用)
        source_ke: Which lesson(s) the transmissions came from
    """
    chu_chuan: str          # 初傳 - Initial
    zhong_chuan: str        # 中傳 - Middle
    mo_chuan: str           # 末傳 - Final
    derivation_method: str  # How it was derived
    source_ke: Optional[int] = None  # Which lesson it came from (1-4)

    # Optional: associated generals
    chu_general: Optional[TianJiang] = None
    zhong_general: Optional[TianJiang] = None
    mo_general: Optional[TianJiang] = None

    def __str__(self) -> str:
        return f"三傳: {self.chu_chuan} → {self.zhong_chuan} → {self.mo_chuan}"

    def __repr__(self) -> str:
        return f"SanChuan({self.chu_chuan}, {self.zhong_chuan}, {self.mo_chuan})"

    @property
    def initial(self) -> str:
        """Alias for chu_chuan (初傳)."""
        return self.chu_chuan

    @property
    def middle(self) -> str:
        """Alias for zhong_chuan (中傳)."""
        return self.zhong_chuan

    @property
    def final(self) -> str:
        """Alias for mo_chuan (末傳)."""
        return self.mo_chuan

    def get_all_branches(self) -> List[str]:
        """Get all three transmission branches as a list."""
        return [self.chu_chuan, self.zhong_chuan, self.mo_chuan]

    def has_repeated_branch(self) -> bool:
        """Check if any branch appears more than once."""
        branches = self.get_all_branches()
        return len(branches) != len(set(branches))

    def get_elements(self) -> List[str]:
        """Get the elements of all three transmissions."""
        from .constants import BRANCH_ELEMENT
        return [
            BRANCH_ELEMENT.get(self.chu_chuan, ''),
            BRANCH_ELEMENT.get(self.zhong_chuan, ''),
            BRANCH_ELEMENT.get(self.mo_chuan, ''),
        ]

    def is_all_same_element(self) -> bool:
        """Check if all three transmissions have the same element."""
        elements = self.get_elements()
        return len(set(elements)) == 1

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the Three Transmissions."""
        from .constants import BRANCH_ELEMENT
        return {
            'chu_chuan': {
                'branch': self.chu_chuan,
                'element': BRANCH_ELEMENT.get(self.chu_chuan, ''),
                'general': self.chu_general.chinese if self.chu_general else None,
            },
            'zhong_chuan': {
                'branch': self.zhong_chuan,
                'element': BRANCH_ELEMENT.get(self.zhong_chuan, ''),
                'general': self.zhong_general.chinese if self.zhong_general else None,
            },
            'mo_chuan': {
                'branch': self.mo_chuan,
                'element': BRANCH_ELEMENT.get(self.mo_chuan, ''),
                'general': self.mo_general.chinese if self.mo_general else None,
            },
            'derivation_method': self.derivation_method,
            'has_repetition': self.has_repeated_branch(),
        }


# =============================================================================
# Singleton Instance (單例實例)
# =============================================================================

# Global instance of TwelveGenerals for easy access
_twelve_generals: Optional[TwelveGenerals] = None


def get_twelve_generals() -> TwelveGenerals:
    """
    Get the singleton instance of TwelveGenerals.

    Returns:
        TwelveGenerals instance
    """
    global _twelve_generals
    if _twelve_generals is None:
        _twelve_generals = TwelveGenerals()
    return _twelve_generals
