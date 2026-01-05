"""
Taiyi Shenshu Sixteen Spirits (十六神)

The Sixteen Spirits are the core celestial entities in Taiyi Shenshu.
They correspond to the 12 Earthly Branches plus 4 Corner Trigrams (艮巽坤乾).

From 太乙金鏡式經:
"何謂十六神？以天有十二次配地十二辰，以天有四時配地有四維，通之十六也。"

The spirits are traversed starting from 武德 (Wude), with special "重留" (extra count)
rules at 陰德 (Yinde) and 大武 (Dawu).

碼道長存 — The Way of Code endures
"""

from enum import IntEnum
from typing import Optional, List, Dict, Set
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.core import Element, Polarity


# =============================================================================
# Spirit Position Enumeration
# =============================================================================

class SpiritPosition(IntEnum):
    """
    Positions of the Sixteen Spirits in their natural order.

    The order follows: 12 Earthly Branches + 4 Corner Trigrams
    子(1) → 丑(2) → 艮(3) → 寅(4) → ... → 乾(15) → 亥(16)

    Note: Traversal for Tianmu calculation starts from 武德(申=12) not 地主(子=1).
    """
    ZI = 1      # 子 - 地主
    CHOU = 2    # 丑 - 阳德
    GEN = 3     # 艮 - 和德
    YIN = 4     # 寅 - 吕申
    MAO = 5     # 卯 - 高丛
    CHEN = 6    # 辰 - 太阳
    XUN = 7     # 巽 - 太灵
    SI = 8      # 巳 - 太神
    WU = 9      # 午 - 大威
    WEI = 10    # 未 - 天道
    KUN = 11    # 坤 - 大武 [重留]
    SHEN = 12   # 申 - 武德 [起點]
    YOU = 13    # 酉 - 太簇
    XU = 14     # 戌 - 阴主
    QIAN = 15   # 乾 - 阴德 [重留]
    HAI = 16    # 亥 - 大义


# =============================================================================
# Individual Spirit Class
# =============================================================================

@dataclass
class TaiyiSpirit:
    """
    Represents a single Taiyi Spirit with all its attributes.

    Each of the Sixteen Spirits has:
    - Position in the sequence
    - Chinese name and meaning
    - Associated element
    - Month/season association
    - Whether it requires extra count (重留)

    From 太乙金鏡式經:
    "子神曰地主。建子之月，陽氣初發，萬物陰生，故曰地主也"
    """
    position: SpiritPosition
    branch: str          # Earthly Branch or Trigram (子, 丑, 艮, etc.)
    chinese: str         # Spirit name (地主, 阳德, etc.)
    pinyin: str          # Romanization
    element: Element     # Five Element association
    polarity: Polarity   # Yin or Yang
    month_desc: str      # Month/season description
    meaning: str         # Classical meaning
    is_zhong_liu: bool   # True if requires extra count (重留)

    def __str__(self) -> str:
        return f"{self.chinese} ({self.pinyin})"

    def __repr__(self) -> str:
        zl = " [重留]" if self.is_zhong_liu else ""
        return f"TaiyiSpirit({self.branch}={self.chinese}{zl})"

    @property
    def is_traversal_start(self) -> bool:
        """武德 is the starting point for Tianmu traversal."""
        return self.chinese == "武德"

    @property
    def traversal_index(self) -> int:
        """
        Get the index in the traversal sequence (starting from 武德=0).

        Traversal order: 武德→太簇→阴主→阴德→大义→地主→...→大武→武德
        """
        # Convert natural position to traversal position
        # Natural: 子(1)...申(12)...亥(16)
        # Traversal: 武德(申=12) is index 0
        natural_pos = self.position.value
        traversal_pos = (natural_pos - SpiritPosition.SHEN.value) % 16
        return traversal_pos


# =============================================================================
# Sixteen Spirits Container Class
# =============================================================================

class SixteenSpirits:
    """
    Container class for all Sixteen Spirits.

    Provides lookup methods by Chinese name, branch, position, or traversal index.

    Usage:
        spirits = SixteenSpirits()
        wude = spirits.get_by_chinese("武德")
        dizhu = spirits.get_by_branch("子")
        zhong_liu = spirits.get_zhong_liu_spirits()
    """

    # Spirits requiring extra count during traversal
    ZHONG_LIU_NAMES: Set[str] = {"阴德", "大武"}

    def __init__(self):
        self.spirits = self._create_spirits()
        self._build_indices()

    def _create_spirits(self) -> List[TaiyiSpirit]:
        """Create all sixteen spirits with their classical attributes."""
        return [
            # =================================================================
            # Northern Quadrant (Winter - Water)
            # =================================================================
            TaiyiSpirit(
                position=SpiritPosition.ZI,
                branch="子", chinese="地主", pinyin="dìzhǔ",
                element=Element.WATER, polarity=Polarity.YANG,
                month_desc="建子之月",
                meaning="阳气初发，万物阴生，故曰地主也",
                is_zhong_liu=False
            ),
            TaiyiSpirit(
                position=SpiritPosition.CHOU,
                branch="丑", chinese="阳德", pinyin="yángdé",
                element=Element.EARTH, polarity=Polarity.YIN,
                month_desc="建丑之月",
                meaning="二阳用事，布育万物，故曰阳德也",
                is_zhong_liu=False
            ),

            # =================================================================
            # Northeastern Corner (艮 - Earth/Mountain)
            # =================================================================
            TaiyiSpirit(
                position=SpiritPosition.GEN,
                branch="艮", chinese="和德", pinyin="hédé",
                element=Element.EARTH, polarity=Polarity.YANG,
                month_desc="冬春将交",
                meaning="阴阳气合，群物方生，故曰和德也",
                is_zhong_liu=False
            ),

            # =================================================================
            # Eastern Quadrant (Spring - Wood)
            # =================================================================
            TaiyiSpirit(
                position=SpiritPosition.YIN,
                branch="寅", chinese="吕申", pinyin="lǚshēn",
                element=Element.WOOD, polarity=Polarity.YANG,
                month_desc="建寅之月",
                meaning="阳气大申，草木甲拆，故曰吕申也",
                is_zhong_liu=False
            ),
            TaiyiSpirit(
                position=SpiritPosition.MAO,
                branch="卯", chinese="高丛", pinyin="gāocóng",
                element=Element.WOOD, polarity=Polarity.YIN,
                month_desc="建卯之月",
                meaning="万物皆出，自地丛生，故曰高丛也",
                is_zhong_liu=False
            ),
            TaiyiSpirit(
                position=SpiritPosition.CHEN,
                branch="辰", chinese="太阳", pinyin="tàiyáng",
                element=Element.EARTH, polarity=Polarity.YANG,
                month_desc="建辰之月",
                meaning="雷出震势，阳气大盛，故曰太阳也",
                is_zhong_liu=False
            ),

            # =================================================================
            # Southeastern Corner (巽 - Wood/Wind)
            # =================================================================
            TaiyiSpirit(
                position=SpiritPosition.XUN,
                branch="巽", chinese="太灵", pinyin="tàilíng",
                element=Element.WOOD, polarity=Polarity.YIN,
                month_desc="春夏将交",
                meaning="盛暑方至，阳气炎酷，故曰太灵",
                is_zhong_liu=False
            ),

            # =================================================================
            # Southern Quadrant (Summer - Fire)
            # =================================================================
            TaiyiSpirit(
                position=SpiritPosition.SI,
                branch="巳", chinese="太神", pinyin="tàishén",
                element=Element.FIRE, polarity=Polarity.YIN,
                month_desc="建巳之月",
                meaning="少阴用事，阴阳不测，故曰太神",
                is_zhong_liu=False
            ),
            TaiyiSpirit(
                position=SpiritPosition.WU,
                branch="午", chinese="大威", pinyin="dàwēi",
                element=Element.FIRE, polarity=Polarity.YANG,
                month_desc="建午之月",
                meaning="阳附阴生，刑暴始行，故曰大威也",
                is_zhong_liu=False
            ),
            TaiyiSpirit(
                position=SpiritPosition.WEI,
                branch="未", chinese="天道", pinyin="tiāndào",
                element=Element.EARTH, polarity=Polarity.YIN,
                month_desc="建未之月",
                meaning="火能生土，土王于未，故曰天道也",
                is_zhong_liu=False
            ),

            # =================================================================
            # Southwestern Corner (坤 - Earth) [重留]
            # =================================================================
            TaiyiSpirit(
                position=SpiritPosition.KUN,
                branch="坤", chinese="大武", pinyin="dàwǔ",
                element=Element.EARTH, polarity=Polarity.YIN,
                month_desc="夏秋将交",
                meaning="阴气施令，杀伤万物，故曰大武",
                is_zhong_liu=True  # 重留一算
            ),

            # =================================================================
            # Western Quadrant (Autumn - Metal) [武德 = 起點]
            # =================================================================
            TaiyiSpirit(
                position=SpiritPosition.SHEN,
                branch="申", chinese="武德", pinyin="wǔdé",
                element=Element.METAL, polarity=Polarity.YANG,
                month_desc="建申之月",
                meaning="万物欲死，葬麦将生，故曰武德也",
                is_zhong_liu=False
            ),
            TaiyiSpirit(
                position=SpiritPosition.YOU,
                branch="酉", chinese="太簇", pinyin="tàicù",
                element=Element.METAL, polarity=Polarity.YIN,
                month_desc="建酉之月",
                meaning="万物皆成，有大品蔟，故曰太簇也",
                is_zhong_liu=False
            ),
            TaiyiSpirit(
                position=SpiritPosition.XU,
                branch="戌", chinese="阴主", pinyin="yīnzhǔ",
                element=Element.EARTH, polarity=Polarity.YANG,
                month_desc="建戌之月",
                meaning="阳气不长，阴气用事，故曰阴主也",
                is_zhong_liu=False
            ),

            # =================================================================
            # Northwestern Corner (乾 - Metal/Heaven) [重留]
            # =================================================================
            TaiyiSpirit(
                position=SpiritPosition.QIAN,
                branch="乾", chinese="阴德", pinyin="yīndé",
                element=Element.METAL, polarity=Polarity.YANG,
                month_desc="秋冬将交",
                meaning="阴前生阳，大有其情，故曰阴德也",
                is_zhong_liu=True  # 重留一算
            ),

            # =================================================================
            # Northern Quadrant (Late Autumn/Early Winter - Water)
            # =================================================================
            TaiyiSpirit(
                position=SpiritPosition.HAI,
                branch="亥", chinese="大义", pinyin="dàyì",
                element=Element.WATER, polarity=Polarity.YIN,
                month_desc="建亥之月",
                meaning="万物怀垢，群阳欲尽，故曰大义",
                is_zhong_liu=False
            ),
        ]

    def _build_indices(self) -> None:
        """Build lookup indices for efficient access."""
        self.by_chinese: Dict[str, TaiyiSpirit] = {
            s.chinese: s for s in self.spirits
        }
        self.by_branch: Dict[str, TaiyiSpirit] = {
            s.branch: s for s in self.spirits
        }
        self.by_position: Dict[SpiritPosition, TaiyiSpirit] = {
            s.position: s for s in self.spirits
        }
        # Traversal order index (starting from 武德)
        self._traversal_order = self._build_traversal_order()

    def _build_traversal_order(self) -> List[TaiyiSpirit]:
        """
        Build the traversal order starting from 武德.

        Order: 武德→太簇→阴主→阴德→大义→地主→阳德→和德→
               吕申→高丛→太阳→太灵→太神→大威→天道→大武
        """
        start_pos = SpiritPosition.SHEN.value  # 武德 = 12
        ordered = []
        for i in range(16):
            pos = ((start_pos - 1 + i) % 16) + 1
            spirit = self.by_position[SpiritPosition(pos)]
            ordered.append(spirit)
        return ordered

    # =========================================================================
    # Lookup Methods
    # =========================================================================

    def get_by_chinese(self, name: str) -> Optional[TaiyiSpirit]:
        """Get spirit by Chinese name (e.g., '武德', '地主')."""
        return self.by_chinese.get(name)

    def get_by_branch(self, branch: str) -> Optional[TaiyiSpirit]:
        """Get spirit by Earthly Branch or Trigram (e.g., '子', '申', '艮')."""
        return self.by_branch.get(branch)

    def get_by_position(self, position: SpiritPosition) -> Optional[TaiyiSpirit]:
        """Get spirit by position enum."""
        return self.by_position.get(position)

    def get_by_traversal_index(self, index: int) -> TaiyiSpirit:
        """
        Get spirit by traversal index (0-15, starting from 武德).

        Args:
            index: 0 = 武德, 1 = 太簇, ..., 15 = 大武
        """
        return self._traversal_order[index % 16]

    def get_zhong_liu_spirits(self) -> List[TaiyiSpirit]:
        """Get spirits that require extra count (重留): 阴德 and 大武."""
        return [s for s in self.spirits if s.is_zhong_liu]

    def get_start_spirit(self) -> TaiyiSpirit:
        """Get the traversal start spirit (武德)."""
        return self.by_chinese["武德"]

    def get_spirits_by_element(self, element: Element) -> List[TaiyiSpirit]:
        """Get all spirits of a given element."""
        return [s for s in self.spirits if s.element == element]

    # =========================================================================
    # Traversal Methods
    # =========================================================================

    def calculate_tianmu_spirit(self, steps: int) -> TaiyiSpirit:
        """
        Calculate which spirit serves as Tianmu (天目) after given steps.

        This implements the classical algorithm:
        "命起武德，順行十六神，遇陰德大武，重留一算"

        Args:
            steps: Number of steps to traverse (from 天目周法 remainder)

        Returns:
            The spirit serving as Tianmu
        """
        if steps <= 0:
            return self.get_start_spirit()

        current_index = 0  # Start at 武德
        remaining_steps = steps

        while remaining_steps > 0:
            spirit = self._traversal_order[current_index % 16]
            remaining_steps -= 1

            # Extra count for 阴德 and 大武
            if spirit.is_zhong_liu and remaining_steps > 0:
                remaining_steps -= 1

            current_index += 1

        return self._traversal_order[(current_index - 1) % 16]

    def get_traversal_sequence(self) -> List[TaiyiSpirit]:
        """Get the full traversal sequence starting from 武德."""
        return self._traversal_order.copy()

    # =========================================================================
    # Display Methods
    # =========================================================================

    def format_spirit_table(self) -> str:
        """Format all spirits as a display table."""
        lines = [
            "┌────┬────┬────────┬────────┬────┬─────────────────────────────────┐",
            "│ #  │ 支 │ 名稱   │ Pinyin │ 行 │ 意義                            │",
            "├────┼────┼────────┼────────┼────┼─────────────────────────────────┤",
        ]

        for spirit in self.spirits:
            zl_mark = "◆" if spirit.is_zhong_liu else " "
            start_mark = "★" if spirit.is_traversal_start else " "
            marks = f"{start_mark}{zl_mark}"

            lines.append(
                f"│ {spirit.position.value:2d} │ {spirit.branch} │ "
                f"{spirit.chinese}{marks} │ {spirit.pinyin:6s} │ "
                f"{spirit.element.value} │ {spirit.meaning[:30]:30s}...│"
            )

        lines.append(
            "└────┴────┴────────┴────────┴────┴─────────────────────────────────┘"
        )
        lines.append("★ = 起點 (Traversal Start)  ◆ = 重留 (Extra Count)")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.spirits)

    def __iter__(self):
        return iter(self.spirits)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_by_chinese(key) or self.get_by_branch(key)
        elif isinstance(key, int):
            return self.get_by_traversal_index(key)
        elif isinstance(key, SpiritPosition):
            return self.get_by_position(key)
        raise KeyError(f"Invalid key type: {type(key)}")


# =============================================================================
# Five Element Relationships for Spirits
# =============================================================================

class SpiritElementRelations:
    """
    Five Element relationships specific to Taiyi spirits.

    From 太乙金鏡式經:
    "假令高叢木以呂申、太靈同類為旺，以大義、地主為相，
     武德、太簇為死，和德、大武、太陽、天道為囚，大神、大威為休"
    """

    # Generating cycle (相生): Wood→Fire→Earth→Metal→Water→Wood
    GENERATING = {
        Element.WOOD: Element.FIRE,
        Element.FIRE: Element.EARTH,
        Element.EARTH: Element.METAL,
        Element.METAL: Element.WATER,
        Element.WATER: Element.WOOD,
    }

    # Controlling cycle (相剋): Wood→Earth→Water→Fire→Metal→Wood
    CONTROLLING = {
        Element.WOOD: Element.EARTH,
        Element.EARTH: Element.WATER,
        Element.WATER: Element.FIRE,
        Element.FIRE: Element.METAL,
        Element.METAL: Element.WOOD,
    }

    @classmethod
    def get_relationship(cls, spirit1: TaiyiSpirit, spirit2: TaiyiSpirit) -> str:
        """
        Determine the relationship between two spirits based on elements.

        Returns:
            One of: 旺(wang), 相(xiang), 休(xiu), 囚(qiu), 死(si)
        """
        e1, e2 = spirit1.element, spirit2.element

        if e1 == e2:
            return "旺"  # Same element = prosperous
        elif cls.GENERATING[e2] == e1:
            return "相"  # e2 generates e1 = phase
        elif cls.GENERATING[e1] == e2:
            return "休"  # e1 generates e2 = rest
        elif cls.CONTROLLING[e1] == e2:
            return "囚"  # e1 controls e2 = imprisoned
        elif cls.CONTROLLING[e2] == e1:
            return "死"  # e2 controls e1 = dead
        else:
            return "平"  # Neutral
