"""
Da Liu Ren Plate Structures (大六壬盤結構)

Plate structures for Liu Ren divination:
- LiuRenPalace: Individual palace in the 12-position system
- TwelvePalaces: Container for all 12 palaces
- HeavenPlate (天盤): The rotating plate based on monthly general
- EarthPlate (地盤): The fixed plate of 12 branches
- LiuRenPlate: Complete plate combining all elements

The Liu Ren system uses a 12-palace system (unlike Qimen's 9),
representing the 12 Earthly Branches arranged in a circle.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..core.core import Element, Direction
from .components import TianJiang, SiKe, SanChuan


# =============================================================================
# Branch Properties (支屬性)
# =============================================================================

# Element mapping for branches
BRANCH_ELEMENT_MAP: Dict[str, Element] = {
    '子': Element.WATER, '丑': Element.EARTH, '寅': Element.WOOD,
    '卯': Element.WOOD,  '辰': Element.EARTH, '巳': Element.FIRE,
    '午': Element.FIRE,  '未': Element.EARTH, '申': Element.METAL,
    '酉': Element.METAL, '戌': Element.EARTH, '亥': Element.WATER,
}

# Direction mapping for branches
BRANCH_DIRECTION_MAP: Dict[str, Direction] = {
    '子': Direction.NORTH,
    '丑': Direction.NORTHEAST,
    '寅': Direction.NORTHEAST,
    '卯': Direction.EAST,
    '辰': Direction.SOUTHEAST,
    '巳': Direction.SOUTHEAST,
    '午': Direction.SOUTH,
    '未': Direction.SOUTHWEST,
    '申': Direction.SOUTHWEST,
    '酉': Direction.WEST,
    '戌': Direction.NORTHWEST,
    '亥': Direction.NORTHWEST,
}


# =============================================================================
# LiuRenPalace (六壬宮位)
# =============================================================================

@dataclass
class LiuRenPalace:
    """
    Individual palace in the Liu Ren 12-position system.

    Each palace corresponds to one of the 12 Earthly Branches
    and can hold information about:
    - Its fixed earth branch (地支)
    - The overlaid heaven branch (天盤支)
    - Any heavenly general positioned here

    Attributes:
        position: Position number (1-12)
        earth_branch: The fixed earthly branch for this palace
        element: Five Element association
        direction: Directional association
        heaven_branch: The overlaid branch from the Heaven Plate
        general: The Heavenly General at this position
    """
    position: int
    earth_branch: str
    element: Element
    direction: Direction

    # Dynamic content (set during calculation)
    heaven_branch: Optional[str] = None
    general: Optional[TianJiang] = None

    def __str__(self) -> str:
        hb = self.heaven_branch or '?'
        return f"宮{self.position}({self.earth_branch}): 天盤{hb}"

    def __repr__(self) -> str:
        return f"LiuRenPalace({self.position}, {self.earth_branch})"

    @property
    def is_same_branch(self) -> bool:
        """Check if heaven and earth branches are the same (伏吟)."""
        return self.heaven_branch == self.earth_branch

    @property
    def is_clash_branch(self) -> bool:
        """Check if heaven and earth branches clash (反吟)."""
        from .constants import SIX_CLASHES
        if self.heaven_branch:
            return SIX_CLASHES.get(self.earth_branch) == self.heaven_branch
        return False

    def reset(self) -> None:
        """Reset dynamic attributes for recalculation."""
        self.heaven_branch = None
        self.general = None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this palace."""
        return {
            'position': self.position,
            'earth_branch': self.earth_branch,
            'heaven_branch': self.heaven_branch,
            'element': self.element.value if self.element else None,
            'direction': self.direction.value if self.direction else None,
            'general': self.general.chinese if self.general else None,
            'is_fu_yin': self.is_same_branch,
            'is_fan_yin': self.is_clash_branch,
        }


# =============================================================================
# TwelvePalaces (十二宮)
# =============================================================================

class TwelvePalaces:
    """
    Container for the 12 palaces of Liu Ren divination.

    The 12 palaces are arranged in a circle corresponding
    to the 12 Earthly Branches, starting from 子 (North).

    Provides lookup by:
    - Position number (1-12)
    - Branch character
    """

    def __init__(self):
        self.palaces: List[LiuRenPalace] = self._create_palaces()
        self._build_indices()

    def _create_palaces(self) -> List[LiuRenPalace]:
        """Create all 12 palaces."""
        from .constants import TWELVE_BRANCHES

        palaces = []
        for i, branch in enumerate(TWELVE_BRANCHES):
            palace = LiuRenPalace(
                position=i + 1,
                earth_branch=branch,
                element=BRANCH_ELEMENT_MAP.get(branch, Element.EARTH),
                direction=BRANCH_DIRECTION_MAP.get(branch, Direction.CENTER),
            )
            palaces.append(palace)
        return palaces

    def _build_indices(self) -> None:
        """Build lookup indices."""
        self.by_branch: Dict[str, LiuRenPalace] = {
            p.earth_branch: p for p in self.palaces
        }
        self.by_position: Dict[int, LiuRenPalace] = {
            p.position: p for p in self.palaces
        }

    def get_by_branch(self, branch: str) -> Optional[LiuRenPalace]:
        """Get palace by its earth branch."""
        return self.by_branch.get(branch)

    def get_by_position(self, position: int) -> Optional[LiuRenPalace]:
        """Get palace by position number (1-12)."""
        return self.by_position.get(position)

    def reset_all(self) -> None:
        """Reset all palaces for recalculation."""
        for palace in self.palaces:
            palace.reset()

    def get_fu_yin_palaces(self) -> List[LiuRenPalace]:
        """Get all palaces with 伏吟 (same branch)."""
        return [p for p in self.palaces if p.is_same_branch]

    def get_fan_yin_palaces(self) -> List[LiuRenPalace]:
        """Get all palaces with 反吟 (clashing branches)."""
        return [p for p in self.palaces if p.is_clash_branch]

    def __getitem__(self, key) -> Optional[LiuRenPalace]:
        """Flexible accessor."""
        if isinstance(key, int):
            return self.get_by_position(key)
        elif isinstance(key, str):
            return self.get_by_branch(key)
        return None

    def __iter__(self):
        return iter(self.palaces)

    def __len__(self) -> int:
        return len(self.palaces)


# =============================================================================
# HeavenPlate (天盤)
# =============================================================================

@dataclass
class HeavenPlate:
    """
    天盤 - The Heaven Plate

    The Heaven Plate rotates based on:
    1. The Monthly General (月將) - which branch represents the month
    2. The Query Hour (占時) - the hour branch when divination is cast

    The Monthly General is placed at the Query Hour position,
    and all other branches rotate accordingly.

    Attributes:
        monthly_general_branch: The branch of the current monthly general
        query_hour_branch: The branch of the query hour
        positions: Mapping of earth branches to overlaid heaven branches
    """
    monthly_general_branch: str
    query_hour_branch: str
    positions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.positions:
            self._calculate_positions()

    def _calculate_positions(self) -> None:
        """
        Calculate the overlay positions.

        Algorithm:
        1. The Monthly General branch is placed at the Query Hour position
        2. Other branches follow in sequence
        """
        from .constants import TWELVE_BRANCHES, get_branch_index, get_offset_branch

        # Calculate offset: how many positions to shift
        # Monthly general goes to query hour position
        mg_idx = get_branch_index(self.monthly_general_branch)
        qh_idx = get_branch_index(self.query_hour_branch)
        offset = qh_idx - mg_idx

        # For each earth branch position, calculate which heaven branch overlays it
        for earth_branch in TWELVE_BRANCHES:
            earth_idx = get_branch_index(earth_branch)
            # The heaven branch at this position is the one that's been shifted
            heaven_idx = (earth_idx - offset) % 12
            heaven_branch = TWELVE_BRANCHES[heaven_idx]
            self.positions[earth_branch] = heaven_branch

    def get_heaven_branch_at(self, earth_branch: str) -> str:
        """Get the heaven branch overlaying a given earth branch position."""
        return self.positions.get(earth_branch, earth_branch)

    def get_earth_branch_for(self, heaven_branch: str) -> Optional[str]:
        """Get the earth branch position where a heaven branch falls."""
        for earth, heaven in self.positions.items():
            if heaven == heaven_branch:
                return earth
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the Heaven Plate."""
        return {
            'monthly_general': self.monthly_general_branch,
            'query_hour': self.query_hour_branch,
            'overlay': self.positions.copy(),
        }


# =============================================================================
# EarthPlate (地盤)
# =============================================================================

@dataclass
class EarthPlate:
    """
    地盤 - The Earth Plate

    The Earth Plate is fixed and represents the 12 Earthly Branches
    in their standard positions. It serves as the reference frame
    for the rotating Heaven Plate.

    The Earth Plate never changes - it is always:
    子→子, 丑→丑, 寅→寅, ..., 亥→亥
    """
    positions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.positions:
            from .constants import TWELVE_BRANCHES
            self.positions = {b: b for b in TWELVE_BRANCHES}

    def get_branch_at(self, position: str) -> str:
        """Get the branch at a given position (always same as position)."""
        return self.positions.get(position, position)


# =============================================================================
# LiuRenPlate (六壬盤)
# =============================================================================

@dataclass
class LiuRenPlate:
    """
    Complete Liu Ren Plate for a given datetime.

    This is the main output of Liu Ren divination calculation,
    containing all the calculated elements:
    - Date/time information
    - Day stem and branch (日干支)
    - Hour branch (時支)
    - Monthly General (月將)
    - Heaven and Earth Plates (天盤/地盤)
    - The 12 Palaces with overlays and generals
    - Four Lessons (四課)
    - Three Transmissions (三傳)
    - Noble Person position (貴人)
    - Identified patterns (課體)
    - Special conditions (伏吟/反吟)
    """
    # Time information
    query_datetime: datetime
    lunar_date: Any  # LunarDate from core.lunar_calendar

    # Day pillar
    day_stem: str
    day_branch: str

    # Hour information
    hour_branch: str
    is_daytime: bool

    # Monthly General
    monthly_general_name: str
    monthly_general_branch: str

    # Plates
    heaven_plate: HeavenPlate
    earth_plate: EarthPlate
    palaces: TwelvePalaces

    # Core divination results
    si_ke: SiKe
    san_chuan: SanChuan

    # Noble Person
    noble_person_branch: str
    noble_person: Optional[TianJiang] = None

    # Generals positioned on the plate
    general_positions: Dict[str, TianJiang] = field(default_factory=dict)

    # Pattern identification
    lesson_pattern: str = ""
    special_patterns: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"六壬盤 {self.query_datetime.strftime('%Y-%m-%d %H:%M')}\n"
            f"日干支: {self.day_stem}{self.day_branch}\n"
            f"時支: {self.hour_branch} ({'晝' if self.is_daytime else '夜'})\n"
            f"月將: {self.monthly_general_name}({self.monthly_general_branch})\n"
            f"{self.si_ke}\n"
            f"{self.san_chuan}"
        )

    def __repr__(self) -> str:
        return f"LiuRenPlate({self.day_stem}{self.day_branch}, {self.hour_branch})"

    def get_heaven_branch_at(self, earth_branch: str) -> str:
        """Get the heaven branch at a given earth position."""
        return self.heaven_plate.get_heaven_branch_at(earth_branch)

    def get_general_at(self, branch: str) -> Optional[TianJiang]:
        """Get the Heavenly General at a given branch position."""
        return self.general_positions.get(branch)

    def get_palace_by_branch(self, branch: str) -> Optional[LiuRenPalace]:
        """Get palace by earth branch."""
        return self.palaces.get_by_branch(branch)

    def is_fu_yin(self) -> bool:
        """
        Check if this is a 伏吟 (Hidden/Buried) pattern.
        All heaven branches are same as earth branches.
        """
        fu_yin_count = len(self.palaces.get_fu_yin_palaces())
        return fu_yin_count == 12

    def is_fan_yin(self) -> bool:
        """
        Check if this is a 反吟 (Reverse) pattern.
        All heaven branches clash with earth branches.
        """
        fan_yin_count = len(self.palaces.get_fan_yin_palaces())
        return fan_yin_count == 12

    def format_display(self) -> str:
        """Format the plate for text display."""
        lines = []
        lines.append("=" * 60)
        dt = self.query_datetime
        date_str = f"{dt.year}年{dt.month}月{dt.day}日 {dt.hour:02d}:{dt.minute:02d}"
        lines.append(f"  大六壬盤 - {date_str}")
        lines.append("=" * 60)
        lines.append("")

        # Basic info
        lines.append(f"  日干支: {self.day_stem}{self.day_branch}")
        lines.append(f"  時支: {self.hour_branch} ({'晝占' if self.is_daytime else '夜占'})")
        lines.append(f"  月將: {self.monthly_general_name} ({self.monthly_general_branch})")
        lines.append(f"  貴人: {self.noble_person.chinese if self.noble_person else '?'} "
                    f"({self.noble_person_branch})")
        lines.append("")

        # Four Lessons
        lines.append("  【四課】")
        for ke in self.si_ke.lessons:
            rel = ke.get_relation()
            rel_str = f" [{rel.chinese}]" if rel.chinese != "無" else ""
            lines.append(f"    第{ke.index}課: {ke.shang} / {ke.xia}{rel_str}")
        lines.append("")

        # Three Transmissions
        lines.append("  【三傳】")
        lines.append(f"    初傳: {self.san_chuan.chu_chuan}")
        lines.append(f"    中傳: {self.san_chuan.zhong_chuan}")
        lines.append(f"    末傳: {self.san_chuan.mo_chuan}")
        lines.append(f"    取法: {self.san_chuan.derivation_method}")
        lines.append("")

        # Patterns
        if self.lesson_pattern:
            lines.append(f"  【課體】{self.lesson_pattern}")

        if self.special_patterns:
            lines.append(f"  【特殊】{', '.join(self.special_patterns)}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Get a complete summary of the plate."""
        return {
            'datetime': self.query_datetime.isoformat(),
            'day_stem': self.day_stem,
            'day_branch': self.day_branch,
            'hour_branch': self.hour_branch,
            'is_daytime': self.is_daytime,
            'monthly_general': {
                'name': self.monthly_general_name,
                'branch': self.monthly_general_branch,
            },
            'noble_person': {
                'branch': self.noble_person_branch,
                'general': self.noble_person.chinese if self.noble_person else None,
            },
            'si_ke': self.si_ke.get_summary(),
            'san_chuan': self.san_chuan.get_summary(),
            'lesson_pattern': self.lesson_pattern,
            'special_patterns': self.special_patterns,
            'is_fu_yin': self.is_fu_yin(),
            'is_fan_yin': self.is_fan_yin(),
            'general_positions': {
                branch: gen.chinese
                for branch, gen in self.general_positions.items()
            },
        }
