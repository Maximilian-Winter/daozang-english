"""
Taiyi Shenshu Nine Palaces (九宮)

The Nine Palaces form the spatial foundation of Taiyi Shenshu.
Unlike Qimen Dunjia, Taiyi NEVER enters the center palace (五宮).

Palace arrangement follows the Luoshu (洛書) magic square:
    ┌───┬───┬───┐
    │ 4 │ 9 │ 2 │   巽  離  坤
    ├───┼───┼───┤
    │ 3 │ 5 │ 7 │   震  中  兌
    ├───┼───┼───┤
    │ 8 │ 1 │ 6 │   艮  坎  乾
    └───┴───┴───┘

From 太乙金鏡式經:
"命起一宮，順行八宮，不游中五"

碼道長存 — The Way of Code endures
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, NamedTuple, Any
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.core import Element, Direction, Trigram
from .constants import TaiyiDunType, PalaceType, TaiyiConstants
from .spirits import TaiyiSpirit, SixteenSpirits


# =============================================================================
# Palace Data Structure
# =============================================================================

@dataclass
class TaiyiPalace:
    """
    Represents a single palace in the Taiyi grid.

    Static attributes (inherent to the palace):
    - number: Palace number (1-9)
    - trigram: Associated trigram
    - direction: Compass direction
    - element: Five Element association
    - palace_type: Yang/Yin classification

    Dynamic attributes (vary by calculation):
    - taiyi: True if Taiyi is in this palace
    - tianmu: Spirit serving as Tianmu if present
    - jishen: 計神 position if present
    - generals: Dictionary of generals in this palace
    """

    # Static attributes
    number: int
    trigram: str
    trigram_name: str
    direction: Direction
    element: Element
    palace_type: PalaceType

    # Dynamic attributes (set during calculation)
    taiyi: bool = False
    tianmu: Optional[TaiyiSpirit] = None
    jishen_branch: Optional[str] = None
    heshen_branch: Optional[str] = None

    # Generals in this palace
    host_general: Optional[str] = None
    guest_general: Optional[str] = None
    host_deputy: Optional[str] = None
    guest_deputy: Optional[str] = None

    # Calculation values
    host_calculation: Optional[int] = None  # 主算
    guest_calculation: Optional[int] = None  # 客算

    def __post_init__(self):
        """Validate palace number."""
        if not 1 <= self.number <= 9:
            raise ValueError(f"Palace number must be 1-9, got {self.number}")

    @property
    def is_center(self) -> bool:
        """Check if this is the center palace (Taiyi skips this)."""
        return self.number == 5

    @property
    def is_yang(self) -> bool:
        """Check if this is a Yang palace (3, 4, 8, 9)."""
        return self.number in {3, 4, 8, 9}

    @property
    def is_yin(self) -> bool:
        """Check if this is a Yin palace (1, 2, 6, 7)."""
        return self.number in {1, 2, 6, 7}

    @property
    def favors_host(self) -> bool:
        """Check if this palace favors the host/defender (地内宫)."""
        return self.number in {3, 4, 8}  # Not 9

    @property
    def favors_guest(self) -> bool:
        """Check if this palace favors the guest/attacker (天外宫)."""
        return self.number in {2, 6, 7, 9}

    def get_opposite_palace(self) -> int:
        """Get the opposite palace number (對沖)."""
        opposites = {1: 9, 9: 1, 2: 8, 8: 2, 3: 7, 7: 3, 4: 6, 6: 4, 5: 5}
        return opposites[self.number]

    def get_summary(self) -> Dict[str, Any]:
        """Get a dictionary summary of the palace state."""
        return {
            "number": self.number,
            "trigram": f"{self.trigram} ({self.trigram_name})",
            "direction": self.direction.value,
            "element": self.element.value,
            "type": self.palace_type.value,
            "taiyi": self.taiyi,
            "tianmu": str(self.tianmu) if self.tianmu else None,
            "jishen": self.jishen_branch,
            "heshen": self.heshen_branch,
            "host_general": self.host_general,
            "guest_general": self.guest_general,
            "host_calculation": self.host_calculation,
            "guest_calculation": self.guest_calculation,
        }

    def format_display(self) -> str:
        """Format palace for display."""
        lines = [
            f"宮{self.number} {self.trigram}({self.trigram_name})",
            f"{self.direction.value} {self.element.value}",
        ]

        if self.taiyi:
            lines.append("【太乙】")
        if self.tianmu:
            lines.append(f"天目:{self.tianmu.chinese}")
        if self.jishen_branch:
            lines.append(f"計神:{self.jishen_branch}")

        return "\n".join(lines)


# =============================================================================
# Nine Palaces Container
# =============================================================================

class NinePalaces:
    """
    Container for the Nine Palaces grid.

    Provides methods for:
    - Palace lookup by number, direction, or trigram
    - Grid operations (adjacent, opposite palaces)
    - Display formatting
    """

    # Grid positions for display (row, col)
    GRID_POSITIONS = {
        4: (0, 0), 9: (0, 1), 2: (0, 2),  # Top row
        3: (1, 0), 5: (1, 1), 7: (1, 2),  # Middle row
        8: (2, 0), 1: (2, 1), 6: (2, 2),  # Bottom row
    }

    # Adjacent palace relationships
    ADJACENT = {
        1: [8, 6, 2],      # Palace 1 is adjacent to 8, 6, 2
        2: [9, 7, 1],
        3: [4, 8, 5],
        4: [9, 3, 5],
        5: [4, 9, 2, 7, 6, 3, 8],  # Center touches all except 1
        6: [7, 1, 8],
        7: [2, 6, 5],
        8: [3, 1, 5],
        9: [4, 2, 5],
    }

    def __init__(self):
        self.palaces: Dict[int, TaiyiPalace] = self._create_palaces()

    def _create_palaces(self) -> Dict[int, TaiyiPalace]:
        """Create all nine palaces with their static attributes."""
        palace_data = [
            (1, "坎", "Kan/Water", Direction.NORTH, Element.WATER, PalaceType.YIN_PALACE),
            (2, "坤", "Kun/Earth", Direction.SOUTHWEST, Element.EARTH, PalaceType.YIN_PALACE),
            (3, "震", "Zhen/Thunder", Direction.EAST, Element.WOOD, PalaceType.YANG_PALACE),
            (4, "巽", "Xun/Wind", Direction.SOUTHEAST, Element.WOOD, PalaceType.YANG_PALACE),
            (5, "中", "Center", Direction.CENTER, Element.EARTH, PalaceType.CENTER),
            (6, "乾", "Qian/Heaven", Direction.NORTHWEST, Element.METAL, PalaceType.YIN_PALACE),
            (7, "兌", "Dui/Lake", Direction.WEST, Element.METAL, PalaceType.YIN_PALACE),
            (8, "艮", "Gen/Mountain", Direction.NORTHEAST, Element.EARTH, PalaceType.YANG_PALACE),
            (9, "離", "Li/Fire", Direction.SOUTH, Element.FIRE, PalaceType.YANG_PALACE),
        ]

        palaces = {}
        for num, trigram, name, direction, element, ptype in palace_data:
            palaces[num] = TaiyiPalace(
                number=num,
                trigram=trigram,
                trigram_name=name,
                direction=direction,
                element=element,
                palace_type=ptype
            )
        return palaces

    def get_palace(self, number: int) -> Optional[TaiyiPalace]:
        """Get palace by number (1-9)."""
        return self.palaces.get(number)

    def get_palace_by_direction(self, direction: Direction) -> Optional[TaiyiPalace]:
        """Get palace by direction."""
        for palace in self.palaces.values():
            if palace.direction == direction:
                return palace
        return None

    def get_palace_by_trigram(self, trigram: str) -> Optional[TaiyiPalace]:
        """Get palace by trigram character."""
        for palace in self.palaces.values():
            if palace.trigram == trigram:
                return palace
        return None

    def get_outer_palaces(self) -> List[TaiyiPalace]:
        """Get all outer palaces (excluding center palace 5)."""
        return [p for p in self.palaces.values() if p.number != 5]

    def get_adjacent_palaces(self, palace_number: int) -> List[TaiyiPalace]:
        """Get palaces adjacent to the given palace."""
        adjacent_nums = self.ADJACENT.get(palace_number, [])
        return [self.palaces[n] for n in adjacent_nums if n in self.palaces]

    def get_opposite_palace(self, palace_number: int) -> Optional[TaiyiPalace]:
        """Get the opposite palace (對沖)."""
        palace = self.palaces.get(palace_number)
        if palace:
            opposite_num = palace.get_opposite_palace()
            return self.palaces.get(opposite_num)
        return None

    def get_yang_palaces(self) -> List[TaiyiPalace]:
        """Get all Yang palaces (3, 4, 8, 9)."""
        return [p for p in self.palaces.values() if p.is_yang]

    def get_yin_palaces(self) -> List[TaiyiPalace]:
        """Get all Yin palaces (1, 2, 6, 7)."""
        return [p for p in self.palaces.values() if p.is_yin]

    def reset_dynamic_attributes(self) -> None:
        """Reset all dynamic attributes to prepare for new calculation."""
        for palace in self.palaces.values():
            palace.taiyi = False
            palace.tianmu = None
            palace.jishen_branch = None
            palace.heshen_branch = None
            palace.host_general = None
            palace.guest_general = None
            palace.host_deputy = None
            palace.guest_deputy = None
            palace.host_calculation = None
            palace.guest_calculation = None

    def to_grid_display(self) -> str:
        """Format palaces as a 3x3 grid display."""
        # Create empty 3x3 grid
        grid = [[None, None, None] for _ in range(3)]

        # Fill grid with palace displays
        for num, (row, col) in self.GRID_POSITIONS.items():
            palace = self.palaces[num]
            grid[row][col] = palace.format_display()

        # Format as string
        lines = ["┌" + "─" * 20 + "┬" + "─" * 20 + "┬" + "─" * 20 + "┐"]

        for row_idx, row in enumerate(grid):
            # Split each cell into lines
            cell_lines = [cell.split("\n") if cell else [""] for cell in row]
            max_lines = max(len(cl) for cl in cell_lines)

            for line_idx in range(max_lines):
                line_parts = []
                for cell in cell_lines:
                    if line_idx < len(cell):
                        line_parts.append(f"{cell[line_idx]:^20}")
                    else:
                        line_parts.append(" " * 20)
                lines.append("│" + "│".join(line_parts) + "│")

            if row_idx < 2:
                lines.append("├" + "─" * 20 + "┼" + "─" * 20 + "┼" + "─" * 20 + "┤")

        lines.append("└" + "─" * 20 + "┴" + "─" * 20 + "┴" + "─" * 20 + "┘")

        return "\n".join(lines)

    def __getitem__(self, key: int) -> TaiyiPalace:
        """Allow dict-like access: palaces[1]."""
        if key not in self.palaces:
            raise KeyError(f"Palace {key} not found")
        return self.palaces[key]

    def __iter__(self):
        """Iterate over palaces in numerical order."""
        return iter(sorted(self.palaces.values(), key=lambda p: p.number))

    def __len__(self):
        return len(self.palaces)


# =============================================================================
# Taiyi Plate (Complete Calculation Result)
# =============================================================================

class TaiyiPlate(NamedTuple):
    """
    Complete immutable result of a Taiyi Shenshu calculation.

    Contains all computed positions and values for a given datetime.
    """

    # Datetime information
    datetime_info: Dict[str, Any]

    # Dun type and era
    dun_type: TaiyiDunType
    era: 'TaiyiEra'  # Forward reference
    era_year: int  # Year within the era (1-10)

    # Core positions
    taiyi_palace: int
    years_in_palace: int
    tianmu_spirit: TaiyiSpirit
    jishen_branch: str  # 計神
    heshen_branch: str  # 合神

    # Calculations
    host_calculation: int  # 主算
    guest_calculation: int  # 客算

    # Generals (palace positions)
    host_general_palace: int      # 主大将
    guest_general_palace: int     # 客大将
    host_deputy_palace: int       # 主参将
    guest_deputy_palace: int      # 客参将

    # Eyes (始击/文昌)
    shiji_spirit: Optional[TaiyiSpirit] = None  # 始击将 (Upper Eye)
    wenchang_palace: Optional[int] = None        # 文昌将 (Lower Eye)

    # Complete palace states
    palaces: Dict[int, TaiyiPalace] = {}

    # Full generals data (optional, for detailed analysis)
    generals: Optional[Dict[str, Any]] = None

    # Analysis results
    is_calculation_harmonious: bool = False  # 算和
    favors_host: bool = False
    favors_guest: bool = False

    # Battle advantage analysis
    battle_advantage: Optional[Dict[str, Any]] = None
    special_formations: Optional[List[Dict[str, str]]] = None

    # Eight Gates data
    ruling_gate: Optional[str] = None
    gate_analysis: Optional[Dict[str, Any]] = None

    def get_palace(self, number: int) -> Optional[TaiyiPalace]:
        """Get a specific palace by number."""
        return self.palaces.get(number)

    def get_taiyi_palace_info(self) -> TaiyiPalace:
        """Get the palace where Taiyi is located."""
        return self.palaces[self.taiyi_palace]

    def format_summary(self) -> str:
        """Format a summary of the plate."""
        lines = [
            "═" * 60,
            "太乙神數 Taiyi Shenshu Calculation",
            "═" * 60,
            f"遁類: {self.dun_type.chinese}",
            f"紀元: {self.era.chinese_name} ({self.era.pattern_name})",
            f"紀內年: {self.era_year}",
            "",
            "【核心位置】",
            f"太乙: 宮{self.taiyi_palace} (入宮{self.years_in_palace}年)",
            f"天目: {self.tianmu_spirit.chinese} ({self.tianmu_spirit.branch})",
            f"計神: {self.jishen_branch}",
            f"合神: {self.heshen_branch}",
            "",
            "【主客計算】",
            f"主算: {self.host_calculation}",
            f"客算: {self.guest_calculation}",
            f"算和: {'是' if self.is_calculation_harmonious else '否'}",
            "",
            "【五将】",
        ]

        # Add eye positions if available
        if self.shiji_spirit:
            lines.append(f"始击将(上目): {self.shiji_spirit.chinese}")
        if self.wenchang_palace:
            lines.append(f"文昌将(下目): 宮{self.wenchang_palace}")

        lines.extend([
            f"主大將: 宮{self.host_general_palace}",
            f"客大將: 宮{self.guest_general_palace}",
            f"主參將: 宮{self.host_deputy_palace}",
            f"客參將: 宮{self.guest_deputy_palace}",
            "",
            "【勝負判斷】",
            f"利主: {'是' if self.favors_host else '否'}",
            f"利客: {'是' if self.favors_guest else '否'}",
        ])

        # Add battle advantage if available
        if self.battle_advantage:
            adv = self.battle_advantage.get("advantage_chinese", "")
            if adv:
                lines.append(f"判斷: {adv}")

        # Add special formations if available
        if self.special_formations:
            lines.append("")
            lines.append("【特殊格局】")
            for formation in self.special_formations:
                lines.append(f"  • {formation.get('name', '')}: {formation.get('description', '')}")

        # Add gate information if available
        if self.gate_analysis:
            lines.append("")
            lines.append("【八門】")
            lines.append(f"直門: {self.ruling_gate or 'N/A'}")
            if self.gate_analysis.get("harmony"):
                harmony = self.gate_analysis["harmony"]
                lines.append(f"太乙門: {harmony.get('taiyi_gate', 'N/A')}")
                lines.append(f"主將門: {harmony.get('host_gate', 'N/A')}")
                lines.append(f"客將門: {harmony.get('guest_gate', 'N/A')}")
                lines.append(f"門合: {harmony.get('interpretation', 'N/A')}")

        lines.append("═" * 60)
        return "\n".join(lines)


# Import for forward reference resolution
from .constants import TaiyiEra
