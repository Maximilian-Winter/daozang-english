"""
Qimen Dunjia Palace Structures

This module defines the Nine Palaces grid and the complete plate structure.

The Nine Palaces (九宮) are arranged according to the Later Heaven (後天)
Bagua pattern, with each palace containing:
- A trigram association
- Direction and element
- Rotating elements: Star, Gate, Spirit, Earth/Heaven plate stems

Later Heaven arrangement:
    巽(4)  離(9)  坤(2)
    震(3)  中(5)  兌(7)
    艮(8)  坎(1)  乾(6)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, NamedTuple, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.core import Trigram, Direction, Element, DunType, Yuan, SolarTerm
from .components import NineStar, EightGate, EightSpirit


# =============================================================================
# Palace Structure
# =============================================================================

@dataclass
class Palace:
    """
    Represents a single palace in the Nine Palaces grid.

    A palace contains both static attributes (number, trigram, direction, element)
    and dynamic content that changes based on the calculation (stems, star, gate, spirit).
    """
    # Static attributes
    number: int
    trigram: Optional[Trigram]
    direction: Direction
    base_element: Element

    # Dynamic content (populated during plate calculation)
    earth_plate_stem: Optional[str] = None      # 地盘干
    heaven_plate_stem: Optional[str] = None     # 天盘干
    star: Optional[NineStar] = None             # 九星
    gate: Optional[EightGate] = None            # 八门
    spirit: Optional[EightSpirit] = None        # 八神

    def __str__(self) -> str:
        trigram_str = self.trigram.chinese if self.trigram else "中"
        return f"Palace {self.number} ({trigram_str})"

    def __repr__(self) -> str:
        return (f"Palace({self.number}, trigram={self.trigram.chinese if self.trigram else 'None'}, "
                f"earth={self.earth_plate_stem}, heaven={self.heaven_plate_stem})")

    @property
    def element(self) -> Element:
        """Get the palace's base element"""
        return self.base_element

    def get_summary(self) -> Dict[str, Any]:
        """Get a dictionary summary of the palace contents"""
        return {
            'number': self.number,
            'trigram': self.trigram.chinese if self.trigram else "中",
            'direction': self.direction.value,
            'element': self.base_element.value,
            'earth_stem': self.earth_plate_stem,
            'heaven_stem': self.heaven_plate_stem,
            'star': self.star.chinese if self.star else None,
            'gate': self.gate.chinese if self.gate else None,
            'spirit': self.spirit.chinese if self.spirit else None,
        }

    def has_three_wonders(self) -> bool:
        """Check if this palace has one of the Three Wonders (三奇: 乙丙丁)"""
        three_wonders = ['乙', '丙', '丁']
        return (self.heaven_plate_stem in three_wonders or
                self.earth_plate_stem in three_wonders)

    def get_stem_combination_name(self) -> Optional[str]:
        """
        Get the name of special stem combinations if present.

        Special combinations include:
        - 伏吟 (Fu Yin): Heaven and Earth stems are the same
        - 反吟 (Fan Yin): Heaven and Earth stems are opposing
        """
        if not self.earth_plate_stem or not self.heaven_plate_stem:
            return None

        if self.earth_plate_stem == self.heaven_plate_stem:
            return "伏吟"  # Same stem - stagnation

        # Check for opposing stems (6 positions apart in the 10-stem cycle)
        stems = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
        if self.earth_plate_stem in stems and self.heaven_plate_stem in stems:
            earth_idx = stems.index(self.earth_plate_stem)
            heaven_idx = stems.index(self.heaven_plate_stem)
            if abs(earth_idx - heaven_idx) == 5:
                return "反吟"  # Opposing stem - reversal

        return None


# =============================================================================
# Nine Palaces Container
# =============================================================================

class NinePalaces:
    """
    Container class for the Nine Palaces grid.

    Provides access to individual palaces and methods for grid-based operations.
    The palaces are arranged in the Later Heaven (後天) Bagua pattern.
    """

    def __init__(self):
        # Create the nine palaces with their static attributes
        self.palaces: Dict[int, Palace] = {
            1: Palace(1, Trigram.KAN, Direction.NORTH, Element.WATER),
            2: Palace(2, Trigram.KUN, Direction.SOUTHWEST, Element.EARTH),
            3: Palace(3, Trigram.ZHEN, Direction.EAST, Element.WOOD),
            4: Palace(4, Trigram.XUN, Direction.SOUTHEAST, Element.WOOD),
            5: Palace(5, None, Direction.CENTER, Element.EARTH),  # Center has no trigram
            6: Palace(6, Trigram.QIAN, Direction.NORTHWEST, Element.METAL),
            7: Palace(7, Trigram.DUI, Direction.WEST, Element.METAL),
            8: Palace(8, Trigram.GEN, Direction.NORTHEAST, Element.EARTH),
            9: Palace(9, Trigram.LI, Direction.SOUTH, Element.FIRE),
        }

        # Grid layout for display (row, column) positions
        # Grid is 3x3:
        #   巽(4)  離(9)  坤(2)
        #   震(3)  中(5)  兌(7)
        #   艮(8)  坎(1)  乾(6)
        self._grid_positions = {
            4: (0, 0), 9: (0, 1), 2: (0, 2),
            3: (1, 0), 5: (1, 1), 7: (1, 2),
            8: (2, 0), 1: (2, 1), 6: (2, 2),
        }

        # Reverse lookup
        self._position_to_palace = {v: k for k, v in self._grid_positions.items()}

    def get_palace(self, number: int) -> Optional[Palace]:
        """Get palace by number (1-9)"""
        return self.palaces.get(number)

    def get_palace_at_position(self, row: int, col: int) -> Optional[Palace]:
        """Get palace at grid position (row, col)"""
        palace_num = self._position_to_palace.get((row, col))
        if palace_num:
            return self.palaces[palace_num]
        return None

    def get_grid_position(self, palace_number: int) -> Optional[tuple]:
        """Get grid position (row, col) for a palace number"""
        return self._grid_positions.get(palace_number)

    def get_adjacent_palaces(self, palace_number: int) -> list:
        """
        Get adjacent palace numbers.

        Adjacency is based on the Luo Shu magic square relationships.
        """
        adjacency = {
            1: [8, 6, 3, 9],      # Kan: Gen, Qian, Zhen, Li (no Kun direct)
            2: [9, 7, 5],          # Kun: Li, Dui, Center
            3: [4, 8, 5, 1],       # Zhen: Xun, Gen, Center, Kan
            4: [9, 3, 5],          # Xun: Li, Zhen, Center
            5: [2, 4, 6, 8],       # Center: all corners
            6: [1, 7, 5],          # Qian: Kan, Dui, Center
            7: [2, 6, 5, 8],       # Dui: Kun, Qian, Center, Gen
            8: [3, 1, 5, 6],       # Gen: Zhen, Kan, Center, Qian
            9: [4, 2, 5, 3, 7],    # Li: Xun, Kun, Center, Zhen, Dui
        }
        return adjacency.get(palace_number, [])

    def get_opposite_palace(self, palace_number: int) -> Optional[int]:
        """Get the palace opposite to the given palace through the center"""
        opposites = {
            1: 9, 9: 1,  # Kan <-> Li
            2: 8, 8: 2,  # Kun <-> Gen
            3: 7, 7: 3,  # Zhen <-> Dui
            4: 6, 6: 4,  # Xun <-> Qian
            5: 5,        # Center is its own opposite
        }
        return opposites.get(palace_number)

    def create_fresh_copy(self) -> 'NinePalaces':
        """Create a fresh copy of the nine palaces with cleared dynamic content"""
        new_palaces = NinePalaces()
        return new_palaces

    def to_grid_display(self) -> str:
        """Generate a text display of the nine palaces grid"""
        lines = []

        # Grid layout
        grid_order = [[4, 9, 2], [3, 5, 7], [8, 1, 6]]

        for row in grid_order:
            row_parts = []
            for palace_num in row:
                palace = self.palaces[palace_num]
                cell = f"[{palace_num}]"

                if palace.trigram:
                    cell += f" {palace.trigram.chinese}"
                else:
                    cell += " 中"

                if palace.earth_plate_stem:
                    cell += f" {palace.earth_plate_stem}"
                if palace.heaven_plate_stem:
                    cell += f"/{palace.heaven_plate_stem}"

                if palace.star:
                    cell += f" {palace.star.chinese[:2]}"
                if palace.gate:
                    cell += f" {palace.gate.chinese[:2]}"

                row_parts.append(cell.ljust(20))

            lines.append(" | ".join(row_parts))
            lines.append("-" * 65)

        return "\n".join(lines)


# =============================================================================
# Qimen Plate - Complete Calculation Result
# =============================================================================

class QimenPlate(NamedTuple):
    """
    Complete Qimen Dunjia plate configuration for a specific moment.

    This NamedTuple contains all the information needed to interpret
    a Qimen reading, including the calculation parameters and all
    plate positions.
    """
    # Time information (from lunar calendar)
    datetime_info: Any  # LunarDate from lunar_calendar.py

    # Calculation parameters
    dun_type: DunType           # Yang or Yin Dun
    yuan: Yuan                  # Upper, Middle, or Lower Yuan
    ju_number: int              # Ju configuration number (1-9)
    solar_term: SolarTerm       # Current solar term

    # Duty elements
    duty_star: Optional[NineStar]   # 值符 - The star at duty
    duty_gate: Optional[EightGate]  # 值使 - The gate at duty
    duty_palace: int                # Palace where duty star/gate reside

    # Complete palace configurations
    palaces: Dict[int, Palace]      # Palace number -> Palace object

    # Individual plate mappings for quick access
    earth_plate: Dict[int, str]     # Palace number -> earth stem
    heaven_plate: Dict[int, str]    # Palace number -> heaven stem
    star_positions: Dict[int, NineStar]     # Palace number -> star
    gate_positions: Dict[int, EightGate]    # Palace number -> gate
    spirit_positions: Dict[int, EightSpirit]  # Palace number -> spirit

    def get_palace(self, number: int) -> Optional[Palace]:
        """Get palace by number"""
        return self.palaces.get(number)

    def get_palace_by_direction(self, direction: Direction) -> Optional[Palace]:
        """Get palace by direction"""
        direction_map = {
            Direction.NORTH: 1,
            Direction.SOUTHWEST: 2,
            Direction.EAST: 3,
            Direction.SOUTHEAST: 4,
            Direction.CENTER: 5,
            Direction.NORTHWEST: 6,
            Direction.WEST: 7,
            Direction.NORTHEAST: 8,
            Direction.SOUTH: 9,
        }
        palace_num = direction_map.get(direction)
        if palace_num:
            return self.palaces.get(palace_num)
        return None

    def find_stem_palace(self, stem: str, plate: str = 'heaven') -> Optional[int]:
        """
        Find which palace contains a specific stem.

        Args:
            stem: The stem to find (e.g., '乙', '戊')
            plate: 'heaven' or 'earth'

        Returns:
            Palace number or None if not found
        """
        target_plate = self.heaven_plate if plate == 'heaven' else self.earth_plate
        for palace_num, palace_stem in target_plate.items():
            if palace_stem == stem:
                return palace_num
        return None

    def find_star_palace(self, star_chinese: str) -> Optional[int]:
        """Find which palace contains a specific star"""
        for palace_num, star in self.star_positions.items():
            if star and star.chinese == star_chinese:
                return palace_num
        return None

    def find_gate_palace(self, gate_chinese: str) -> Optional[int]:
        """Find which palace contains a specific gate"""
        for palace_num, gate in self.gate_positions.items():
            if gate and gate.chinese == gate_chinese:
                return palace_num
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary dictionary of the plate"""
        return {
            'dun_type': self.dun_type.chinese,
            'yuan': self.yuan.chinese,
            'ju_number': self.ju_number,
            'solar_term': self.solar_term.chinese,
            'duty_star': self.duty_star.chinese if self.duty_star else None,
            'duty_gate': self.duty_gate.chinese if self.duty_gate else None,
            'duty_palace': self.duty_palace,
            'four_pillars': {
                'year': f"{self.datetime_info.year_stem.chinese}{self.datetime_info.year_branch.chinese}",
                'month': f"{self.datetime_info.month_stem.chinese}{self.datetime_info.month_branch.chinese}",
                'day': f"{self.datetime_info.day_stem.chinese}{self.datetime_info.day_branch.chinese}",
                'hour': f"{self.datetime_info.hour_stem.chinese}{self.datetime_info.hour_branch.chinese}",
            }
        }

    def format_display(self) -> str:
        """Format the plate for text display"""
        lines = []
        lines.append("=" * 70)
        lines.append("                    奇門遁甲盤 - QIMEN DUNJIA PLATE")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  遁法: {self.dun_type.chinese}    元: {self.yuan.chinese}    "
                    f"局: {self.ju_number}    節氣: {self.solar_term.chinese}")
        lines.append(f"  值符: {self.duty_star.chinese if self.duty_star else '-'}    "
                    f"值使: {self.duty_gate.chinese if self.duty_gate else '-'}    "
                    f"落宮: {self.duty_palace}")
        lines.append("")

        # Four pillars
        lines.append(f"  四柱: {self.datetime_info.year_stem.chinese}{self.datetime_info.year_branch.chinese} "
                    f"{self.datetime_info.month_stem.chinese}{self.datetime_info.month_branch.chinese} "
                    f"{self.datetime_info.day_stem.chinese}{self.datetime_info.day_branch.chinese} "
                    f"{self.datetime_info.hour_stem.chinese}{self.datetime_info.hour_branch.chinese}")
        lines.append("")
        lines.append("-" * 70)

        # Grid display
        grid_order = [[4, 9, 2], [3, 5, 7], [8, 1, 6]]

        for row in grid_order:
            # First line: Palace number, trigram, direction
            row1_parts = []
            for palace_num in row:
                palace = self.palaces[palace_num]
                trigram = palace.trigram.chinese if palace.trigram else "中"
                row1_parts.append(f"[{palace_num}] {trigram} {palace.direction.value}".center(22))
            lines.append(" | ".join(row1_parts))

            # Second line: Earth/Heaven stems
            row2_parts = []
            for palace_num in row:
                palace = self.palaces[palace_num]
                e_stem = palace.earth_plate_stem or "-"
                h_stem = palace.heaven_plate_stem or "-"
                row2_parts.append(f"地:{e_stem} 天:{h_stem}".center(22))
            lines.append(" | ".join(row2_parts))

            # Third line: Star, Gate
            row3_parts = []
            for palace_num in row:
                palace = self.palaces[palace_num]
                star = palace.star.chinese if palace.star else "--"
                gate = palace.gate.chinese if palace.gate else "--"
                row3_parts.append(f"{star} {gate}".center(22))
            lines.append(" | ".join(row3_parts))

            # Fourth line: Spirit
            row4_parts = []
            for palace_num in row:
                palace = self.palaces[palace_num]
                spirit = palace.spirit.chinese if palace.spirit else "--"
                row4_parts.append(f"{spirit}".center(22))
            lines.append(" | ".join(row4_parts))

            lines.append("-" * 70)

        return "\n".join(lines)
