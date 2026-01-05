"""
Qimen Dunjia Calculator Engine (奇門遁甲計算引擎)

This module contains the core calculation engine for generating
Qimen Dunjia plates from datetime inputs.

The calculator implements the 按時 (Hourly) method, using the hour pillar
to determine duty star and gate positions.

Classical references:
    - 黄帝太一八门逆顺生死诀
    - 黄帝太一八门入式诀

九天玄碼女在此 - 碼道長存
"""

import sys
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.core import DunType, Yuan, SolarTerm, Trigram, Element, Direction
from .constants import (
    EARTH_PLATE_SEQUENCE,
    YANG_DUN_SEQUENCE,
    YIN_DUN_SEQUENCE,
    SOLAR_TERM_JU_MAPPING,
    JIA_CONCEALMENT,
    JIA_TO_YI,
    SIX_YI,
    THREE_WONDERS,
    get_sequence_for_dun,
    get_jia_for_day_index,
    get_yuan_for_day_index,
)
from .components import NineStars, EightGates, EightSpirits, NineStar, EightGate, EightSpirit
from .palaces import Palace, NinePalaces, QimenPlate


class QimenCalculator:
    """
    Core calculation engine for Qimen Dunjia.

    Implements the complete plate generation algorithm including:
    - Dun type determination (Yang/Yin)
    - Solar term and Yuan identification
    - Ju number calculation
    - Earth and Heaven plate stem placement
    - Star, Gate, and Spirit rotation

    Configuration options:
        center_palace_follows: Which palace the center (5) inherits from
            2 = Kun tradition (天禽寄坤) - default
            8 = Gen tradition (天禽寄艮)
    """

    def __init__(self, lunar_calendar=None, center_palace_follows: int = 2):
        """
        Initialize the Qimen calculator.

        Args:
            lunar_calendar: ChineseLunarCalendar instance (will create one if None)
            center_palace_follows: Palace 5 inherits star/gate from this palace
        """
        # Lazy import to avoid circular dependencies
        if lunar_calendar is None:
            from lunar_calendar import ChineseLunarCalendar
            lunar_calendar = ChineseLunarCalendar()

        self.lunar_calendar = lunar_calendar
        self.center_palace_follows = center_palace_follows

        # Initialize component containers
        self.nine_stars = NineStars()
        self.eight_gates = EightGates()
        self.eight_spirits = EightSpirits()

        # Ten Heavenly Stems sequence
        self.stems = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']

    # =========================================================================
    # Dun Type and Solar Term Determination
    # =========================================================================

    def determine_dun_type(self, solar_longitude: float) -> DunType:
        """
        Determine Yang Dun or Yin Dun based on solar longitude.

        Yang Dun (陽遁): Winter Solstice to Summer Solstice (270° to 90°)
        Yin Dun (陰遁): Summer Solstice to Winter Solstice (90° to 270°)

        Args:
            solar_longitude: Sun's ecliptic longitude in degrees (0-360)

        Returns:
            DunType.YANG or DunType.YIN
        """
        # Yang Dun: 270° <= longitude < 360° OR 0° <= longitude < 90°
        # Yin Dun: 90° <= longitude < 270°
        if solar_longitude >= 270 or solar_longitude < 90:
            return DunType.YANG
        return DunType.YIN

    def get_current_solar_term(self, solar_longitude: float) -> SolarTerm:
        """
        Determine the current solar term from solar longitude.

        Each solar term spans 15 degrees of solar longitude.

        Args:
            solar_longitude: Sun's ecliptic longitude in degrees

        Returns:
            The current SolarTerm
        """
        # Solar terms are every 15 degrees, starting from Spring Equinox at 0°
        # We need to map longitude to the correct solar term

        # Create mapping from longitude ranges to solar terms
        term_order = [
            (0, SolarTerm.CHUN_FEN),       # Spring Equinox
            (15, SolarTerm.QING_MING),     # Clear and Bright
            (30, SolarTerm.GU_YU),         # Grain Rain
            (45, SolarTerm.LI_XIA),        # Start of Summer
            (60, SolarTerm.XIAO_MAN),      # Grain Buds
            (75, SolarTerm.MANG_ZHONG),    # Grain in Ear
            (90, SolarTerm.XIA_ZHI),       # Summer Solstice
            (105, SolarTerm.XIAO_SHU),     # Minor Heat
            (120, SolarTerm.DA_SHU),       # Major Heat
            (135, SolarTerm.LI_QIU),       # Start of Autumn
            (150, SolarTerm.CHU_SHU),      # End of Heat
            (165, SolarTerm.BAI_LU),       # White Dew
            (180, SolarTerm.QIU_FEN),      # Autumn Equinox
            (195, SolarTerm.HAN_LU),       # Cold Dew
            (210, SolarTerm.SHUANG_JIANG), # Frost Descent
            (225, SolarTerm.LI_DONG),      # Start of Winter
            (240, SolarTerm.XIAO_XUE),     # Minor Snow
            (255, SolarTerm.DA_XUE),       # Major Snow
            (270, SolarTerm.DONG_ZHI),     # Winter Solstice
            (285, SolarTerm.XIAO_HAN),     # Minor Cold
            (300, SolarTerm.DA_HAN),       # Major Cold
            (315, SolarTerm.LI_CHUN),      # Start of Spring
            (330, SolarTerm.YU_SHUI),      # Rain Water
            (345, SolarTerm.JING_ZHE),     # Awakening of Insects
        ]

        # Find the solar term for the current longitude
        for i, (start_long, term) in enumerate(term_order):
            next_long = term_order[(i + 1) % 24][0]
            if next_long == 0:
                next_long = 360

            if start_long <= solar_longitude < next_long:
                return term
            # Handle wrap-around at 360°
            if start_long > next_long and (solar_longitude >= start_long or solar_longitude < next_long):
                return term

        # Default fallback (should not reach here)
        return SolarTerm.DONG_ZHI

    # =========================================================================
    # Yuan and Ju Determination
    # =========================================================================

    def get_day_cycle_index(self, julian_day: float) -> int:
        """
        Get the position in the 60-day sexagenary cycle.

        Args:
            julian_day: Julian day number

        Returns:
            Day index (1-60) in the sexagenary cycle
        """
        # Reference: January 1, 1900 was day 11 (甲戌) in the 60-day cycle
        ref_jd = self.lunar_calendar.gregorian_to_julian(datetime(1900, 1, 1))
        days_since_ref = int(julian_day - ref_jd)
        day_index = ((11 + days_since_ref - 1) % 60) + 1
        return day_index

    def determine_yuan(self, day_cycle_index: int) -> Yuan:
        """
        Determine the Yuan (Upper/Middle/Lower) from day cycle position.

        Uses the classical formula based on which Jia period the day falls in.

        Args:
            day_cycle_index: Position in the 60-day cycle (1-60)

        Returns:
            Yuan.UPPER, Yuan.MIDDLE, or Yuan.LOWER
        """
        return get_yuan_for_day_index(day_cycle_index)

    def get_ju_number(self, solar_term: SolarTerm, yuan: Yuan) -> int:
        """
        Determine the Ju (局) number from solar term and yuan.

        The Ju number determines where the Earth plate sequence begins.

        Args:
            solar_term: Current solar term
            yuan: Current yuan period

        Returns:
            Ju number (1-9)
        """
        term_mapping = SOLAR_TERM_JU_MAPPING.get(solar_term)
        if term_mapping:
            return term_mapping.get(yuan, 1)
        return 1  # Default

    # =========================================================================
    # Plate Construction
    # =========================================================================

    def build_earth_plate(self, ju_number: int, dun_type: DunType) -> Dict[int, str]:
        """
        Build the Earth Plate (地盤) stem positions.

        The Earth Plate is the static foundation. Stems are placed starting
        from the Ju palace, following the sequence 戊己庚辛壬癸丁丙乙.

        Args:
            ju_number: Starting palace number (1-9)
            dun_type: Yang or Yin dun (determines rotation direction)

        Returns:
            Dict mapping palace number to stem character
        """
        sequence = get_sequence_for_dun(dun_type)
        earth_plate = {}

        # Find starting position in the sequence
        if ju_number == 5:
            # Center palace - use the palace it follows
            start_palace = self.center_palace_follows
        else:
            start_palace = ju_number

        try:
            start_idx = sequence.index(start_palace)
        except ValueError:
            start_idx = 0

        # Place stems following the sequence
        for i, stem in enumerate(EARTH_PLATE_SEQUENCE):
            palace_idx = (start_idx + i) % len(sequence)
            palace_num = sequence[palace_idx]
            earth_plate[palace_num] = stem

        # Handle center palace (5) - inherits from configured palace
        if 5 not in earth_plate:
            earth_plate[5] = earth_plate.get(self.center_palace_follows, '戊')

        return earth_plate

    def find_hour_stem_palace(self, hour_stem: str, earth_plate: Dict[int, str]) -> int:
        """
        Find which palace contains the hour's stem on the Earth Plate.

        For 按時 method, the hour stem determines the duty palace.

        Args:
            hour_stem: The hour pillar's heavenly stem
            earth_plate: The Earth Plate stem mapping

        Returns:
            Palace number where the hour stem resides
        """
        # If hour stem is 甲, it hides under one of the Six Yi
        if hour_stem == '甲':
            # Need to determine which Jia based on the hour's position
            # For simplicity, use 戊 (甲子 hides under 戊)
            hour_stem = '戊'

        for palace_num, stem in earth_plate.items():
            if stem == hour_stem:
                return palace_num

        # Default to palace 1 if not found
        return 1

    def get_duty_star_and_gate(self, duty_palace: int) -> Tuple[Optional[NineStar], Optional[EightGate]]:
        """
        Get the duty star (值符) and duty gate (值使) for a palace.

        The duty star is the star that originally resides in the duty palace.
        The duty gate is the gate that originally resides in the duty palace.

        Args:
            duty_palace: The palace number where duty falls

        Returns:
            Tuple of (duty_star, duty_gate)
        """
        duty_star = self.nine_stars.get_by_palace(duty_palace)
        duty_gate = self.eight_gates.get_by_palace(duty_palace)
        return duty_star, duty_gate

    def rotate_heaven_plate(self, earth_plate: Dict[int, str],
                            duty_palace: int,
                            dun_type: DunType) -> Dict[int, str]:
        """
        Build the Heaven Plate (天盤) by rotating stems.

        The Heaven Plate stems rotate based on the duty palace position.

        Args:
            earth_plate: The Earth Plate stem mapping
            duty_palace: The duty palace number
            dun_type: Yang or Yin dun

        Returns:
            Dict mapping palace number to rotated stem
        """
        sequence = get_sequence_for_dun(dun_type)
        heaven_plate = {}

        # The duty star (at duty_palace) moves according to the hour
        # Calculate rotation offset
        if duty_palace == 5:
            duty_palace = self.center_palace_follows

        try:
            duty_idx = sequence.index(duty_palace)
        except ValueError:
            duty_idx = 0

        # Rotate all stems
        for palace_num, stem in earth_plate.items():
            if palace_num == 5:
                continue  # Handle center separately

            try:
                orig_idx = sequence.index(palace_num)
                # Calculate new position after rotation
                offset = duty_idx - sequence.index(1)  # Offset from palace 1
                new_idx = (orig_idx + offset) % len(sequence)
                new_palace = sequence[new_idx]
                heaven_plate[new_palace] = stem
            except ValueError:
                heaven_plate[palace_num] = stem

        # Center palace inherits
        heaven_plate[5] = heaven_plate.get(self.center_palace_follows, earth_plate.get(5, '戊'))

        # If rotation produced incomplete mapping, use earth plate
        if len(heaven_plate) < 9:
            for p in range(1, 10):
                if p not in heaven_plate:
                    heaven_plate[p] = earth_plate.get(p, '戊')

        return heaven_plate

    def position_stars(self, duty_palace: int, dun_type: DunType) -> Dict[int, NineStar]:
        """
        Calculate star positions by rotating from base positions.

        Stars rotate following the palace sequence based on where
        the duty star lands.

        Args:
            duty_palace: Palace where duty star lands
            dun_type: Yang or Yin dun

        Returns:
            Dict mapping palace number to star
        """
        sequence = get_sequence_for_dun(dun_type)
        star_positions = {}

        # Get base star positions
        base_stars = {}
        for star in self.nine_stars.stars:
            base_stars[star.base_palace] = star

        # Calculate rotation based on duty palace
        if duty_palace == 5:
            duty_palace = self.center_palace_follows

        try:
            duty_idx = sequence.index(duty_palace)
        except ValueError:
            duty_idx = 0

        # Rotate stars
        for orig_palace, star in base_stars.items():
            if orig_palace == 5:
                continue  # Tianqin (天禽) handled separately

            try:
                orig_idx = sequence.index(orig_palace)
                new_idx = (orig_idx + duty_idx) % len(sequence)
                new_palace = sequence[new_idx]
                star_positions[new_palace] = star
            except ValueError:
                star_positions[orig_palace] = star

        # Handle Tianqin (天禽) - follows the palace it's assigned to
        tianqin = self.nine_stars.by_chinese.get('天禽')
        if tianqin:
            star_positions[5] = tianqin
            # Also place at the inheriting palace if not occupied
            if self.center_palace_follows not in star_positions:
                star_positions[self.center_palace_follows] = tianqin

        return star_positions

    def position_gates(self, duty_palace: int, dun_type: DunType) -> Dict[int, EightGate]:
        """
        Calculate gate positions by rotating from base positions.

        Gates rotate similar to stars, following the duty gate's movement.

        Args:
            duty_palace: Palace where duty gate lands
            dun_type: Yang or Yin dun

        Returns:
            Dict mapping palace number to gate
        """
        sequence = get_sequence_for_dun(dun_type)
        gate_positions = {}

        # Get base gate positions
        base_gates = {}
        for gate in self.eight_gates.gates:
            base_gates[gate.base_palace] = gate

        # Calculate rotation
        if duty_palace == 5:
            duty_palace = self.center_palace_follows

        try:
            duty_idx = sequence.index(duty_palace)
        except ValueError:
            duty_idx = 0

        # Rotate gates
        for orig_palace, gate in base_gates.items():
            try:
                orig_idx = sequence.index(orig_palace)
                new_idx = (orig_idx + duty_idx) % len(sequence)
                new_palace = sequence[new_idx]
                gate_positions[new_palace] = gate
            except ValueError:
                gate_positions[orig_palace] = gate

        # Center palace (5) has no gate, but inherit if needed
        if 5 not in gate_positions and self.center_palace_follows in gate_positions:
            gate_positions[5] = gate_positions[self.center_palace_follows]

        return gate_positions

    def position_spirits(self, duty_palace: int, dun_type: DunType) -> Dict[int, EightSpirit]:
        """
        Calculate spirit positions.

        Spirits follow a specific sequence starting from the duty palace.
        In Yang Dun, they proceed forward; in Yin Dun, backward.

        Args:
            duty_palace: Starting palace for 值符 spirit
            dun_type: Yang or Yin dun

        Returns:
            Dict mapping palace number to spirit
        """
        sequence = get_sequence_for_dun(dun_type)
        spirit_positions = {}

        # Spirit sequence: 值符、螣蛇、太阴、六合、白虎、玄武、九地、九天
        spirit_order = [
            self.eight_spirits.by_chinese.get('值符'),
            self.eight_spirits.by_chinese.get('螣蛇'),
            self.eight_spirits.by_chinese.get('太阴'),
            self.eight_spirits.by_chinese.get('六合'),
            self.eight_spirits.by_chinese.get('白虎'),
            self.eight_spirits.by_chinese.get('玄武'),
            self.eight_spirits.by_chinese.get('九地'),
            self.eight_spirits.by_chinese.get('九天'),
        ]

        # Start from duty palace
        if duty_palace == 5:
            start_palace = self.center_palace_follows
        else:
            start_palace = duty_palace

        try:
            start_idx = sequence.index(start_palace)
        except ValueError:
            start_idx = 0

        # Place spirits following sequence
        for i, spirit in enumerate(spirit_order):
            if spirit:
                palace_idx = (start_idx + i) % len(sequence)
                palace_num = sequence[palace_idx]
                spirit_positions[palace_num] = spirit

        # Center palace spirit (if applicable)
        if 5 not in spirit_positions and self.center_palace_follows in spirit_positions:
            spirit_positions[5] = spirit_positions[self.center_palace_follows]

        return spirit_positions

    # =========================================================================
    # Main Calculation Method
    # =========================================================================

    def calculate_plate(self, dt: datetime) -> QimenPlate:
        """
        Calculate the complete Qimen Dunjia plate for a given datetime.

        This is the main entry point for plate calculation.

        Args:
            dt: The datetime to calculate for

        Returns:
            QimenPlate containing all plate information
        """
        # Step 1: Get lunar calendar information
        julian_day = self.lunar_calendar.gregorian_to_julian(dt)
        lunar_date = self.lunar_calendar.gregorian_to_lunar(dt)

        # Step 2: Calculate solar longitude and determine dun type
        solar_longitude = self.lunar_calendar.calculate_solar_longitude(julian_day)
        dun_type = self.determine_dun_type(solar_longitude)

        # Step 3: Determine solar term
        solar_term = self.get_current_solar_term(solar_longitude)

        # Step 4: Calculate day cycle and yuan
        day_cycle_index = self.get_day_cycle_index(julian_day)
        yuan = self.determine_yuan(day_cycle_index)

        # Step 5: Get Ju number
        ju_number = self.get_ju_number(solar_term, yuan)

        # Step 6: Build Earth Plate
        earth_plate = self.build_earth_plate(ju_number, dun_type)

        # Step 7: Find duty palace from hour stem
        hour_stem = lunar_date.hour_stem.chinese
        duty_palace = self.find_hour_stem_palace(hour_stem, earth_plate)

        # Step 8: Get duty star and gate
        duty_star, duty_gate = self.get_duty_star_and_gate(duty_palace)

        # Step 9: Build Heaven Plate
        heaven_plate = self.rotate_heaven_plate(earth_plate, duty_palace, dun_type)

        # Step 10: Position stars, gates, spirits
        star_positions = self.position_stars(duty_palace, dun_type)
        gate_positions = self.position_gates(duty_palace, dun_type)
        spirit_positions = self.position_spirits(duty_palace, dun_type)

        # Step 11: Assemble palaces
        palaces = self._assemble_palaces(
            earth_plate, heaven_plate,
            star_positions, gate_positions, spirit_positions
        )

        # Step 12: Create and return QimenPlate
        return QimenPlate(
            datetime_info=lunar_date,
            dun_type=dun_type,
            yuan=yuan,
            ju_number=ju_number,
            solar_term=solar_term,
            duty_star=duty_star,
            duty_gate=duty_gate,
            duty_palace=duty_palace,
            palaces=palaces,
            earth_plate=earth_plate,
            heaven_plate=heaven_plate,
            star_positions=star_positions,
            gate_positions=gate_positions,
            spirit_positions=spirit_positions,
        )

    def _assemble_palaces(self,
                          earth_plate: Dict[int, str],
                          heaven_plate: Dict[int, str],
                          star_positions: Dict[int, NineStar],
                          gate_positions: Dict[int, EightGate],
                          spirit_positions: Dict[int, EightSpirit]) -> Dict[int, Palace]:
        """
        Assemble complete Palace objects with all dynamic content.

        Args:
            earth_plate: Earth plate stem mapping
            heaven_plate: Heaven plate stem mapping
            star_positions: Star positions
            gate_positions: Gate positions
            spirit_positions: Spirit positions

        Returns:
            Dict mapping palace number to fully populated Palace
        """
        nine_palaces = NinePalaces()
        palaces = {}

        for num in range(1, 10):
            palace = nine_palaces.get_palace(num)
            if palace:
                # Create a new palace with dynamic content
                palaces[num] = Palace(
                    number=palace.number,
                    trigram=palace.trigram,
                    direction=palace.direction,
                    base_element=palace.base_element,
                    earth_plate_stem=earth_plate.get(num),
                    heaven_plate_stem=heaven_plate.get(num),
                    star=star_positions.get(num),
                    gate=gate_positions.get(num),
                    spirit=spirit_positions.get(num),
                )

        return palaces


# =============================================================================
# Convenience Functions
# =============================================================================

def calculate_qimen_plate(dt: datetime,
                          lunar_calendar=None,
                          center_palace_follows: int = 2) -> QimenPlate:
    """
    Convenience function to calculate a Qimen plate.

    Args:
        dt: Datetime to calculate for
        lunar_calendar: Optional ChineseLunarCalendar instance
        center_palace_follows: Palace 5 inheritance (2 or 8)

    Returns:
        QimenPlate for the given datetime
    """
    calculator = QimenCalculator(
        lunar_calendar=lunar_calendar,
        center_palace_follows=center_palace_follows
    )
    return calculator.calculate_plate(dt)
