"""
Taiyi Shenshu Calculator (太乙神數計算器)

Core calculation engine implementing the classical Taiyi algorithms.

From 太乙金鏡式經:
"推上元積年，以周紀法三百六十去之；不盡，以元法七十二去之；
 又不盡，以太乙小周法二十四除之，又不盡，以三約之，為宮數"

This module implements:
- Taiyi position calculation (積太乙所在法)
- Tianmu (天目) calculation
- Jishen (計神) and Heshen (合神) calculation
- Host/Guest calculations (主算/客算)
- General positions (大將/參將)

碼道長存 — The Way of Code endures
"""

from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.lunar_calendar import ChineseLunarCalendar
from .constants import (
    TaiyiConstants,
    TaiyiDunType,
    TaiyiEra,
    PALACE_DEFINITIONS,
    get_opposite_palace,
)
from .spirits import SixteenSpirits, TaiyiSpirit
from .palaces import NinePalaces, TaiyiPalace, TaiyiPlate
from .generals import FiveGenerals
from .gates import EightGates


# =============================================================================
# Taiyi Calculator Class
# =============================================================================

class TaiyiCalculator:
    """
    Core calculation engine for Taiyi Shenshu.

    Implements the classical algorithms for:
    - Year calculation (歲計)
    - Month calculation (月計)
    - Day calculation (日計)
    - Hour calculation (時計)

    Usage:
        calculator = TaiyiCalculator()
        plate = calculator.calculate_year(datetime.now())
    """

    # Default epoch: 周厲王三十七年甲子 (as per 太乙金鏡式經)
    # This is approximately 841 BCE
    DEFAULT_EPOCH_OFFSET = 1561  # Years from epoch to Tang 開元12年 (724 CE)

    def __init__(self, lunar_calendar: Optional[ChineseLunarCalendar] = None):
        """
        Initialize the Taiyi calculator.

        Args:
            lunar_calendar: Optional lunar calendar instance. If not provided,
                           a new instance will be created.
        """
        self.lunar_calendar = lunar_calendar or ChineseLunarCalendar()
        self.spirits = SixteenSpirits()
        self.palaces = NinePalaces()
        self.five_generals = FiveGenerals()
        self.eight_gates = EightGates()

    # =========================================================================
    # Core Calculation Methods
    # =========================================================================

    def calculate_accumulated_years(self, dt: datetime,
                                     custom_epoch: Optional[int] = None) -> int:
        """
        Calculate accumulated years from the upper epoch (上元).

        For modern calculations, we use a simplified approach based on
        the sexagenary cycle position.

        Args:
            dt: The datetime to calculate for
            custom_epoch: Optional custom epoch year offset

        Returns:
            Number of accumulated years from epoch
        """
        # Get lunar year info
        lunar_date = self.lunar_calendar.gregorian_to_lunar(dt)

        # Use sexagenary cycle for calculation
        # The cycle repeats every 60 years, and we need to find
        # the position within the larger Taiyi cycles

        # For practical purposes, we calculate from a known reference point
        # 1984 was a 甲子 year (start of sexagenary cycle)
        reference_year = 1984
        years_from_reference = lunar_date.year - reference_year

        # Adjust to get accumulated years (positive value)
        # We add enough complete cycles to ensure positive result
        accumulated = years_from_reference + (360 * 100)  # Ensure positive

        return accumulated

    def calculate_taiyi_position(self, accumulated: int,
                                   dun_type: TaiyiDunType) -> Tuple[int, int]:
        """
        Calculate Taiyi's palace position.

        Implements the classical algorithm:
        "以周紀法三百六十去之；不盡，以元法七十二去之；
         又不盡，以太乙小周法二十四除之，又不盡，以三約之，為宮數"

        Args:
            accumulated: Accumulated time units (years/months/days)
            dun_type: Yang or Yin Dun

        Returns:
            Tuple of (palace_number, units_in_palace)
        """
        # Step 1: Remove complete grand cycles (360)
        remainder = accumulated % TaiyiConstants.ZHOU_JI_FA

        # Step 2: Remove complete eras (72)
        remainder = remainder % TaiyiConstants.YUAN_FA

        # Step 3: Remove complete small cycles (24)
        remainder = remainder % TaiyiConstants.TAIYI_XIAO_ZHOU

        # Step 4: Divide by 3 to get palace count
        palace_count = remainder // TaiyiConstants.PALACE_STAY
        units_in_palace = remainder % TaiyiConstants.PALACE_STAY

        # Step 5: Map to actual palace using traversal sequence
        palace_index = palace_count % TaiyiConstants.NUM_OUTER_PALACES
        palace = dun_type.palace_sequence[palace_index]

        return (palace, units_in_palace)

    def calculate_tianmu(self, accumulated: int) -> TaiyiSpirit:
        """
        Calculate which spirit serves as Tianmu (天目).

        Implements:
        "以天目周法十八去之，命起武德，順行十六神，遇陰德大武，重留一算"

        Args:
            accumulated: Accumulated time units

        Returns:
            The spirit serving as Tianmu
        """
        # Step 1: Remove complete era cycles
        remainder = accumulated % TaiyiConstants.YUAN_FA

        # Step 2: Remove complete Tianmu cycles (18)
        steps = remainder % TaiyiConstants.TIANMU_ZHOU_FA

        # Step 3: Traverse spirits from 武德 with 重留 rule
        return self.spirits.calculate_tianmu_spirit(steps)

    def calculate_jishen(self, era: TaiyiEra) -> str:
        """
        Calculate 計神 (Jishen) position.

        Based on the era, Jishen is typically at 寅.

        Args:
            era: The current era

        Returns:
            Earthly branch for Jishen
        """
        # Classical texts indicate 計神寅 for most eras
        return "寅"

    def calculate_heshen(self, era: TaiyiEra) -> str:
        """
        Calculate 合神 (Heshen) position.

        Based on the era, Heshen is typically at 丑.

        Args:
            era: The current era

        Returns:
            Earthly branch for Heshen
        """
        # Classical texts indicate 合神丑 for most eras
        return "丑"

    def determine_era(self, accumulated: int) -> Tuple[TaiyiEra, int]:
        """
        Determine which era and year within era.

        Args:
            accumulated: Accumulated years

        Returns:
            Tuple of (TaiyiEra, year_in_era)
        """
        # Get position within 60-year cycle
        cycle_position = (accumulated % 60) + 1  # 1-60

        # Determine era (each era is 10 years)
        era_index = ((cycle_position - 1) // 10) % 6
        year_in_era = ((cycle_position - 1) % 10) + 1  # 1-10

        era = list(TaiyiEra)[era_index]
        return (era, year_in_era)

    def determine_dun_type(self, dt: datetime) -> TaiyiDunType:
        """
        Determine Yang or Yin Dun based on solar position.

        Args:
            dt: The datetime to check

        Returns:
            TaiyiDunType (YANG or YIN)
        """
        # Calculate solar longitude
        jd = self.lunar_calendar.gregorian_to_julian(dt)
        solar_longitude = self.lunar_calendar.calculate_solar_longitude(jd)

        return TaiyiDunType.from_solar_longitude(solar_longitude)

    # =========================================================================
    # Host/Guest Calculation Methods (主算/客算)
    # =========================================================================

    def calculate_host_value(self, taiyi_palace: int, tianmu: TaiyiSpirit,
                              accumulated: int) -> int:
        """
        Calculate host value (主算).

        The host calculation is based on the positions of Taiyi, Tianmu,
        and other factors.

        Args:
            taiyi_palace: Palace where Taiyi is located
            tianmu: The Tianmu spirit
            accumulated: Accumulated time units

        Returns:
            Host calculation value
        """
        # Basic calculation based on palace and tianmu position
        base = taiyi_palace

        # Add Tianmu's traversal index
        tianmu_value = tianmu.traversal_index

        # Combine with modular arithmetic
        host_value = (base + tianmu_value + (accumulated % 10)) % 40

        # Ensure non-zero
        if host_value == 0:
            host_value = 40

        return host_value

    def calculate_guest_value(self, taiyi_palace: int, tianmu: TaiyiSpirit,
                               accumulated: int) -> int:
        """
        Calculate guest value (客算).

        Args:
            taiyi_palace: Palace where Taiyi is located
            tianmu: The Tianmu spirit
            accumulated: Accumulated time units

        Returns:
            Guest calculation value
        """
        # Guest calculation is typically based on opposite relationships
        opposite_palace = get_opposite_palace(taiyi_palace)

        base = opposite_palace
        tianmu_value = tianmu.traversal_index

        guest_value = (base + tianmu_value + (accumulated % 10) + 6) % 40

        if guest_value == 0:
            guest_value = 40

        return guest_value

    def is_calculation_harmonious(self, host: int, guest: int) -> bool:
        """
        Check if the calculations are harmonious (算和).

        Classical definition varies, but generally involves checking
        if the difference or sum meets certain criteria.

        Args:
            host: Host calculation value
            guest: Guest calculation value

        Returns:
            True if harmonious
        """
        # Check for harmony conditions
        difference = abs(host - guest)
        total = host + guest

        # Harmonious if difference is small or total is divisible by key numbers
        return (difference <= 5) or (total % 10 == 0) or (total % 12 == 0)

    # =========================================================================
    # General Position Methods (大將/參將)
    # =========================================================================

    def calculate_general_positions(self, host_calc: int, guest_calc: int,
                                      dun_type: TaiyiDunType) -> Dict[str, int]:
        """
        Calculate positions for all generals.

        Args:
            host_calc: Host calculation value
            guest_calc: Guest calculation value
            dun_type: Yang or Yin Dun

        Returns:
            Dictionary with general positions
        """
        sequence = dun_type.palace_sequence

        # Host grand general (主大將): based on host calculation
        host_general_idx = (host_calc % 8)
        host_general = sequence[host_general_idx]

        # Guest grand general (客大將): based on guest calculation
        guest_general_idx = (guest_calc % 8)
        guest_general = sequence[guest_general_idx]

        # Host deputy (主參將): 3× host calculation
        host_deputy_idx = ((host_calc * 3) % 8)
        host_deputy = sequence[host_deputy_idx]

        # Guest deputy (客參將): 3× guest calculation
        guest_deputy_idx = ((guest_calc * 3) % 8)
        guest_deputy = sequence[guest_deputy_idx]

        return {
            "host_general": host_general,
            "guest_general": guest_general,
            "host_deputy": host_deputy,
            "guest_deputy": guest_deputy,
        }

    # =========================================================================
    # Main Calculation Entry Points
    # =========================================================================

    def calculate_year(self, dt: datetime) -> TaiyiPlate:
        """
        Perform a complete Taiyi year calculation (歲計).

        This is the primary calculation method for annual predictions.

        Args:
            dt: The datetime to calculate for

        Returns:
            Complete TaiyiPlate with all calculations
        """
        # Reset palaces for new calculation
        self.palaces.reset_dynamic_attributes()

        # Step 1: Determine Dun type
        dun_type = self.determine_dun_type(dt)

        # Step 2: Calculate accumulated years
        accumulated = self.calculate_accumulated_years(dt)

        # Step 3: Determine era
        era, year_in_era = self.determine_era(accumulated)

        # Step 4: Calculate Taiyi position
        taiyi_palace, years_in_palace = self.calculate_taiyi_position(
            accumulated, dun_type
        )

        # Step 5: Calculate Tianmu
        tianmu = self.calculate_tianmu(accumulated)

        # Step 6: Calculate Jishen and Heshen
        jishen = self.calculate_jishen(era)
        heshen = self.calculate_heshen(era)

        # Step 7: Calculate host/guest values
        host_calc = self.calculate_host_value(taiyi_palace, tianmu, accumulated)
        guest_calc = self.calculate_guest_value(taiyi_palace, tianmu, accumulated)

        # Step 8: Check calculation harmony
        is_harmonious = self.is_calculation_harmonious(host_calc, guest_calc)

        # Step 9: Calculate all generals using FiveGenerals system
        all_generals = self.five_generals.calculate_all_generals(
            taiyi_palace=taiyi_palace,
            tianmu=tianmu,
            host_calculation=host_calc,
            guest_calculation=guest_calc,
            palace_sequence=dun_type.palace_sequence,
            accumulated=accumulated
        )

        # Extract individual general positions
        host_general_palace = all_generals["host_grand"].palace
        guest_general_palace = all_generals["guest_grand"].palace
        host_deputy_palace = all_generals["host_deputy"].palace
        guest_deputy_palace = all_generals["guest_deputy"].palace

        # Get eye positions
        shiji_spirit = all_generals["upper_eye"].spirit
        wenchang_palace = all_generals["lower_eye"].palace

        # Step 10: Analyze battle advantage
        battle_advantage = self.five_generals.analyze_battle_advantage(all_generals)
        special_formations = self.five_generals.check_special_formations(
            all_generals, taiyi_palace
        )

        # Step 11: Calculate Eight Gates
        gate_analysis = self.eight_gates.calculate_all_gates(
            accumulated=accumulated,
            taiyi_palace=taiyi_palace,
            host_general_palace=host_general_palace,
            guest_general_palace=guest_general_palace
        )
        ruling_gate = gate_analysis.get("ruling_gate", None)

        # Step 12: Update palace states
        self.palaces[taiyi_palace].taiyi = True
        self.palaces[taiyi_palace].tianmu = tianmu

        # Determine who is favored
        palace_info = self.palaces[taiyi_palace]
        favors_host = palace_info.favors_host
        favors_guest = palace_info.favors_guest

        # Get datetime info
        lunar_date = self.lunar_calendar.gregorian_to_lunar(dt)
        datetime_info = {
            "gregorian": dt.strftime("%Y-%m-%d %H:%M"),
            "lunar_year": lunar_date.year,
            "lunar_month": lunar_date.month,
            "lunar_day": lunar_date.day,
            "year_stem_branch": f"{lunar_date.year_stem}{lunar_date.year_branch}",
            "calculation_type": "歲計",
        }

        # Create immutable plate with full generals data
        return TaiyiPlate(
            datetime_info=datetime_info,
            dun_type=dun_type,
            era=era,
            era_year=year_in_era,
            taiyi_palace=taiyi_palace,
            years_in_palace=years_in_palace,
            tianmu_spirit=tianmu,
            jishen_branch=jishen,
            heshen_branch=heshen,
            host_calculation=host_calc,
            guest_calculation=guest_calc,
            host_general_palace=host_general_palace,
            guest_general_palace=guest_general_palace,
            host_deputy_palace=host_deputy_palace,
            guest_deputy_palace=guest_deputy_palace,
            shiji_spirit=shiji_spirit,
            wenchang_palace=wenchang_palace,
            palaces=dict(self.palaces.palaces),
            generals={k: v.format_summary() for k, v in all_generals.items()},
            is_calculation_harmonious=is_harmonious,
            favors_host=favors_host,
            favors_guest=favors_guest,
            battle_advantage=battle_advantage,
            special_formations=special_formations,
            ruling_gate=ruling_gate,
            gate_analysis=gate_analysis,
        )

    def _build_plate(self, dt: datetime, accumulated: int,
                      era: TaiyiEra, year_in_era: int,
                      dun_type: TaiyiDunType,
                      taiyi_palace: int, time_in_palace: int,
                      tianmu: TaiyiSpirit,
                      calculation_type: str) -> TaiyiPlate:
        """
        Helper method to build a complete TaiyiPlate.

        Reduces code duplication across year/month/day/hour calculations.
        """
        # Calculate Jishen and Heshen
        jishen = self.calculate_jishen(era)
        heshen = self.calculate_heshen(era)

        # Calculate host/guest values
        host_calc = self.calculate_host_value(taiyi_palace, tianmu, accumulated)
        guest_calc = self.calculate_guest_value(taiyi_palace, tianmu, accumulated)

        # Check calculation harmony
        is_harmonious = self.is_calculation_harmonious(host_calc, guest_calc)

        # Calculate all generals using FiveGenerals system
        all_generals = self.five_generals.calculate_all_generals(
            taiyi_palace=taiyi_palace,
            tianmu=tianmu,
            host_calculation=host_calc,
            guest_calculation=guest_calc,
            palace_sequence=dun_type.palace_sequence,
            accumulated=accumulated
        )

        # Extract positions
        host_general_palace = all_generals["host_grand"].palace
        guest_general_palace = all_generals["guest_grand"].palace
        host_deputy_palace = all_generals["host_deputy"].palace
        guest_deputy_palace = all_generals["guest_deputy"].palace
        shiji_spirit = all_generals["upper_eye"].spirit
        wenchang_palace = all_generals["lower_eye"].palace

        # Analyze battle advantage
        battle_advantage = self.five_generals.analyze_battle_advantage(all_generals)
        special_formations = self.five_generals.check_special_formations(
            all_generals, taiyi_palace
        )

        # Calculate Eight Gates
        gate_analysis = self.eight_gates.calculate_all_gates(
            accumulated=accumulated,
            taiyi_palace=taiyi_palace,
            host_general_palace=host_general_palace,
            guest_general_palace=guest_general_palace
        )
        ruling_gate = gate_analysis.get("ruling_gate", None)

        # Update palace states
        self.palaces.reset_dynamic_attributes()
        self.palaces[taiyi_palace].taiyi = True
        self.palaces[taiyi_palace].tianmu = tianmu

        # Determine who is favored
        palace_info = self.palaces[taiyi_palace]
        favors_host = palace_info.favors_host
        favors_guest = palace_info.favors_guest

        # Build datetime info
        lunar_date = self.lunar_calendar.gregorian_to_lunar(dt)
        datetime_info = {
            "gregorian": dt.strftime("%Y-%m-%d %H:%M"),
            "lunar_year": lunar_date.year,
            "lunar_month": lunar_date.month,
            "lunar_day": lunar_date.day,
            "year_stem_branch": f"{lunar_date.year_stem}{lunar_date.year_branch}",
            "calculation_type": calculation_type,
        }

        return TaiyiPlate(
            datetime_info=datetime_info,
            dun_type=dun_type,
            era=era,
            era_year=year_in_era,
            taiyi_palace=taiyi_palace,
            years_in_palace=time_in_palace,
            tianmu_spirit=tianmu,
            jishen_branch=jishen,
            heshen_branch=heshen,
            host_calculation=host_calc,
            guest_calculation=guest_calc,
            host_general_palace=host_general_palace,
            guest_general_palace=guest_general_palace,
            host_deputy_palace=host_deputy_palace,
            guest_deputy_palace=guest_deputy_palace,
            shiji_spirit=shiji_spirit,
            wenchang_palace=wenchang_palace,
            palaces=dict(self.palaces.palaces),
            generals={k: v.format_summary() for k, v in all_generals.items()},
            is_calculation_harmonious=is_harmonious,
            favors_host=favors_host,
            favors_guest=favors_guest,
            battle_advantage=battle_advantage,
            special_formations=special_formations,
            ruling_gate=ruling_gate,
            gate_analysis=gate_analysis,
        )

    def calculate_month(self, dt: datetime) -> TaiyiPlate:
        """
        Perform a Taiyi month calculation (月計).

        Similar to year calculation but uses accumulated months.

        Args:
            dt: The datetime to calculate for

        Returns:
            Complete TaiyiPlate
        """
        # Calculate accumulated values
        accumulated_years = self.calculate_accumulated_years(dt)
        lunar_date = self.lunar_calendar.gregorian_to_lunar(dt)
        accumulated = accumulated_years * 12 + lunar_date.month

        # Determine dun type and era
        dun_type = self.determine_dun_type(dt)
        era, year_in_era = self.determine_era(accumulated_years)

        # Calculate positions
        taiyi_palace, time_in_palace = self.calculate_taiyi_position(accumulated, dun_type)
        tianmu = self.calculate_tianmu(accumulated)

        return self._build_plate(
            dt, accumulated, era, year_in_era, dun_type,
            taiyi_palace, time_in_palace, tianmu, "月計"
        )

    def calculate_day(self, dt: datetime) -> TaiyiPlate:
        """
        Perform a Taiyi day calculation (日計).

        Args:
            dt: The datetime to calculate for

        Returns:
            Complete TaiyiPlate
        """
        # Calculate Julian day for day accumulation
        jd = self.lunar_calendar.gregorian_to_julian(dt)
        accumulated = int(jd) % (TaiyiConstants.ZHOU_JI_FA * 100)

        # Determine dun type and era
        dun_type = self.determine_dun_type(dt)
        accumulated_years = self.calculate_accumulated_years(dt)
        era, year_in_era = self.determine_era(accumulated_years)

        # Calculate positions
        taiyi_palace, time_in_palace = self.calculate_taiyi_position(accumulated, dun_type)
        tianmu = self.calculate_tianmu(accumulated)

        return self._build_plate(
            dt, accumulated, era, year_in_era, dun_type,
            taiyi_palace, time_in_palace, tianmu, "日計"
        )

    def calculate_hour(self, dt: datetime) -> TaiyiPlate:
        """
        Perform a Taiyi hour calculation (時計).

        Uses the most granular level of calculation.

        Args:
            dt: The datetime to calculate for

        Returns:
            Complete TaiyiPlate
        """
        # Calculate accumulated hours (2-hour periods in traditional time)
        jd = self.lunar_calendar.gregorian_to_julian(dt)

        # Convert to traditional double-hours (12 per day)
        accumulated_days = int(jd)
        hour_in_day = (dt.hour + 1) // 2  # Convert to 12-hour system
        accumulated = accumulated_days * 12 + hour_in_day

        # Determine dun type and era
        dun_type = self.determine_dun_type(dt)
        accumulated_years = self.calculate_accumulated_years(dt)
        era, year_in_era = self.determine_era(accumulated_years)

        # Calculate positions
        taiyi_palace, time_in_palace = self.calculate_taiyi_position(accumulated, dun_type)
        tianmu = self.calculate_tianmu(accumulated)

        return self._build_plate(
            dt, accumulated, era, year_in_era, dun_type,
            taiyi_palace, time_in_palace, tianmu, "時計"
        )

    def calculate(self, dt: datetime,
                   calculation_type: str = "year") -> TaiyiPlate:
        """
        Universal calculation method.

        Args:
            dt: The datetime to calculate for
            calculation_type: One of "year", "month", "day", "hour"

        Returns:
            Complete TaiyiPlate
        """
        methods = {
            "year": self.calculate_year,
            "month": self.calculate_month,
            "day": self.calculate_day,
            "hour": self.calculate_hour,
        }

        if calculation_type not in methods:
            raise ValueError(
                f"Invalid calculation type: {calculation_type}. "
                f"Must be one of: {list(methods.keys())}"
            )

        return methods[calculation_type](dt)
