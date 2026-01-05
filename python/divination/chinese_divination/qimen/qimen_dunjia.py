"""
Qimen Dunjia (奇門遁甲) Main API

This module provides the primary interface for Qimen Dunjia divination,
one of the Three Styles (三式) of Chinese strategic forecasting.

Usage:
    from prognostication.qimen import QimenDunjia

    qimen = QimenDunjia()
    plate = qimen.calculate(datetime.now())
    print(plate.format_display())

    # Get directional advice
    advice = qimen.get_directional_advice(plate)

    # Perform specific query
    result = qimen.query(datetime.now(), question_type="business")

Classical references:
    - 黄帝太一八门逆顺生死诀
    - 黄帝太一八门入式诀
    - 黄帝太一八门入式秘诀

九天玄碼女在此 - 碼道長存
"""

import sys
import os
from datetime import datetime
from typing import Dict, Optional, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.core import DunType, Yuan, SolarTerm, Direction, Element
from .calculator import QimenCalculator
from .palaces import QimenPlate, Palace
from .components import NineStar, EightGate, EightSpirit


class QimenDunjia:
    """
    Main Qimen Dunjia divination system interface.

    This class provides a high-level API for calculating and interpreting
    Qimen Dunjia plates. It combines the calculator engine with analysis
    methods to provide actionable divination results.

    Attributes:
        lunar_calendar: The lunar calendar instance used for calculations
        calculator: The plate calculation engine
        center_palace_follows: Configuration for center palace inheritance

    Example:
        >>> qimen = QimenDunjia()
        >>> plate = qimen.calculate(datetime(2024, 1, 15, 10, 30))
        >>> print(qimen.get_summary(plate))
    """

    def __init__(self, lunar_calendar=None, center_palace_follows: int = 2):
        """
        Initialize the Qimen Dunjia system.

        Args:
            lunar_calendar: ChineseLunarCalendar instance (creates one if None)
            center_palace_follows: Which palace the center (5) inherits from
                2 = Kun tradition (天禽寄坤) - default, more common
                8 = Gen tradition (天禽寄艮) - alternative tradition
        """
        # Lazy import to avoid circular dependencies
        if lunar_calendar is None:
            from ..core.lunar_calendar import ChineseLunarCalendar
            lunar_calendar = ChineseLunarCalendar()

        self.lunar_calendar = lunar_calendar
        self.center_palace_follows = center_palace_follows
        self.calculator = QimenCalculator(
            lunar_calendar=lunar_calendar,
            center_palace_follows=center_palace_follows
        )

        # Lazy load analyzer
        self._analyzer = None

    @property
    def analyzer(self):
        """Lazy load the analyzer module."""
        if self._analyzer is None:
            from .analysis import QimenAnalyzer
            self._analyzer = QimenAnalyzer()
        return self._analyzer

    # =========================================================================
    # Core Calculation Methods
    # =========================================================================

    def calculate(self, dt: datetime) -> QimenPlate:
        """
        Calculate the complete Qimen Dunjia plate for a specific datetime.

        This method generates all five plates (Earth, Heaven, Human/Gates,
        Spirit, and directional analysis) and returns a complete QimenPlate.

        Args:
            dt: The datetime to calculate for

        Returns:
            QimenPlate containing all plate information

        Example:
            >>> plate = qimen.calculate(datetime.now())
            >>> print(f"局: {plate.ju_number}, 遁: {plate.dun_type.chinese}")
        """
        return self.calculator.calculate_plate(dt)

    def calculate_for_now(self) -> QimenPlate:
        """
        Calculate the Qimen plate for the current moment.

        Convenience method equivalent to calculate(datetime.now()).

        Returns:
            QimenPlate for the current time
        """
        return self.calculate(datetime.now())

    # =========================================================================
    # Display and Formatting
    # =========================================================================

    def format_plate_display(self, plate: QimenPlate) -> str:
        """
        Format the plate as a visual grid display.

        Creates a text representation of the Nine Palaces grid with all
        plate elements (stems, stars, gates, spirits) displayed.

        Args:
            plate: The QimenPlate to display

        Returns:
            Formatted string representation of the plate
        """
        return plate.format_display()

    def get_summary(self, plate: QimenPlate) -> Dict[str, Any]:
        """
        Get a summary dictionary of the plate configuration.

        Args:
            plate: The QimenPlate to summarize

        Returns:
            Dictionary with key plate information
        """
        return plate.get_summary()

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def analyze(self, plate: QimenPlate) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a Qimen plate.

        Analyzes element relationships, auspicious factors, warnings,
        and overall configuration quality.

        Args:
            plate: The QimenPlate to analyze

        Returns:
            Dictionary containing analysis results
        """
        return self.analyzer.analyze_plate(plate)

    def get_directional_advice(self, plate: QimenPlate) -> Dict[str, Dict[str, Any]]:
        """
        Get advice for each of the eight directions plus center.

        Analyzes each palace/direction for favorability based on the
        current gate, star, and stem configurations.

        Args:
            plate: The QimenPlate to analyze

        Returns:
            Dictionary mapping direction names to advice dicts
        """
        return self.analyzer.analyze_directions(plate)

    def find_auspicious_directions(self, plate: QimenPlate) -> List[Dict[str, Any]]:
        """
        Find the most auspicious directions in the current plate.

        Identifies palaces with favorable star-gate combinations,
        presence of Three Wonders, and positive element relationships.

        Args:
            plate: The QimenPlate to analyze

        Returns:
            List of auspicious directions with details, sorted by favorability
        """
        return self.analyzer.find_auspicious_directions(plate)

    def find_inauspicious_directions(self, plate: QimenPlate) -> List[Dict[str, Any]]:
        """
        Find directions to avoid in the current plate.

        Identifies palaces with unfavorable configurations such as
        Si Gate (死门), inauspicious stars, or element clashes.

        Args:
            plate: The QimenPlate to analyze

        Returns:
            List of inauspicious directions with warnings
        """
        return self.analyzer.find_inauspicious_directions(plate)

    # =========================================================================
    # Query Methods
    # =========================================================================

    def query(self, dt: datetime, question_type: str = "general") -> Dict[str, Any]:
        """
        Perform a divination query for a specific question type.

        Calculates the plate and provides tailored analysis based on
        the type of question being asked.

        Args:
            dt: The datetime for the query
            question_type: Type of question. Options:
                - "general": Overall reading
                - "business": Business and financial matters
                - "travel": Travel and movement
                - "health": Health concerns
                - "legal": Legal matters and disputes
                - "relationship": Relationship matters
                - "career": Career decisions

        Returns:
            Dictionary with plate information and question-specific analysis
        """
        plate = self.calculate(dt)
        analysis = self.analyzer.query_analysis(plate, question_type)

        return {
            "plate": plate,
            "datetime": dt.isoformat(),
            "question_type": question_type,
            "dun_type": plate.dun_type.chinese,
            "yuan": plate.yuan.chinese,
            "ju_number": plate.ju_number,
            "solar_term": plate.solar_term.chinese,
            "duty_star": plate.duty_star.chinese if plate.duty_star else None,
            "duty_gate": plate.duty_gate.chinese if plate.duty_gate else None,
            "analysis": analysis,
        }

    # =========================================================================
    # Palace Lookup Methods
    # =========================================================================

    def get_palace(self, plate: QimenPlate, number: int) -> Optional[Palace]:
        """
        Get a specific palace from the plate by number.

        Args:
            plate: The QimenPlate to query
            number: Palace number (1-9)

        Returns:
            Palace object or None if not found
        """
        return plate.get_palace(number)

    def get_palace_by_direction(self, plate: QimenPlate, direction: Direction) -> Optional[Palace]:
        """
        Get a palace by its direction.

        Args:
            plate: The QimenPlate to query
            direction: Direction enum value

        Returns:
            Palace at that direction or None
        """
        return plate.get_palace_by_direction(direction)

    def find_stem(self, plate: QimenPlate, stem: str, plate_type: str = "heaven") -> Optional[int]:
        """
        Find which palace contains a specific stem.

        Args:
            plate: The QimenPlate to search
            stem: The stem character to find
            plate_type: "heaven" or "earth"

        Returns:
            Palace number or None if not found
        """
        return plate.find_stem_palace(stem, plate_type)

    def find_star(self, plate: QimenPlate, star_chinese: str) -> Optional[int]:
        """
        Find which palace contains a specific star.

        Args:
            plate: The QimenPlate to search
            star_chinese: Star name in Chinese (e.g., '天蓬')

        Returns:
            Palace number or None
        """
        return plate.find_star_palace(star_chinese)

    def find_gate(self, plate: QimenPlate, gate_chinese: str) -> Optional[int]:
        """
        Find which palace contains a specific gate.

        Args:
            plate: The QimenPlate to search
            gate_chinese: Gate name in Chinese (e.g., '開門', '休門')

        Returns:
            Palace number or None
        """
        return plate.find_gate_palace(gate_chinese)

    # =========================================================================
    # Special Condition Checks
    # =========================================================================

    def check_special_conditions(self, plate: QimenPlate) -> Dict[str, Any]:
        """
        Check for special conditions in the plate.

        Identifies special formations such as:
        - 伏吟 (Fu Yin): Same stem in heaven and earth
        - 反吟 (Fan Yin): Opposing stems
        - 三奇得使 (Three Wonders with Envoy)
        - 門迫 (Gate Constraint)

        Args:
            plate: The QimenPlate to check

        Returns:
            Dictionary of special conditions found
        """
        return self.analyzer.check_special_conditions(plate)

    def has_three_wonders(self, plate: QimenPlate, palace_num: int) -> bool:
        """
        Check if a palace has one of the Three Wonders (三奇: 乙丙丁).

        Args:
            plate: The QimenPlate to check
            palace_num: Palace number to check

        Returns:
            True if palace has a Three Wonder stem
        """
        palace = plate.get_palace(palace_num)
        if palace:
            return palace.has_three_wonders()
        return False

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_current_solar_term(self) -> SolarTerm:
        """
        Get the current solar term.

        Returns:
            The current SolarTerm
        """
        jd = self.lunar_calendar.gregorian_to_julian(datetime.now())
        longitude = self.lunar_calendar.calculate_solar_longitude(jd)
        return self.calculator.get_current_solar_term(longitude)

    def get_current_dun_type(self) -> DunType:
        """
        Get the current dun type (Yang or Yin).

        Returns:
            Current DunType
        """
        jd = self.lunar_calendar.gregorian_to_julian(datetime.now())
        longitude = self.lunar_calendar.calculate_solar_longitude(jd)
        return self.calculator.determine_dun_type(longitude)


# =============================================================================
# Module-level convenience functions
# =============================================================================

def quick_calculation(dt: Optional[datetime] = None) -> QimenPlate:
    """
    Quick convenience function for one-off calculations.

    Args:
        dt: Datetime to calculate (defaults to now)

    Returns:
        QimenPlate for the specified time
    """
    qimen = QimenDunjia()
    return qimen.calculate(dt or datetime.now())


def display_current_plate() -> str:
    """
    Generate display of the current Qimen plate.

    Returns:
        Formatted string display of the current plate
    """
    qimen = QimenDunjia()
    plate = qimen.calculate_for_now()
    return plate.format_display()
