"""
Da Liu Ren Main API (大六壬主接口)

Primary interface for Da Liu Ren divination.

This is the main entry point for using the Liu Ren divination system.
It provides a clean, high-level API for calculating and analyzing
Liu Ren plates.

Usage:
    from chinese_divination.liuren import LiuRen

    # Basic usage
    liuren = LiuRen()
    plate = liuren.calculate_now()
    print(liuren.get_summary(plate))

    # Calculate for specific datetime
    from datetime import datetime
    dt = datetime(2024, 6, 15, 10, 30)
    plate = liuren.calculate(dt)
    analysis = liuren.analyze(plate)

    # Query-specific divination
    from chinese_divination.liuren.analysis import QueryType
    result = liuren.query(dt, QueryType.CAREER)

Reference: 六壬大全 (Ming Dynasty, 郭載騋)
"""

from datetime import datetime
from typing import Dict, Any, Optional

from .calculator import LiuRenCalculator
from .analysis import LiuRenAnalyzer, QueryType, Favorability
from .patterns import analyze_patterns, apply_pattern_to_plate
from .plates import LiuRenPlate
from .components import (
    TianJiang, TwelveGenerals, get_twelve_generals,
    SiKe, SanChuan,
)


class LiuRen:
    """
    Primary interface for Da Liu Ren (大六壬) divination.

    Da Liu Ren is one of the Three Great Oracles (三式) of classical
    Chinese divination, alongside Qimen Dunjia (奇門遁甲) and
    Taiyi Shenshu (太乙神數).

    This class provides methods to:
    - Calculate complete Liu Ren plates for any datetime
    - Analyze plates with pattern recognition
    - Get directional and query-specific advice
    - Format results for display

    Attributes:
        calculator: The LiuRenCalculator instance
        analyzer: The LiuRenAnalyzer instance (lazy-loaded)
        latitude: Latitude for sunrise/sunset calculation

    Example:
        >>> liuren = LiuRen()
        >>> plate = liuren.calculate_now()
        >>> print(plate.format_display())

        >>> # For specific time
        >>> from datetime import datetime
        >>> plate = liuren.calculate(datetime(2024, 1, 15, 10, 0))
        >>> summary = liuren.get_summary(plate)
    """

    def __init__(self, lunar_calendar=None, latitude: float = 35.0):
        """
        Initialize the LiuRen divination system.

        Args:
            lunar_calendar: Optional ChineseLunarCalendar instance.
                           If None, one will be created automatically.
            latitude: Latitude for sunrise/sunset calculation.
                     Default is 35°N (central China).
        """
        self.calculator = LiuRenCalculator(lunar_calendar, latitude)
        self.latitude = latitude
        self._analyzer: Optional[LiuRenAnalyzer] = None

    @property
    def analyzer(self) -> LiuRenAnalyzer:
        """Lazy-load the analyzer when needed."""
        if self._analyzer is None:
            self._analyzer = LiuRenAnalyzer()
        return self._analyzer

    # =========================================================================
    # Core Calculation Methods (核心計算方法)
    # =========================================================================

    def calculate(self, dt: datetime) -> LiuRenPlate:
        """
        Calculate a complete Liu Ren plate for the given datetime.

        This is the primary calculation method. It computes:
        - Heaven and Earth Plates (天盤/地盤)
        - Four Lessons (四課)
        - Three Transmissions (三傳)
        - Noble Person and General positions (貴人/天將)
        - Pattern identification (課體)

        Args:
            dt: The datetime for divination

        Returns:
            Complete LiuRenPlate with all calculated elements

        Example:
            >>> liuren = LiuRen()
            >>> plate = liuren.calculate(datetime(2024, 6, 15, 10, 30))
            >>> print(plate.day_stem, plate.day_branch)
        """
        # Calculate the plate
        plate = self.calculator.calculate_plate(dt)

        # Apply pattern analysis
        apply_pattern_to_plate(plate)

        return plate

    def calculate_now(self) -> LiuRenPlate:
        """
        Calculate a Liu Ren plate for the current moment.

        Convenience method equivalent to calculate(datetime.now()).

        Returns:
            Complete LiuRenPlate for the current time

        Example:
            >>> liuren = LiuRen()
            >>> plate = liuren.calculate_now()
            >>> print(plate.format_display())
        """
        return self.calculate(datetime.now())

    # =========================================================================
    # Summary and Display Methods (摘要與顯示方法)
    # =========================================================================

    def get_summary(self, plate: LiuRenPlate) -> str:
        """
        Get a human-readable summary of the plate.

        This provides a concise text overview of the divination
        results, suitable for display or logging.

        Args:
            plate: The calculated LiuRenPlate

        Returns:
            Formatted string summary

        Example:
            >>> plate = liuren.calculate_now()
            >>> print(liuren.get_summary(plate))
        """
        return plate.format_display()

    def get_plate_data(self, plate: LiuRenPlate) -> Dict[str, Any]:
        """
        Get the plate data as a dictionary.

        This provides structured data suitable for serialization
        or further processing.

        Args:
            plate: The calculated LiuRenPlate

        Returns:
            Dictionary containing all plate data
        """
        return plate.get_summary()

    # =========================================================================
    # Analysis Methods (分析方法)
    # =========================================================================

    def analyze(self, plate: LiuRenPlate) -> Dict[str, Any]:
        """
        Perform complete analysis of a Liu Ren plate.

        This provides comprehensive interpretation including:
        - Pattern analysis
        - Three Transmissions interpretation
        - General position analysis
        - Directional advice
        - Overall favorability rating

        Args:
            plate: The calculated LiuRenPlate

        Returns:
            Dictionary with complete analysis

        Example:
            >>> plate = liuren.calculate_now()
            >>> analysis = liuren.analyze(plate)
            >>> print(analysis['overall_favorability'])
        """
        return self.analyzer.analyze(plate)

    def analyze_san_chuan(self, plate: LiuRenPlate) -> Dict[str, Any]:
        """
        Analyze the Three Transmissions in detail.

        The Three Transmissions represent:
        - 初傳 (Initial): The beginning or cause
        - 中傳 (Middle): The development or process
        - 末傳 (Final): The outcome or result

        Args:
            plate: The calculated LiuRenPlate

        Returns:
            Detailed analysis of the Three Transmissions
        """
        return self.analyzer.analyze_san_chuan(plate.san_chuan, plate)

    def get_directional_advice(self, plate: LiuRenPlate) -> Dict[str, Any]:
        """
        Get directional recommendations from the plate.

        Analyzes which directions are favorable or unfavorable
        based on general positions, clashes, and harmonies.

        Args:
            plate: The calculated LiuRenPlate

        Returns:
            Dictionary mapping directions to recommendations
        """
        advice = self.analyzer.get_directional_advice(plate)
        return {
            direction: {
                'favorability': a.favorability.chinese,
                'reason': a.reason,
                'general': a.associated_general,
                'branch': a.associated_branch,
            }
            for direction, a in advice.items()
        }

    # =========================================================================
    # Query Methods (占問方法)
    # =========================================================================

    def query(self, dt: datetime,
             query_type: QueryType = QueryType.GENERAL) -> Dict[str, Any]:
        """
        Perform a query-specific divination.

        This combines calculation and analysis for a specific
        type of question (career, wealth, relationships, etc.).

        Args:
            dt: The datetime for divination
            query_type: The type of question being asked

        Returns:
            Query-specific analysis with targeted advice

        Example:
            >>> from chinese_divination.liuren.analysis import QueryType
            >>> result = liuren.query(datetime.now(), QueryType.CAREER)
            >>> print(result['advice'])
        """
        plate = self.calculate(dt)
        return self.analyzer.query(plate, query_type)

    def query_now(self, query_type: QueryType = QueryType.GENERAL) -> Dict[str, Any]:
        """
        Perform a query for the current moment.

        Convenience method equivalent to query(datetime.now(), query_type).

        Args:
            query_type: The type of question being asked

        Returns:
            Query-specific analysis
        """
        return self.query(datetime.now(), query_type)

    # =========================================================================
    # Utility Methods (工具方法)
    # =========================================================================

    def get_twelve_generals(self) -> TwelveGenerals:
        """
        Get the TwelveGenerals container.

        This provides access to information about all 12 Heavenly
        Generals, including their attributes and meanings.

        Returns:
            TwelveGenerals container instance
        """
        return get_twelve_generals()

    def get_overall_favorability(self, plate: LiuRenPlate) -> str:
        """
        Get the overall favorability rating for a plate.

        Returns a simple favorability rating:
        - 大吉 (Very Auspicious)
        - 吉 (Auspicious)
        - 平 (Neutral)
        - 凶 (Inauspicious)
        - 大凶 (Very Inauspicious)

        Args:
            plate: The calculated LiuRenPlate

        Returns:
            Chinese favorability rating string
        """
        return self.analyzer._get_overall_favorability(plate).chinese


# =============================================================================
# Module-Level Convenience Functions (模組級便捷函數)
# =============================================================================

def divine(dt: datetime = None, query_type: QueryType = None) -> Dict[str, Any]:
    """
    Quick divination function.

    A convenience function for quick divination without
    explicitly creating a LiuRen instance.

    Args:
        dt: The datetime for divination (default: now)
        query_type: Optional query type for specific analysis

    Returns:
        Dictionary with plate summary and analysis

    Example:
        >>> from chinese_divination.liuren import divine
        >>> result = divine()
        >>> print(result['summary'])
    """
    liuren = LiuRen()

    if dt is None:
        dt = datetime.now()

    plate = liuren.calculate(dt)

    result = {
        'summary': liuren.get_summary(plate),
        'data': liuren.get_plate_data(plate),
    }

    if query_type:
        result['query_analysis'] = liuren.analyzer.query(plate, query_type)
    else:
        result['analysis'] = liuren.analyze(plate)

    return result
