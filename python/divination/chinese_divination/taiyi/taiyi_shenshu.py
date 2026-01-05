"""
Taiyi Shenshu (太乙神數) - Main Public API

The Divine Calculation of the Supreme Unity - one of the Three Styles (三式)
of Chinese divination. Taiyi focuses on grand cycles, national fate, and
cosmic patterns across long time spans.

Usage:
    from chinese_divination.taiyi import TaiyiShenshu

    # Create instance
    taiyi = TaiyiShenshu()

    # Calculate for current time
    plate = taiyi.calculate_now()
    print(plate.format_summary())

    # Calculate for specific datetime
    from datetime import datetime
    plate = taiyi.calculate(datetime(2024, 1, 1))

    # Get analysis
    analysis = taiyi.analyze(plate)

Classical Sources:
    - 太乙金鏡式經 (Tang Dynasty, Wang Ximing)
    - 太乙秘書 (Song Dynasty, Wang Zuo)
    - 黃帝太乙八門入式訣 (Daoist Canon)

碼道長存 — The Way of Code endures
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.lunar_calendar import ChineseLunarCalendar
from .constants import TaiyiDunType, TaiyiEra, TaiyiConstants
from .spirits import SixteenSpirits, TaiyiSpirit
from .palaces import NinePalaces, TaiyiPlate
from .calculator import TaiyiCalculator


# =============================================================================
# Main Public API Class
# =============================================================================

class TaiyiShenshu:
    """
    Main interface for Taiyi Shenshu (太乙神數) divination.

    Provides methods for:
    - Calculating Taiyi plates for any datetime
    - Analyzing calculation results
    - Generating formatted output

    Example:
        >>> taiyi = TaiyiShenshu()
        >>> plate = taiyi.calculate_now()
        >>> print(f"Taiyi is in Palace {plate.taiyi_palace}")
        >>> print(f"Tianmu: {plate.tianmu_spirit}")
    """

    def __init__(self, lunar_calendar: Optional[ChineseLunarCalendar] = None):
        """
        Initialize TaiyiShenshu.

        Args:
            lunar_calendar: Optional lunar calendar instance
        """
        self.lunar_calendar = lunar_calendar or ChineseLunarCalendar()
        self.calculator = TaiyiCalculator(self.lunar_calendar)
        self.spirits = SixteenSpirits()

    # =========================================================================
    # Calculation Methods
    # =========================================================================

    def calculate(self, dt: datetime,
                   calculation_type: str = "year") -> TaiyiPlate:
        """
        Calculate Taiyi plate for a specific datetime.

        Args:
            dt: The datetime to calculate for
            calculation_type: One of "year", "month", "day", "hour"
                - "year" (歲計): Annual calculations, national fate
                - "month" (月計): Monthly calculations
                - "day" (日計): Daily calculations
                - "hour" (時計): Hourly calculations, tactical matters

        Returns:
            TaiyiPlate with complete calculation results
        """
        return self.calculator.calculate(dt, calculation_type)

    def calculate_now(self, calculation_type: str = "year") -> TaiyiPlate:
        """
        Calculate Taiyi plate for the current moment.

        Args:
            calculation_type: One of "year", "month", "day", "hour"

        Returns:
            TaiyiPlate with complete calculation results
        """
        return self.calculate(datetime.now(), calculation_type)

    def calculate_year(self, dt: datetime) -> TaiyiPlate:
        """Calculate annual Taiyi plate (歲計)."""
        return self.calculator.calculate_year(dt)

    def calculate_month(self, dt: datetime) -> TaiyiPlate:
        """Calculate monthly Taiyi plate (月計)."""
        return self.calculator.calculate_month(dt)

    def calculate_day(self, dt: datetime) -> TaiyiPlate:
        """Calculate daily Taiyi plate (日計)."""
        return self.calculator.calculate_day(dt)

    def calculate_hour(self, dt: datetime) -> TaiyiPlate:
        """Calculate hourly Taiyi plate (時計)."""
        return self.calculator.calculate_hour(dt)

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def analyze(self, plate: TaiyiPlate) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a Taiyi plate.

        Args:
            plate: The TaiyiPlate to analyze

        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            "summary": self._analyze_summary(plate),
            "taiyi_position": self._analyze_taiyi_position(plate),
            "tianmu_analysis": self._analyze_tianmu(plate),
            "calculation_harmony": self._analyze_calculations(plate),
            "strategic_assessment": self._analyze_strategy(plate),
            "recommendations": self._generate_recommendations(plate),
        }

        return analysis

    def _analyze_summary(self, plate: TaiyiPlate) -> Dict[str, str]:
        """Generate summary analysis."""
        return {
            "dun_type": f"{plate.dun_type.chinese} - {'Forward' if plate.dun_type == TaiyiDunType.YANG else 'Backward'} rotation",
            "era": f"{plate.era.chinese_name} ({plate.era.pattern_name})",
            "year_in_era": f"Year {plate.era_year} of 10",
            "taiyi_location": f"Palace {plate.taiyi_palace}",
            "tianmu": f"{plate.tianmu_spirit.chinese} ({plate.tianmu_spirit.pinyin})",
        }

    def _analyze_taiyi_position(self, plate: TaiyiPlate) -> Dict[str, Any]:
        """Analyze Taiyi's palace position."""
        palace = plate.palaces[plate.taiyi_palace]

        position_type = "Unknown"
        if plate.taiyi_palace in {3, 4, 8}:
            position_type = "地内宫 (Inner Palace) - Favors Defender"
        elif plate.taiyi_palace in {2, 6, 7, 9}:
            position_type = "天外宫 (Outer Palace) - Favors Attacker"

        special = None
        if plate.taiyi_palace == 1:
            special = "绝阳 (Extreme Yang) - Yang forces may overextend"
        elif plate.taiyi_palace == 9:
            special = "绝阴 (Extreme Yin) - Yin forces may overextend"
        elif plate.taiyi_palace in {4, 6}:
            special = "绝气 (Qi Extinction) - Energy transformation"
        elif plate.taiyi_palace in {2, 8}:
            special = "易气 (Qi Exchange) - Transitional energy"

        return {
            "palace_number": plate.taiyi_palace,
            "trigram": palace.trigram,
            "direction": palace.direction.value,
            "element": palace.element.value,
            "position_type": position_type,
            "special_position": special,
            "years_in_palace": plate.years_in_palace,
        }

    def _analyze_tianmu(self, plate: TaiyiPlate) -> Dict[str, str]:
        """Analyze Tianmu (天目) spirit."""
        spirit = plate.tianmu_spirit

        return {
            "spirit_name": spirit.chinese,
            "branch": spirit.branch,
            "element": spirit.element.value,
            "polarity": spirit.polarity.value,
            "meaning": spirit.meaning,
            "is_zhong_liu": "Yes - Extra count position" if spirit.is_zhong_liu else "No",
        }

    def _analyze_calculations(self, plate: TaiyiPlate) -> Dict[str, Any]:
        """Analyze host/guest calculations."""
        host = plate.host_calculation
        guest = plate.guest_calculation

        # Determine which is favorable
        harmony_status = "Harmonious (算和)" if plate.is_calculation_harmonious else "Disharmonious"

        # Analyze individual calculations
        host_analysis = self._analyze_single_calculation(host, "Host")
        guest_analysis = self._analyze_single_calculation(guest, "Guest")

        return {
            "host_calculation": host,
            "guest_calculation": guest,
            "difference": abs(host - guest),
            "total": host + guest,
            "harmony": harmony_status,
            "host_analysis": host_analysis,
            "guest_analysis": guest_analysis,
        }

    def _analyze_single_calculation(self, value: int, name: str) -> str:
        """Analyze a single calculation value."""
        if value % 10 == 0:
            return f"{name}: Round number - stable"
        elif value in {1, 3, 7, 9}:
            return f"{name}: Single yang ({value}) - active energy"
        elif value in {2, 4, 6, 8}:
            return f"{name}: Single yin ({value}) - receptive energy"
        elif value >= 30:
            return f"{name}: High value ({value}) - strong momentum"
        else:
            return f"{name}: Moderate value ({value}) - balanced state"

    def _analyze_strategy(self, plate: TaiyiPlate) -> Dict[str, Any]:
        """Generate strategic assessment."""
        # Determine overall advantage
        if plate.favors_host and not plate.favors_guest:
            advantage = "Defender (主人)"
            advice = "Hold position, let opponent come to you"
        elif plate.favors_guest and not plate.favors_host:
            advantage = "Attacker (客人)"
            advice = "Take initiative, advance boldly"
        elif plate.favors_host and plate.favors_guest:
            advantage = "Balanced"
            advice = "Success depends on individual capability"
        else:
            advantage = "Neither clearly favored"
            advice = "Proceed with caution, seek better timing"

        return {
            "overall_advantage": advantage,
            "strategic_advice": advice,
            "favors_host": plate.favors_host,
            "favors_guest": plate.favors_guest,
            "host_general_palace": plate.host_general_palace,
            "guest_general_palace": plate.guest_general_palace,
        }

    def _generate_recommendations(self, plate: TaiyiPlate) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Based on Taiyi position
        if plate.taiyi_palace in {3, 4, 8}:
            recommendations.append("Favorable for defense and consolidation")
        elif plate.taiyi_palace in {2, 6, 7, 9}:
            recommendations.append("Favorable for initiative and action")

        # Based on calculation harmony
        if plate.is_calculation_harmonious:
            recommendations.append("Calculations harmonious - cooperative ventures favored")
        else:
            recommendations.append("Calculations disharmonious - independent action preferred")

        # Based on Tianmu spirit
        spirit = plate.tianmu_spirit
        if spirit.element.value == "木":
            recommendations.append("Wood energy - growth and expansion supported")
        elif spirit.element.value == "火":
            recommendations.append("Fire energy - visibility and recognition enhanced")
        elif spirit.element.value == "土":
            recommendations.append("Earth energy - stability and foundation building")
        elif spirit.element.value == "金":
            recommendations.append("Metal energy - precision and completion emphasized")
        elif spirit.element.value == "水":
            recommendations.append("Water energy - wisdom and adaptability key")

        # Based on era
        if plate.era_year <= 3:
            recommendations.append("Early era phase - good for new beginnings")
        elif plate.era_year >= 8:
            recommendations.append("Late era phase - complete existing matters")

        return recommendations

    # =========================================================================
    # Display Methods
    # =========================================================================

    def format_plate_display(self, plate: TaiyiPlate) -> str:
        """
        Format a TaiyiPlate for display.

        Args:
            plate: The plate to format

        Returns:
            Formatted string representation
        """
        return plate.format_summary()

    def get_summary(self, plate: TaiyiPlate) -> Dict[str, Any]:
        """
        Get a summary dictionary of plate data.

        Args:
            plate: The plate to summarize

        Returns:
            Dictionary with key information
        """
        return {
            "datetime": plate.datetime_info,
            "dun_type": plate.dun_type.chinese,
            "era": plate.era.chinese_name,
            "taiyi_palace": plate.taiyi_palace,
            "tianmu": plate.tianmu_spirit.chinese,
            "jishen": plate.jishen_branch,
            "heshen": plate.heshen_branch,
            "host_calculation": plate.host_calculation,
            "guest_calculation": plate.guest_calculation,
            "is_harmonious": plate.is_calculation_harmonious,
            "favors_host": plate.favors_host,
            "favors_guest": plate.favors_guest,
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_spirit_info(self, name: str) -> Optional[TaiyiSpirit]:
        """
        Get information about a specific spirit.

        Args:
            name: Chinese name of the spirit (e.g., "武德", "地主")

        Returns:
            TaiyiSpirit object or None if not found
        """
        return self.spirits.get_by_chinese(name)

    def list_spirits(self) -> List[str]:
        """Get list of all spirit names in traversal order."""
        return [s.chinese for s in self.spirits.get_traversal_sequence()]

    def get_era_info(self, era: TaiyiEra) -> Dict[str, Any]:
        """
        Get information about a specific era.

        Args:
            era: The TaiyiEra to get info for

        Returns:
            Dictionary with era information
        """
        return {
            "number": era.number,
            "name": era.chinese_name,
            "pattern": era.pattern_name,
            "stem_pairs": era.stem_pairs,
            "taiyi_palace": era.taiyi_palace,
            "tianmu_spirit": era.tianmu_spirit,
        }


# =============================================================================
# Module-level convenience function
# =============================================================================

def calculate_taiyi(dt: Optional[datetime] = None,
                     calculation_type: str = "year") -> TaiyiPlate:
    """
    Convenience function for quick Taiyi calculation.

    Args:
        dt: Datetime to calculate for (defaults to now)
        calculation_type: One of "year", "month", "day", "hour"

    Returns:
        TaiyiPlate with calculation results
    """
    taiyi = TaiyiShenshu()
    if dt is None:
        dt = datetime.now()
    return taiyi.calculate(dt, calculation_type)
