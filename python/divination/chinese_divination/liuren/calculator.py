"""
Da Liu Ren Calculator (大六壬計算器)

The core calculation engine for Liu Ren divination.

This module implements the complete calculation workflow:
1. Determine Monthly General from solar term
2. Build Heaven and Earth Plates
3. Calculate Four Lessons (四課)
4. Derive Three Transmissions (三傳)
5. Position Noble Person and Generals
6. Identify patterns

Primary reference: 六壬大全 (Ming Dynasty, 郭載騋)
"""

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from .constants import (
    TWELVE_BRANCHES, TEN_STEMS, BRANCH_TO_INDEX,
    SOLAR_TERM_TO_GENERAL, ZHONG_QI_GENERALS,
    NOBLE_PERSON_DAY, NOBLE_PERSON_NIGHT,
    STEM_RESIDENCE, HOUR_TO_BRANCH,
    BRANCH_ELEMENT, ELEMENT_CONTROLS,
    SIX_CLASHES,
    get_branch_index, get_branch_by_index, get_offset_branch,
    is_yang_stem, is_yin_stem, does_control,
    GENERAL_BY_BRANCH,
)
from .components import (
    TianJiang, TwelveGenerals, get_twelve_generals,
    Ke, SiKe, SanChuan, KeRelation,
    GeneralPosition,
)
from .plates import (
    LiuRenPalace, TwelvePalaces,
    HeavenPlate, EarthPlate, LiuRenPlate,
)


# =============================================================================
# Sunrise/Sunset Calculation (日出日落計算)
# =============================================================================

def calculate_julian_day(dt: datetime) -> float:
    """
    Calculate the Julian Day Number for a given datetime.

    This is a simplified calculation sufficient for sunrise/sunset.
    """
    year = dt.year
    month = dt.month
    day = dt.day + dt.hour / 24.0 + dt.minute / 1440.0

    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)

    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

    return jd


def calculate_sunrise_sunset(dt: datetime, latitude: float = 35.0) -> Tuple[float, float]:
    """
    Calculate approximate sunrise and sunset times for a given date and latitude.

    Uses a simplified algorithm based on the equation of time and solar declination.

    Args:
        dt: The date for calculation
        latitude: Latitude in degrees (default 35°N for central China)

    Returns:
        Tuple of (sunrise_hour, sunset_hour) in 24-hour format
    """
    # Day of year
    N = dt.timetuple().tm_yday

    # Solar declination (simplified)
    # δ = -23.45° × cos(360/365 × (N + 10))
    declination = -23.45 * math.cos(math.radians(360 / 365 * (N + 10)))

    # Convert to radians
    lat_rad = math.radians(latitude)
    dec_rad = math.radians(declination)

    # Hour angle at sunrise/sunset
    # cos(ω) = -tan(φ) × tan(δ)
    cos_omega = -math.tan(lat_rad) * math.tan(dec_rad)

    # Clamp to valid range for polar regions
    cos_omega = max(-1, min(1, cos_omega))

    omega = math.degrees(math.acos(cos_omega))

    # Sunrise and sunset in hours (solar noon at 12:00)
    sunrise = 12.0 - omega / 15.0
    sunset = 12.0 + omega / 15.0

    # Equation of time correction (simplified)
    B = 360 / 365 * (N - 81)
    EoT = 9.87 * math.sin(math.radians(2 * B)) - 7.53 * math.cos(math.radians(B)) - 1.5 * math.sin(math.radians(B))
    EoT_hours = EoT / 60

    # Apply correction
    sunrise -= EoT_hours
    sunset -= EoT_hours

    return (sunrise, sunset)


def is_daytime(dt: datetime, latitude: float = 35.0) -> bool:
    """
    Determine if a given datetime is during daytime.

    Args:
        dt: The datetime to check
        latitude: Latitude in degrees (default 35°N)

    Returns:
        True if daytime (between sunrise and sunset)
    """
    sunrise, sunset = calculate_sunrise_sunset(dt, latitude)
    current_hour = dt.hour + dt.minute / 60.0

    return sunrise <= current_hour <= sunset


# =============================================================================
# LiuRenCalculator Class (六壬計算器)
# =============================================================================

class LiuRenCalculator:
    """
    Core calculation engine for Da Liu Ren divination.

    This class orchestrates all calculations needed to produce
    a complete LiuRenPlate for a given datetime.

    Usage:
        calculator = LiuRenCalculator()
        plate = calculator.calculate_plate(datetime.now())
    """

    def __init__(self, lunar_calendar=None, latitude: float = 35.0):
        """
        Initialize the calculator.

        Args:
            lunar_calendar: Optional ChineseLunarCalendar instance
            latitude: Latitude for sunrise/sunset calculation (default 35°N)
        """
        if lunar_calendar is None:
            from ..core.lunar_calendar import ChineseLunarCalendar
            lunar_calendar = ChineseLunarCalendar()

        self.lunar_calendar = lunar_calendar
        self.latitude = latitude
        self.twelve_generals = get_twelve_generals()

    # =========================================================================
    # Solar Term and Monthly General (節氣與月將)
    # =========================================================================

    def get_current_solar_term(self, solar_longitude: float) -> str:
        """
        Determine the current solar term from solar longitude.

        Args:
            solar_longitude: Sun's ecliptic longitude in degrees (0-360)

        Returns:
            Chinese name of the current solar term
        """
        # Solar terms occur every 15 degrees
        # Starting from Spring Equinox (春分) at 0°
        SOLAR_TERMS = [
            (0, '春分'), (15, '清明'), (30, '穀雨'),
            (45, '立夏'), (60, '小滿'), (75, '芒種'),
            (90, '夏至'), (105, '小暑'), (120, '大暑'),
            (135, '立秋'), (150, '處暑'), (165, '白露'),
            (180, '秋分'), (195, '寒露'), (210, '霜降'),
            (225, '立冬'), (240, '小雪'), (255, '大雪'),
            (270, '冬至'), (285, '小寒'), (300, '大寒'),
            (315, '立春'), (330, '雨水'), (345, '驚蟄'),
        ]

        # Normalize longitude
        lon = solar_longitude % 360

        # Find the current solar term
        for i, (deg, term) in enumerate(SOLAR_TERMS):
            next_deg = SOLAR_TERMS[(i + 1) % 24][0]
            if next_deg == 0:
                next_deg = 360

            if deg <= lon < next_deg:
                return term

        return '春分'  # Default fallback

    def get_monthly_general(self, solar_longitude: float) -> Tuple[str, str]:
        """
        Determine the Monthly General (月將) from solar longitude.

        The Monthly General changes at the Zhong Qi (中氣) solar terms.

        Args:
            solar_longitude: Sun's ecliptic longitude in degrees

        Returns:
            Tuple of (general_name, general_branch)
        """
        solar_term = self.get_current_solar_term(solar_longitude)

        # Look up the general for this solar term
        result = SOLAR_TERM_TO_GENERAL.get(solar_term)
        if result:
            return result

        # Fallback: use ZHONG_QI_GENERALS
        branch = ZHONG_QI_GENERALS.get(solar_term, '子')
        name = GENERAL_BY_BRANCH.get(branch, '神后')
        return (name, branch)

    # =========================================================================
    # Plate Construction (盤構建)
    # =========================================================================

    def build_heaven_plate(self, monthly_general_branch: str,
                          hour_branch: str) -> HeavenPlate:
        """
        Build the Heaven Plate (天盤).

        The Monthly General is placed at the Hour Branch position,
        and other branches rotate accordingly.

        Args:
            monthly_general_branch: The branch of the current monthly general
            hour_branch: The branch of the query hour

        Returns:
            Constructed HeavenPlate
        """
        return HeavenPlate(
            monthly_general_branch=monthly_general_branch,
            query_hour_branch=hour_branch,
        )

    def build_earth_plate(self) -> EarthPlate:
        """
        Build the Earth Plate (地盤).

        The Earth Plate is always fixed.

        Returns:
            Constructed EarthPlate
        """
        return EarthPlate()

    # =========================================================================
    # Four Lessons Calculation (四課計算)
    # =========================================================================

    def calculate_si_ke(self, day_stem: str, day_branch: str,
                        heaven_plate: HeavenPlate) -> SiKe:
        """
        Calculate the Four Lessons (四課).

        Algorithm:
        第一課: 日干寄宮上神 / 日干寄宮
        第二課: 第一課上神的天盤位置上神 / 第一課上神
        第三課: 日支上神 / 日支
        第四課: 第三課上神的天盤位置上神 / 第三課上神

        Args:
            day_stem: The day's Heavenly Stem
            day_branch: The day's Earthly Branch
            heaven_plate: The constructed Heaven Plate

        Returns:
            SiKe containing the four lessons
        """
        # Get day stem's residence (寄宮)
        stem_residence = STEM_RESIDENCE.get(day_stem, '寅')

        # ===== 第一課 =====
        # 上: The heaven branch at stem_residence position
        # 下: The stem_residence itself
        ke1_shang = heaven_plate.get_heaven_branch_at(stem_residence)
        ke1_xia = stem_residence

        # ===== 第二課 =====
        # 上: The heaven branch at ke1_shang's position
        # 下: ke1_shang
        ke2_shang = heaven_plate.get_heaven_branch_at(ke1_shang)
        ke2_xia = ke1_shang

        # ===== 第三課 =====
        # 上: The heaven branch at day_branch position
        # 下: day_branch
        ke3_shang = heaven_plate.get_heaven_branch_at(day_branch)
        ke3_xia = day_branch

        # ===== 第四課 =====
        # 上: The heaven branch at ke3_shang's position
        # 下: ke3_shang
        ke4_shang = heaven_plate.get_heaven_branch_at(ke3_shang)
        ke4_xia = ke3_shang

        lessons = [
            Ke(shang=ke1_shang, xia=ke1_xia, index=1),
            Ke(shang=ke2_shang, xia=ke2_xia, index=2),
            Ke(shang=ke3_shang, xia=ke3_xia, index=3),
            Ke(shang=ke4_shang, xia=ke4_xia, index=4),
        ]

        return SiKe(lessons=lessons, day_stem=day_stem, day_branch=day_branch)

    # =========================================================================
    # Three Transmissions Derivation (三傳推導)
    # =========================================================================

    def derive_san_chuan(self, si_ke: SiKe, day_stem: str,
                        heaven_plate: HeavenPlate) -> SanChuan:
        """
        Derive the Three Transmissions (三傳) from the Four Lessons.

        Selection priority:
        1. 賊克課 - If there's 下賊上 (lower attacks upper), use that lesson
        2. 克賊課 - If there's 上克下 (upper controls lower), use that
        3. 比用課 - If no克, use element comparison
        4. 涉害課 - Special harm analysis

        Args:
            si_ke: The calculated Four Lessons
            day_stem: The day's Heavenly Stem
            heaven_plate: The Heaven Plate

        Returns:
            SanChuan containing the three transmissions
        """
        # Check for 賊 (下賊上) first - highest priority
        zei_lessons = si_ke.get_zei_lessons()
        if zei_lessons:
            return self._derive_from_zei(zei_lessons, day_stem, heaven_plate)

        # Check for 克 (上克下)
        ke_lessons = si_ke.get_shang_ke_lessons()
        if ke_lessons:
            return self._derive_from_ke(ke_lessons, day_stem, heaven_plate)

        # No 克, check for 比用 (same element comparison)
        return self._derive_bi_yong(si_ke, day_stem, heaven_plate)

    def _derive_from_zei(self, zei_lessons: List[Ke], day_stem: str,
                         heaven_plate: HeavenPlate) -> SanChuan:
        """
        Derive transmissions from 賊克 lessons.

        When multiple賊 exist, use the one with deeper harm (涉害).
        """
        # If only one賊, use it
        if len(zei_lessons) == 1:
            selected = zei_lessons[0]
        else:
            # Multiple賊: select by涉害 (harm depth)
            selected = self._select_by_she_hai(zei_lessons)

        return self._generate_transmissions(
            selected.shang, day_stem, heaven_plate, "賊克"
        )

    def _derive_from_ke(self, ke_lessons: List[Ke], day_stem: str,
                        heaven_plate: HeavenPlate) -> SanChuan:
        """
        Derive transmissions from 上克下 lessons.
        """
        if len(ke_lessons) == 1:
            selected = ke_lessons[0]
        else:
            selected = self._select_by_she_hai(ke_lessons)

        return self._generate_transmissions(
            selected.shang, day_stem, heaven_plate, "克賊"
        )

    def _derive_bi_yong(self, si_ke: SiKe, day_stem: str,
                        heaven_plate: HeavenPlate) -> SanChuan:
        """
        Derive transmissions using 比用 (comparison method).

        Used when there's no克 relationship in any lesson.
        """
        # Get the day element
        day_element = BRANCH_ELEMENT.get(si_ke.day_branch, '土')

        # Find lessons with same element as day
        same_element = []
        for ke in si_ke.lessons:
            shang_element = BRANCH_ELEMENT.get(ke.shang, '')
            if shang_element == day_element:
                same_element.append(ke)

        if same_element:
            selected = same_element[0]  # Use first match
            return self._generate_transmissions(
                selected.shang, day_stem, heaven_plate, "比用"
            )

        # If no same element, use first lesson's upper
        return self._generate_transmissions(
            si_ke.lesson_1.shang, day_stem, heaven_plate, "遙克"
        )

    def _select_by_she_hai(self, lessons: List[Ke]) -> Ke:
        """
        Select a lesson by 涉害 (harm depth) analysis.

        The lesson whose upper element "travels" furthest
        through the controlling cycle is selected.
        """
        if not lessons:
            raise ValueError("No lessons to select from")

        if len(lessons) == 1:
            return lessons[0]

        # Calculate harm depth for each
        max_harm = -1
        selected = lessons[0]

        for ke in lessons:
            harm = self._calculate_harm_depth(ke.shang)
            if harm > max_harm:
                max_harm = harm
                selected = ke

        return selected

    def _calculate_harm_depth(self, branch: str) -> int:
        """
        Calculate the harm depth (涉害) for a branch.

        This measures how many controlling steps the element
        must traverse. Higher = more harmful.
        """
        element = BRANCH_ELEMENT.get(branch, '土')

        # Count controlling chain length
        depth = 0
        current = element
        seen = {current}

        while True:
            controlled = ELEMENT_CONTROLS.get(current)
            if not controlled or controlled in seen:
                break
            depth += 1
            current = controlled
            seen.add(current)

        return depth

    def _generate_transmissions(self, initial_branch: str, day_stem: str,
                                heaven_plate: HeavenPlate,
                                method: str) -> SanChuan:
        """
        Generate the three transmissions from an initial branch.

        For Yang days: count forward through branches
        For Yin days: count backward through branches

        Args:
            initial_branch: The starting branch (初傳)
            day_stem: The day's stem (determines direction)
            heaven_plate: For looking up overlaid branches
            method: Description of derivation method

        Returns:
            SanChuan with all three transmissions
        """
        chu_chuan = initial_branch

        # Determine counting direction based on day stem
        if is_yang_stem(day_stem):
            # Yang day: count forward
            # 中傳 is the heaven branch at 初傳's position
            zhong_chuan = heaven_plate.get_heaven_branch_at(chu_chuan)
            # 末傳 is the heaven branch at 中傳's position
            mo_chuan = heaven_plate.get_heaven_branch_at(zhong_chuan)
        else:
            # Yin day: count backward (use earth position of heaven branch)
            zhong_chuan = heaven_plate.get_earth_branch_for(chu_chuan) or chu_chuan
            mo_chuan = heaven_plate.get_earth_branch_for(zhong_chuan) or zhong_chuan

        return SanChuan(
            chu_chuan=chu_chuan,
            zhong_chuan=zhong_chuan,
            mo_chuan=mo_chuan,
            derivation_method=method,
        )

    # =========================================================================
    # Noble Person and Generals (貴人與天將)
    # =========================================================================

    def get_noble_person_branch(self, day_stem: str, is_day: bool) -> str:
        """
        Determine the Noble Person (貴人) branch position.

        The Noble Person position depends on:
        1. The day stem
        2. Whether it's day or night

        Args:
            day_stem: The day's Heavenly Stem
            is_day: True for daytime, False for nighttime

        Returns:
            The branch where Noble Person is positioned
        """
        if is_day:
            return NOBLE_PERSON_DAY.get(day_stem, '丑')
        else:
            return NOBLE_PERSON_NIGHT.get(day_stem, '未')

    def position_generals(self, noble_branch: str,
                         heaven_plate: HeavenPlate) -> Dict[str, TianJiang]:
        """
        Position all 12 Heavenly Generals on the plate.

        Starting from the Noble Person (貴人) position:
        - Forward sequence: 騰蛇, 朱雀, 六合, 勾陳, 青龍
        - Backward sequence: 天后, 太陰, 玄武, 太常, 白虎, 天空

        Args:
            noble_branch: The branch where 貴人 is placed
            heaven_plate: The Heaven Plate

        Returns:
            Dict mapping branches to positioned generals
        """
        positions: Dict[str, TianJiang] = {}
        noble_idx = get_branch_index(noble_branch)

        # The complete sequence of generals
        general_order = [
            '貴人', '騰蛇', '朱雀', '六合', '勾陳', '青龍',
            '天空', '白虎', '太常', '玄武', '太陰', '天后'
        ]

        # Position each general
        for i, gen_name in enumerate(general_order):
            general = self.twelve_generals.get_by_chinese(gen_name)
            if general:
                # Calculate the branch position
                branch_idx = (noble_idx + i) % 12
                branch = TWELVE_BRANCHES[branch_idx]
                positions[branch] = general

        return positions

    # =========================================================================
    # Hour and Time Calculations (時辰計算)
    # =========================================================================

    def get_hour_branch(self, dt: datetime) -> str:
        """
        Get the Earthly Branch for a given hour.

        Each branch covers a 2-hour period:
        子時: 23:00-01:00
        丑時: 01:00-03:00
        ...and so on

        Args:
            dt: The datetime

        Returns:
            The hour's Earthly Branch
        """
        hour = dt.hour
        return HOUR_TO_BRANCH.get(hour, '子')

    # =========================================================================
    # Main Calculation Entry Point (主計算入口)
    # =========================================================================

    def calculate_plate(self, dt: datetime) -> LiuRenPlate:
        """
        Calculate a complete Liu Ren plate for the given datetime.

        This is the main entry point that orchestrates all calculations.

        Args:
            dt: The datetime for divination

        Returns:
            Complete LiuRenPlate with all calculated elements
        """
        # Step 1: Get lunar date and astronomical data
        julian_day = self.lunar_calendar.gregorian_to_julian(dt)
        lunar_date = self.lunar_calendar.gregorian_to_lunar(dt)
        solar_longitude = self.lunar_calendar.calculate_solar_longitude(julian_day)

        # Step 2: Extract day stem and branch
        day_stem = lunar_date.day_stem.chinese
        day_branch = lunar_date.day_branch.chinese

        # Step 3: Get hour branch
        hour_branch = self.get_hour_branch(dt)

        # Step 4: Determine if daytime or nighttime
        is_day = is_daytime(dt, self.latitude)

        # Step 5: Get monthly general
        general_name, general_branch = self.get_monthly_general(solar_longitude)

        # Step 6: Build plates
        heaven_plate = self.build_heaven_plate(general_branch, hour_branch)
        earth_plate = self.build_earth_plate()

        # Step 7: Build palaces and apply overlays
        palaces = TwelvePalaces()
        for palace in palaces:
            palace.heaven_branch = heaven_plate.get_heaven_branch_at(palace.earth_branch)

        # Step 8: Calculate Four Lessons
        si_ke = self.calculate_si_ke(day_stem, day_branch, heaven_plate)

        # Step 9: Derive Three Transmissions
        san_chuan = self.derive_san_chuan(si_ke, day_stem, heaven_plate)

        # Step 10: Position Noble Person
        noble_branch = self.get_noble_person_branch(day_stem, is_day)
        noble_person = self.twelve_generals.get_by_chinese('貴人')

        # Step 11: Position all generals
        general_positions = self.position_generals(noble_branch, heaven_plate)

        # Assign generals to palaces
        for palace in palaces:
            if palace.earth_branch in general_positions:
                palace.general = general_positions[palace.earth_branch]

        # Step 12: Identify patterns
        special_patterns = []
        if self._check_fu_yin(heaven_plate):
            special_patterns.append('伏吟')
        if self._check_fan_yin(heaven_plate):
            special_patterns.append('反吟')

        # Step 13: Assemble the plate
        plate = LiuRenPlate(
            query_datetime=dt,
            lunar_date=lunar_date,
            day_stem=day_stem,
            day_branch=day_branch,
            hour_branch=hour_branch,
            is_daytime=is_day,
            monthly_general_name=general_name,
            monthly_general_branch=general_branch,
            heaven_plate=heaven_plate,
            earth_plate=earth_plate,
            palaces=palaces,
            si_ke=si_ke,
            san_chuan=san_chuan,
            noble_person_branch=noble_branch,
            noble_person=noble_person,
            general_positions=general_positions,
            lesson_pattern="",  # Will be set by patterns module
            special_patterns=special_patterns,
        )

        return plate

    # =========================================================================
    # Pattern Detection Helpers (模式檢測輔助)
    # =========================================================================

    def _check_fu_yin(self, heaven_plate: HeavenPlate) -> bool:
        """
        Check for 伏吟 pattern.

        伏吟 occurs when all heaven branches equal earth branches
        (i.e., monthly general branch equals hour branch).
        """
        return heaven_plate.monthly_general_branch == heaven_plate.query_hour_branch

    def _check_fan_yin(self, heaven_plate: HeavenPlate) -> bool:
        """
        Check for 反吟 pattern.

        反吟 occurs when all heaven branches clash with earth branches
        (i.e., monthly general and hour branch are in opposition).
        """
        clash_branch = SIX_CLASHES.get(heaven_plate.monthly_general_branch)
        return clash_branch == heaven_plate.query_hour_branch


# =============================================================================
# Convenience Function (便捷函數)
# =============================================================================

def calculate_liu_ren_plate(dt: datetime,
                           lunar_calendar=None,
                           latitude: float = 35.0) -> LiuRenPlate:
    """
    Calculate a Liu Ren plate for the given datetime.

    Convenience function that creates a calculator and calls calculate_plate.

    Args:
        dt: The datetime for divination
        lunar_calendar: Optional ChineseLunarCalendar instance
        latitude: Latitude for sunrise/sunset (default 35°N)

    Returns:
        Complete LiuRenPlate
    """
    calculator = LiuRenCalculator(lunar_calendar, latitude)
    return calculator.calculate_plate(dt)
