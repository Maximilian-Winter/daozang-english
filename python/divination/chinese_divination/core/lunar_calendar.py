import math
from datetime import datetime, timedelta
from typing import Tuple, NamedTuple, Dict, List
from enum import Enum

from enum import Enum
from .core import Element, Direction, Polarity, Season


class HeavenlyStem:
    """Represents a single Heavenly Stem with all its attributes"""

    def __init__(self, chinese: str, pinyin: str, element: Element, polarity: Polarity,
                 description: str, strategic_application: str):
        self.chinese = chinese
        self.pinyin = pinyin
        self.element = element
        self.polarity = polarity
        self.description = description
        self.strategic_application = strategic_application

    def __str__(self):
        return f"{self.chinese} ({self.pinyin})"

    def __repr__(self):
        return f"HeavenlyStem({self.chinese}, {self.element.value}, {self.polarity.value})"


class EarthlyBranch:
    """Represents a single Earthly Branch with all its attributes"""

    def __init__(self, chinese: str, pinyin: str, zodiac_chinese: str, zodiac_english: str,
                 time_period: str, hour_range: Tuple[int, int], energy_quality: str,
                 optimal_activities: List[str], avoid_activities: List[str]):
        self.chinese = chinese
        self.pinyin = pinyin
        self.zodiac_chinese = zodiac_chinese
        self.zodiac_english = zodiac_english
        self.time_period = time_period
        self.hour_range = hour_range  # (start_hour, end_hour) in 24-hour format
        self.energy_quality = energy_quality
        self.optimal_activities = optimal_activities
        self.avoid_activities = avoid_activities

    def __str__(self):
        return f"{self.chinese} ({self.pinyin}) - {self.zodiac_english}"

    def __repr__(self):
        return f"EarthlyBranch({self.chinese}, {self.zodiac_english})"

    def is_active_time(self, hour: int) -> bool:
        """Check if given hour falls within this branch's time period"""
        start, end = self.hour_range
        if start <= end:
            return start <= hour < end
        else:  # Crosses midnight (like 23-01)
            return hour >= start or hour < end


class HeavenlyStems:
    """Container class for all ten Heavenly Stems with their attributes"""

    def __init__(self):
        self.stems = [
            HeavenlyStem(
                chinese="甲", pinyin="jiǎ", element=Element.WOOD, polarity=Polarity.YANG,
                description="Towering tree, leadership, pioneering spirit, bold initiative",
                strategic_application="Initiate new projects, take leadership roles"
            ),
            HeavenlyStem(
                chinese="乙", pinyin="yǐ", element=Element.WOOD, polarity=Polarity.YIN,
                description="Flexible vine, adaptation, gentle persistence, diplomatic influence",
                strategic_application="Practice diplomacy, work on gradual improvements"
            ),
            HeavenlyStem(
                chinese="丙", pinyin="bǐng", element=Element.FIRE, polarity=Polarity.YANG,
                description="Blazing sun, brilliant manifestation, public recognition, charismatic authority",
                strategic_application="Engage in public activities, seek recognition"
            ),
            HeavenlyStem(
                chinese="丁", pinyin="dīng", element=Element.FIRE, polarity=Polarity.YIN,
                description="Steady flame, sustained focus, inner illumination, refined culture",
                strategic_application="Focus on cultural refinement, inner development"
            ),
            HeavenlyStem(
                chinese="戊", pinyin="wù", element=Element.EARTH, polarity=Polarity.YANG,
                description="Mountain, stability, reliable foundation, protective strength",
                strategic_application="Build foundations, provide stability for others"
            ),
            HeavenlyStem(
                chinese="己", pinyin="jǐ", element=Element.EARTH, polarity=Polarity.YIN,
                description="Fertile soil, nourishment, transformation, supportive cultivation",
                strategic_application="Nurture relationships, facilitate transformations"
            ),
            HeavenlyStem(
                chinese="庚", pinyin="gēng", element=Element.METAL, polarity=Polarity.YANG,
                description="Sword, decisive action, cutting through obstacles, military precision",
                strategic_application="Make decisive cuts, eliminate obstacles"
            ),
            HeavenlyStem(
                chinese="辛", pinyin="xīn", element=Element.METAL, polarity=Polarity.YIN,
                description="Jewelry, refinement, precious beauty, artistic perfection",
                strategic_application="Refine quality, appreciate beauty"
            ),
            HeavenlyStem(
                chinese="壬", pinyin="rén", element=Element.WATER, polarity=Polarity.YANG,
                description="Ocean, vast potential, deep wisdom, overwhelming force",
                strategic_application="Engage with vast possibilities, deep planning"
            ),
            HeavenlyStem(
                chinese="癸", pinyin="guǐ", element=Element.WATER, polarity=Polarity.YIN,
                description="Dew, subtle influence, gentle nourishment, hidden depth",
                strategic_application="Work subtly, provide hidden support"
            )
        ]

        # Create lookup dictionaries
        self.by_chinese = {stem.chinese: stem for stem in self.stems}
        self.by_index = {i: stem for i, stem in enumerate(self.stems)}

    def get_by_chinese(self, chinese: str) -> HeavenlyStem:
        """Get stem by Chinese character"""
        return self.by_chinese.get(chinese)

    def get_by_index(self, index: int) -> HeavenlyStem:
        """Get stem by index (0-9)"""
        return self.by_index.get(index % 10)

    def get_daily_stem(self, day_number: int) -> HeavenlyStem:
        """Get the heavenly stem for a given day in the 60-day cycle"""
        stem_index = (day_number - 1) % 10
        return self.stems[stem_index]


class EarthlyBranches:
    """Container class for all twelve Earthly Branches with their attributes"""

    def __init__(self):
        self.branches = [
            EarthlyBranch(
                chinese="子", pinyin="zǐ", zodiac_chinese="鼠", zodiac_english="Rat",
                time_period="子時", hour_range=(23, 1),
                energy_quality="Deep yin, hidden potential, new beginnings in darkness",
                optimal_activities=["Deep meditation", "planning", "accessing subconscious wisdom"],
                avoid_activities=["Heavy physical activity", "important decisions requiring yang energy"]
            ),
            EarthlyBranch(
                chinese="丑", pinyin="chǒu", zodiac_chinese="牛", zodiac_english="Ox",
                time_period="丑時", hour_range=(1, 3),
                energy_quality="Yin stabilizing, foundation building, patient endurance",
                optimal_activities=["Detailed work", "persistent effort", "liver detoxification"],
                avoid_activities=["Creative projects requiring inspiration", "social activities"]
            ),
            EarthlyBranch(
                chinese="寅", pinyin="yín", zodiac_chinese="虎", zodiac_english="Tiger",
                time_period="寅時", hour_range=(3, 5),
                energy_quality="Yang birth, courage emerging, brave initiatives",
                optimal_activities=["Spiritual practice", "exercise", "bold planning"],
                avoid_activities=["Timid activities", "excessive caution"]
            ),
            EarthlyBranch(
                chinese="卯", pinyin="mǎo", zodiac_chinese="兔", zodiac_english="Rabbit",
                time_period="卯時", hour_range=(5, 7),
                energy_quality="Yang rising gently, growth, careful advancement",
                optimal_activities=["Gentle exercise", "gradual progress", "diplomatic communication"],
                avoid_activities=["Aggressive actions", "harsh decisions"]
            ),
            EarthlyBranch(
                chinese="辰", pinyin="chén", zodiac_chinese="龙", zodiac_english="Dragon",
                time_period="辰時", hour_range=(7, 9),
                energy_quality="Transformation power, dynamic change, magical potential",
                optimal_activities=["Important transformations", "breakthrough work", "creative projects"],
                avoid_activities=["Routine tasks", "resistance to change"]
            ),
            EarthlyBranch(
                chinese="巳", pinyin="sì", zodiac_chinese="蛇", zodiac_english="Snake",
                time_period="巳時", hour_range=(9, 11),
                energy_quality="Wisdom emerging, intelligent strategy, subtle influence",
                optimal_activities=["Strategic planning", "intellectual work", "subtle negotiations"],
                avoid_activities=["Impulsive actions", "obvious approaches"]
            ),
            EarthlyBranch(
                chinese="午", pinyin="wǔ", zodiac_chinese="马", zodiac_english="Horse",
                time_period="午時", hour_range=(11, 13),
                energy_quality="Peak yang, maximum activity, dynamic movement",
                optimal_activities=["High-energy tasks", "public speaking", "competitive activities"],
                avoid_activities=["Rest", "introspection", "delicate work"]
            ),
            EarthlyBranch(
                chinese="未", pinyin="wèi", zodiac_chinese="羊", zodiac_english="Goat",
                time_period="未時", hour_range=(13, 15),
                energy_quality="Yang declining, group harmony, collective benefit",
                optimal_activities=["Team building", "consensus building", "nurturing others"],
                avoid_activities=["Individual competition", "aggressive self-assertion"]
            ),
            EarthlyBranch(
                chinese="申", pinyin="shēn", zodiac_chinese="猴", zodiac_english="Monkey",
                time_period="申時", hour_range=(15, 17),
                energy_quality="Intelligent adaptation, clever solutions, playful innovation",
                optimal_activities=["Problem-solving", "learning new skills", "adaptive strategies"],
                avoid_activities=["Rigid approaches", "serious formality"]
            ),
            EarthlyBranch(
                chinese="酉", pinyin="yǒu", zodiac_chinese="鸡", zodiac_english="Rooster",
                time_period="酉時", hour_range=(17, 19),
                energy_quality="Precision, punctuality, harvest completion",
                optimal_activities=["Completing tasks", "quality control", "precise work"],
                avoid_activities=["Starting new projects", "imprecise activities"]
            ),
            EarthlyBranch(
                chinese="戌", pinyin="xū", zodiac_chinese="狗", zodiac_english="Dog",
                time_period="戌時", hour_range=(19, 21),
                energy_quality="Loyal protection, security, faithful completion",
                optimal_activities=["Protecting achievements", "security planning", "loyal service"],
                avoid_activities=["Betrayal", "abandoning responsibilities"]
            ),
            EarthlyBranch(
                chinese="亥", pinyin="hài", zodiac_chinese="猪", zodiac_english="Pig",
                time_period="亥時", hour_range=(21, 23),
                energy_quality="Abundant blessing, satisfaction, return to source",
                optimal_activities=["Gratitude practice", "enjoying achievements", "preparing for rest"],
                avoid_activities=["Excessive ambition", "dissatisfaction with current blessings"]
            )
        ]

        # Create lookup dictionaries
        self.by_chinese = {branch.chinese: branch for branch in self.branches}
        self.by_index = {i: branch for i, branch in enumerate(self.branches)}
        self.by_zodiac = {branch.zodiac_english.lower(): branch for branch in self.branches}

    def get_by_chinese(self, chinese: str) -> EarthlyBranch:
        """Get branch by Chinese character"""
        return self.by_chinese.get(chinese)

    def get_by_index(self, index: int) -> EarthlyBranch:
        """Get branch by index (0-11)"""
        return self.by_index.get(index % 12)

    def get_by_hour(self, hour: int) -> EarthlyBranch:
        """Get the earthly branch for a given hour (0-23)"""
        for branch in self.branches:
            if branch.is_active_time(hour):
                return branch
        return None

    def get_by_zodiac(self, zodiac: str) -> EarthlyBranch:
        """Get branch by zodiac animal name (English)"""
        return self.by_zodiac.get(zodiac.lower())


class LunarDate(NamedTuple):
    year: int
    month: int
    day: int
    is_leap_month: bool
    cycle: int
    year_in_cycle: int
    year_stem: HeavenlyStem
    year_branch: EarthlyBranch
    month_stem: HeavenlyStem
    month_branch: EarthlyBranch
    day_stem: HeavenlyStem
    day_branch: EarthlyBranch
    hour_stem: HeavenlyStem
    hour_branch: EarthlyBranch


class ChineseLunarCalendar:
    def __init__(self):
        # Base epoch: January 31, 1900 00:00 UTC (lunar new year 1900)
        self.base_date = datetime(1900, 1, 31)
        self.base_julian = self.gregorian_to_julian(self.base_date)

        # Astronomical constants
        self.LUNAR_MONTH = 29.530588853  # Mean synodic month in days
        self.TROPICAL_YEAR = 365.24219878  # Tropical year in days

        # Initialize the stem and branch systems
        self.heavenly_stems = HeavenlyStems()
        self.earthly_branches = EarthlyBranches()

        # Load mechanism nodes mapping from JSON if available.
        # The JSON file contains details for each stem-branch combination (e.g. "甲子").
        # It is expected to reside in the same directory as this module.
        self.mechanism_nodes = {}
        try:
            import json
            import os
            # Determine the path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "complete_mechanism_nodes.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    self.mechanism_nodes = json.load(f)
        except Exception:
            # Silently ignore errors; mechanism nodes remain empty
            self.mechanism_nodes = {}

    def get_mechanism_node(self, stem: HeavenlyStem, branch: EarthlyBranch) -> Dict:
        """
        Retrieve mechanism node information for a given stem-branch combination.

        The stem and branch are combined into a key (e.g. "甲子") to look up in the
        loaded mechanism_nodes dictionary. If no entry exists, returns an empty dict.

        :param stem: HeavenlyStem instance
        :param branch: EarthlyBranch instance
        :return: Dictionary with keys 'name', 'characteristic', 'application_method', 'victory_point'
                 or an empty dict if not found.
        """
        key = f"{stem.chinese}{branch.chinese}"
        return self.mechanism_nodes.get(key, {})

    def gregorian_to_julian(self, date: datetime) -> float:
        """Convert Gregorian date to Julian Day Number"""
        a = (14 - date.month) // 12
        y = date.year + 4800 - a
        m = date.month + 12 * a - 3

        jdn = date.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045

        # Add fractional part for time
        fraction = (date.hour + date.minute / 60 + date.second / 3600) / 24
        return jdn + fraction - 0.5

    def julian_to_gregorian(self, jd: float) -> datetime:
        """Convert Julian Day Number to Gregorian date"""
        jd = jd + 0.5
        z = int(jd)
        f = jd - z

        if z < 2299161:
            a = z
        else:
            alpha = int((z - 1867216.25) / 36524.25)
            a = z + 1 + alpha - alpha // 4

        b = a + 1524
        c = int((b - 122.1) / 365.25)
        d = int(365.25 * c)
        e = int((b - d) / 30.6001)

        day = b - d - int(30.6001 * e)
        month = e - 1 if e < 14 else e - 13
        year = c - 4716 if month > 2 else c - 4715

        # Convert fraction back to time
        hours = f * 24
        hour = int(hours)
        minutes = (hours - hour) * 60
        minute = int(minutes)
        second = int((minutes - minute) * 60)

        return datetime(year, month, day, hour, minute, second)

    def calculate_new_moon(self, k: int) -> float:
        """
        Calculate the Julian Day of the k-th new moon after January 6, 2000
        Using Jean Meeus's algorithm
        """
        T = k / 1236.85  # Time in Julian centuries from J2000.0

        # Mean new moon
        JDE = 2451550.09766 + 29.530588861 * k + 0.00015437 * T ** 2 - 0.000000150 * T ** 3 + 0.00000000073 * T ** 4

        # Sun's mean anomaly
        M = 2.5534 + 29.10535670 * k - 0.0000014 * T ** 2 - 0.00000011 * T ** 3

        # Moon's mean anomaly
        Mprime = 201.5643 + 385.81693528 * k + 0.0107582 * T ** 2 + 0.00001238 * T ** 3 - 0.000000058 * T ** 4

        # Moon's argument of latitude
        F = 160.7108 + 390.67050284 * k - 0.0016118 * T ** 2 - 0.00000227 * T ** 3 + 0.000000011 * T ** 4

        # Longitude of ascending node
        Omega = 124.7746 - 1.56375588 * k + 0.0020672 * T ** 2 + 0.00000215 * T ** 3

        # Convert to radians
        M = math.radians(M)
        Mprime = math.radians(Mprime)
        F = math.radians(F)
        Omega = math.radians(Omega)

        # Periodic corrections
        correction = (-0.40720 * math.sin(Mprime) +
                      0.17241 * math.sin(M) +
                      0.01608 * math.sin(2 * Mprime) +
                      0.01039 * math.sin(2 * F) +
                      0.00739 * math.sin(Mprime - M) +
                      -0.00514 * math.sin(Mprime + M) +
                      0.00208 * math.sin(2 * M) +
                      -0.00111 * math.sin(Mprime - 2 * F) +
                      -0.00057 * math.sin(Mprime + 2 * F) +
                      0.00056 * math.sin(2 * Mprime + M) +
                      -0.00042 * math.sin(3 * Mprime) +
                      0.00042 * math.sin(M + 2 * F) +
                      0.00038 * math.sin(M - 2 * F) +
                      -0.00024 * math.sin(2 * Mprime - M) +
                      -0.00017 * math.sin(Omega) +
                      -0.00007 * math.sin(Mprime + 2 * M) +
                      0.00004 * math.sin(2 * Mprime - 2 * F) +
                      0.00004 * math.sin(3 * M) +
                      0.00003 * math.sin(Mprime + M - 2 * F) +
                      0.00003 * math.sin(2 * Mprime + 2 * F) +
                      -0.00003 * math.sin(Mprime + M + 2 * F) +
                      0.00003 * math.sin(Mprime - M + 2 * F) +
                      -0.00002 * math.sin(Mprime - M - 2 * F) +
                      -0.00002 * math.sin(3 * Mprime + M) +
                      0.00002 * math.sin(4 * Mprime))

        return JDE + correction

    def calculate_solar_longitude(self, jd: float) -> float:
        """Calculate the solar longitude for a given Julian Day"""
        T = (jd - 2451545.0) / 36525.0  # Julian centuries from J2000.0

        # Mean longitude of the Sun
        L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T ** 2

        # Mean anomaly of the Sun
        M = 357.52911 + 35999.05029 * T - 0.0001537 * T ** 2
        M = math.radians(M)

        # Equation of center
        C = (1.914602 - 0.004817 * T - 0.000014 * T ** 2) * math.sin(M) + \
            (0.019993 - 0.000101 * T) * math.sin(2 * M) + \
            0.000289 * math.sin(3 * M)

        # True longitude
        true_longitude = L0 + C

        # Reduce to 0-360 range
        return true_longitude % 360

    def find_solar_term(self, longitude: float, start_jd: float) -> float:
        """Find the Julian Day when the sun reaches a specific longitude"""
        jd = start_jd

        for _ in range(40):  # Maximum iterations
            current_longitude = self.calculate_solar_longitude(jd)
            diff = (longitude - current_longitude) % 360
            if diff > 180:
                diff -= 360

            if abs(diff) < 0.0001:  # Precision threshold
                break

            # Approximate rate: 1 degree per day
            jd += diff * 0.985

        return jd

    def is_leap_month_year(self, lunar_year: int) -> Tuple[bool, int]:
        """
        Determine if a lunar year has a leap month and which month
        A leap month is inserted when there are 13 new moons between
        two winter solstices
        """
        # Find winter solstice of the year and next year
        winter_solstice_1 = self.find_solar_term(270, self.base_julian + (lunar_year - 1900) * 365.25)
        winter_solstice_2 = self.find_solar_term(270, winter_solstice_1 + 365)

        # Count new moons between winter solstices
        k_start = round((winter_solstice_1 - 2451550.09766) / 29.530588861)
        k_end = round((winter_solstice_2 - 2451550.09766) / 29.530588861)

        new_moon_count = k_end - k_start

        if new_moon_count == 13:
            # Find which month lacks a major solar term (zhongqi)
            # Major solar terms are at longitudes: 330, 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300
            major_terms = [330, 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

            for month in range(1, 13):
                k = k_start + month
                month_start = self.calculate_new_moon(k)
                month_end = self.calculate_new_moon(k + 1)

                # Check if this month contains a major solar term
                has_major_term = False
                for term_longitude in major_terms:
                    term_jd = self.find_solar_term(term_longitude, month_start)
                    if month_start <= term_jd < month_end:
                        has_major_term = True
                        break

                if not has_major_term:
                    return True, month

        return True, 12  # Default to 12th month if no clear candidate

    def get_stem_branch_pair(self, number: int) -> Tuple[HeavenlyStem, EarthlyBranch]:
        """
        Get heavenly stem and terrestrial branch for a given number
        The sexagenary cycle combines 10 stems with 12 branches
        """
        stem = self.heavenly_stems.get_by_index((number - 1) % 10)
        branch = self.earthly_branches.get_by_index((number - 1) % 12)
        return stem, branch

    def get_year_stem_branch(self, lunar_year: int) -> Tuple[HeavenlyStem, EarthlyBranch]:
        """
        Calculate the heavenly stem and terrestrial branch for a lunar year
        The reference point is 1864 (甲子年 - Jiazi year)
        """
        # 1864 is year 1 of the 60-year cycle (甲子)
        year_number = ((lunar_year - 1864) % 60) + 1
        return self.get_stem_branch_pair(year_number)

    def get_month_stem_branch(self, lunar_year: int, lunar_month: int, is_leap: bool) -> Tuple[
        HeavenlyStem, EarthlyBranch]:
        """
        Calculate the heavenly stem and terrestrial branch for a lunar month
        Month stems follow a pattern based on the year's stem
        """
        # Get the year's stem index
        year_stem_index = ((lunar_year - 1864) % 60) % 10

        # Month stem calculation based on year stem
        # The pattern repeats every 5 years for stems, every year for branches
        month_stem_base = {
            0: 2,  # 甲年 starts with 丙寅
            1: 4,  # 乙年 starts with 戊寅
            2: 6,  # 丙年 starts with 庚寅
            3: 8,  # 丁年 starts with 壬寅
            4: 0,  # 戊年 starts with 甲寅
            5: 2,  # 己年 starts with 丙寅
            6: 4,  # 庚年 starts with 戊寅
            7: 6,  # 辛年 starts with 庚寅
            8: 8,  # 壬年 starts with 壬寅
            9: 0  # 癸年 starts with 甲寅
        }

        base_stem = month_stem_base[year_stem_index]

        # Adjust for leap months (leap months use the same stem/branch as the month they follow)
        month_offset = lunar_month - 1
        if is_leap:
            month_offset -= 0.5  # Leap month has same designation as regular month

        stem = self.heavenly_stems.get_by_index((base_stem + int(month_offset)) % 10)
        branch = self.earthly_branches.get_by_index((2 + int(month_offset)) % 12)  # Start from 寅 (index 2)

        return stem, branch

    def get_day_stem_branch(self, julian_day: float) -> Tuple[HeavenlyStem, EarthlyBranch]:
        """
        Calculate the heavenly stem and terrestrial branch for a day
        Based on the continuous 60-day cycle from a known reference point
        """
        # Reference: January 1, 1900 was 甲戌 day (stem=0, branch=10)
        # Julian day for January 1, 1900
        ref_jd = self.gregorian_to_julian(datetime(1900, 1, 1))

        # Calculate days since reference
        days_since_ref = int(julian_day - ref_jd)

        # Reference day was 甲戌 (stem=0, branch=10), which is day 11 in the 60-day cycle
        day_number = (11 + days_since_ref) % 60
        if day_number == 0:
            day_number = 60

        return self.get_stem_branch_pair(day_number)

    def get_hour_stem_branch(self, date: datetime, day_stem: HeavenlyStem) -> Tuple[HeavenlyStem, EarthlyBranch]:
        """
        Calculate the heavenly stem and terrestrial branch for an hour
        Hour branches are fixed (12 two-hour periods), but stems depend on the day's stem
        """
        # Get the hour branch based on time
        hour_branch = self.earthly_branches.get_by_hour(date.hour)

        # Calculate hour stem based on day stem
        # The pattern is based on the day's stem and follows a specific formula
        day_stem_index = self.heavenly_stems.stems.index(day_stem)

        # Hour stem calculation: depends on day stem and hour branch
        hour_stem_base = {
            0: 0,  # 甲日 starts with 甲子时
            1: 2,  # 乙日 starts with 丙子时
            2: 4,  # 丙日 starts with 戊子时
            3: 6,  # 丁日 starts with 庚子时
            4: 8,  # 戊日 starts with 壬子时
            5: 0,  # 己日 starts with 甲子时
            6: 2,  # 庚日 starts with 丙子时
            7: 4,  # 辛日 starts with 戊子时
            8: 6,  # 壬日 starts with 庚子时
            9: 8  # 癸日 starts with 壬子时
        }

        base_stem = hour_stem_base[day_stem_index]
        branch_index = self.earthly_branches.branches.index(hour_branch)
        hour_stem = self.heavenly_stems.get_by_index((base_stem + branch_index) % 10)

        return hour_stem, hour_branch

    def gregorian_to_lunar(self, date: datetime) -> LunarDate:
        """Convert a Gregorian date to Chinese lunar date"""
        if date.year < 1940:
            raise ValueError("This implementation only supports dates from 1940 onwards")

        target_jd = self.gregorian_to_julian(date)

        # Find the lunar year by finding the winter solstice before the target date
        approx_year = date.year
        winter_solstice = self.find_solar_term(270, target_jd - 180)

        # The lunar year starts with the new moon nearest to the winter solstice
        k_winter = round((winter_solstice - 2451550.09766) / 29.530588861)

        # Find new year (first new moon after winter solstice)
        lunar_new_year = self.calculate_new_moon(k_winter)
        if lunar_new_year > target_jd:
            # Target date is before this lunar new year
            lunar_new_year = self.calculate_new_moon(k_winter - 12)
            lunar_year = approx_year - 1
        else:
            lunar_year = approx_year

        # Check for leap month in this year
        has_leap, leap_month_num = self.is_leap_month_year(lunar_year)

        # Find which lunar month the target date falls in
        month = 1
        is_leap_month = False
        k = round((lunar_new_year - 2451550.09766) / 29.530588861)

        while True:
            month_start = self.calculate_new_moon(k)
            month_end = self.calculate_new_moon(k + 1)

            if month_start <= target_jd < month_end:
                # Found the month
                day = int(target_jd - month_start) + 1
                break

            k += 1
            if has_leap and month == leap_month_num and not is_leap_month:
                is_leap_month = True
            else:
                month += 1
                is_leap_month = False

            if month > 12:
                month = 1
                lunar_year += 1
                has_leap, leap_month_num = self.is_leap_month_year(lunar_year)

        # Calculate sexagenary cycle (60-year cycle)
        # The first year of the cycle is 1864 (Jiazi year)
        cycle_start_year = 1864
        years_since_start = lunar_year - cycle_start_year
        cycle = (years_since_start // 60) + 1
        year_in_cycle = (years_since_start % 60) + 1

        # Calculate stems and branches
        year_stem, year_branch = self.get_year_stem_branch(lunar_year)
        month_stem, month_branch = self.get_month_stem_branch(lunar_year, month, is_leap_month)
        day_stem, day_branch = self.get_day_stem_branch(target_jd)
        hour_stem, hour_branch = self.get_hour_stem_branch(date, day_stem)

        return LunarDate(lunar_year, month, day, is_leap_month, cycle, year_in_cycle,
                         year_stem, year_branch, month_stem, month_branch,
                         day_stem, day_branch, hour_stem, hour_branch)

    def lunar_to_gregorian(self, lunar_date: LunarDate) -> datetime:
        """Convert a Chinese lunar date to Gregorian date"""
        if lunar_date.year < 1940:
            raise ValueError("This implementation only supports dates from 1940 onwards")

        # Find the lunar new year for the given year
        winter_solstice = self.find_solar_term(270, self.base_julian + (lunar_date.year - 1900) * 365.25)
        k_winter = round((winter_solstice - 2451550.09766) / 29.530588861)
        lunar_new_year = self.calculate_new_moon(k_winter)

        # Check for leap month in this year
        has_leap, leap_month_num = self.is_leap_month_year(lunar_date.year)

        # Calculate which new moon corresponds to the target month
        k = round((lunar_new_year - 2451550.09766) / 29.530588861)
        current_month = 1

        while current_month < lunar_date.month or (current_month == lunar_date.month and lunar_date.is_leap_month):
            k += 1
            if has_leap and current_month == leap_month_num:
                if not lunar_date.is_leap_month or current_month != lunar_date.month:
                    k += 1  # Skip the leap month
            current_month += 1

        month_start = self.calculate_new_moon(k)
        target_jd = month_start + lunar_date.day - 1

        return self.julian_to_gregorian(target_jd)

    def get_element_attributes(self, element: Element) -> Dict:
        """Get the Five Element attributes for a given element"""
        element_attributes = {
            Element.WOOD: {
                "quality": "Growth, expansion, flexibility, creativity",
                "season": Season.SPRING,
                "direction": Direction.EAST,
                "time": "Dawn (3-7 AM)",
                "emotion": "Benevolence, planning",
                "organ_system": "Liver/Gallbladder",
                "strategic_application": "Initiation, new projects, creative breakthrough"
            },
            Element.FIRE: {
                "quality": "Manifestation, activity, joy, communication",
                "season": Season.SUMMER,
                "direction": Direction.SOUTH,
                "time": "Noon (11 AM-1 PM)",
                "emotion": "Joy, enthusiasm",
                "organ_system": "Heart/Small Intestine",
                "strategic_application": "Peak activity, public presentation, relationship building"
            },
            Element.EARTH: {
                "quality": "Stability, nourishment, transformation, centering",
                "season": Season.LATE_SUMMER,
                "direction": Direction.CENTER,
                "time": "Transitions between other times",
                "emotion": "Thoughtfulness, empathy",
                "organ_system": "Spleen/Stomach",
                "strategic_application": "Consolidation, resource management, team building"
            },
            Element.METAL: {
                "quality": "Refinement, precision, letting go, harvest",
                "season": Season.AUTUMN,
                "direction": Direction.WEST,
                "time": "Evening (5-7 PM)",
                "emotion": "Righteousness, discernment",
                "organ_system": "Lungs/Large Intestine",
                "strategic_application": "Completion, quality control, elimination of non-essentials"
            },
            Element.WATER: {
                "quality": "Depth, wisdom, storage, potential",
                "season": Season.WINTER,
                "direction": Direction.NORTH,
                "time": "Midnight (11 PM-1 AM)",
                "emotion": "Wisdom, will",
                "organ_system": "Kidneys/Bladder",
                "strategic_application": "Deep planning, resource conservation, foundational work"
            }
        }
        return element_attributes.get(element, {})

    def analyze_lunar_date(self, lunar_date: LunarDate) -> Dict:
        """Provide detailed analysis of a lunar date for divination purposes"""
        analysis = {
            "basic_info": {
                "lunar_year": lunar_date.year,
                "lunar_month": lunar_date.month,
                "lunar_day": lunar_date.day,
                "is_leap_month": lunar_date.is_leap_month,
                "cycle": lunar_date.cycle,
                "year_in_cycle": lunar_date.year_in_cycle
            },
            "year_pillar": {
                "stem": lunar_date.year_stem,
                "branch": lunar_date.year_branch,
                "element": lunar_date.year_stem.element,
                "polarity": lunar_date.year_stem.polarity,
                "zodiac": lunar_date.year_branch.zodiac_english,
                "element_attributes": self.get_element_attributes(lunar_date.year_stem.element)
            },
            "month_pillar": {
                "stem": lunar_date.month_stem,
                "branch": lunar_date.month_branch,
                "element": lunar_date.month_stem.element,
                "polarity": lunar_date.month_stem.polarity,
                "zodiac": lunar_date.month_branch.zodiac_english,
                "element_attributes": self.get_element_attributes(lunar_date.month_stem.element)
            },
            "day_pillar": {
                "stem": lunar_date.day_stem,
                "branch": lunar_date.day_branch,
                "element": lunar_date.day_stem.element,
                "polarity": lunar_date.day_stem.polarity,
                "zodiac": lunar_date.day_branch.zodiac_english,
                "element_attributes": self.get_element_attributes(lunar_date.day_stem.element)
            },
            "hour_pillar": {
                "stem": lunar_date.hour_stem,
                "branch": lunar_date.hour_branch,
                "element": lunar_date.hour_stem.element,
                "polarity": lunar_date.hour_stem.polarity,
                "zodiac": lunar_date.hour_branch.zodiac_english,
                "time_period": lunar_date.hour_branch.time_period,
                "energy_quality": lunar_date.hour_branch.energy_quality,
                "optimal_activities": lunar_date.hour_branch.optimal_activities,
                "avoid_activities": lunar_date.hour_branch.avoid_activities,
                "element_attributes": self.get_element_attributes(lunar_date.hour_stem.element)
            },
            "elemental_composition": self._analyze_elemental_composition(lunar_date),
            "recommendations": self._generate_recommendations(lunar_date)
        }
        return analysis

    def _analyze_elemental_composition(self, lunar_date: LunarDate) -> Dict:
        """Analyze the elemental composition of the four pillars"""
        elements = [
            lunar_date.year_stem.element,
            lunar_date.month_stem.element,
            lunar_date.day_stem.element,
            lunar_date.hour_stem.element
        ]

        element_count = {}
        for element in elements:
            element_count[element] = element_count.get(element, 0) + 1

        # Determine dominant and weak elements
        max_count = max(element_count.values())
        min_count = min(element_count.values())

        dominant_elements = [elem for elem, count in element_count.items() if count == max_count]
        weak_elements = [elem for elem, count in element_count.items() if count == min_count]

        return {
            "element_distribution": element_count,
            "dominant_elements": dominant_elements,
            "weak_elements": weak_elements,
            "balance_assessment": "Balanced" if max_count - min_count <= 1 else "Imbalanced"
        }

    def _generate_recommendations(self, lunar_date: LunarDate) -> Dict:
        """Generate strategic recommendations based on the lunar date analysis"""
        hour_branch = lunar_date.hour_branch
        day_stem = lunar_date.day_stem

        return {
            "time_based": {
                "optimal_for": hour_branch.optimal_activities,
                "avoid": hour_branch.avoid_activities,
                "energy_quality": hour_branch.energy_quality
            },
            "element_based": {
                "daily_stem_guidance": day_stem.strategic_application,
                "element_focus": day_stem.element.value,
                "polarity_advice": f"Emphasize {day_stem.polarity.value} energy today"
            }
        }


# Example usage and testing
def main():
    converter = ChineseLunarCalendar()

    # Test with some known dates including specific times
    test_dates = [
        datetime(1958, 7, 3, 1, 15),  # New Year's Day
        datetime(2024, 2, 10, 6, 0),  # Chinese New Year 2024
        datetime(2024, 7, 13, 15, 45),  # Current date example
    ]

    print("Enhanced Chinese Lunar Calendar with Divination Analysis")
    print("=" * 80)

    for date in test_dates:
        lunar = converter.gregorian_to_lunar(date)
        analysis = converter.analyze_lunar_date(lunar)

        print(f"\n{date.strftime('%Y-%m-%d %H:%M')} ->")
        print(f"Lunar: Year {lunar.year}, Month {lunar.month}", end="")
        if lunar.is_leap_month:
            print(" (Leap)", end="")
        print(f", Day {lunar.day}")

        print(f"\nFour Pillars (四柱):")
        print(f"  Year:  {lunar.year_stem} {lunar.year_branch} ({lunar.year_branch.zodiac_english})")
        print(f"  Month: {lunar.month_stem} {lunar.month_branch}")
        print(f"  Day:   {lunar.day_stem} {lunar.day_branch}")
        print(f"  Hour:  {lunar.hour_stem} {lunar.hour_branch} ({lunar.hour_branch.time_period})")

        print(f"\nElemental Composition:")
        for element, count in analysis["elemental_composition"]["element_distribution"].items():
            print(f"  {element.value}: {count}")

        print(f"\nHour Period Analysis:")
        print(f"  Energy: {lunar.hour_branch.energy_quality}")
        print(f"  Optimal: {', '.join(lunar.hour_branch.optimal_activities[:2])}...")
        print(f"  Avoid: {', '.join(lunar.hour_branch.avoid_activities[:2])}...")

        print(f"\nDaily Stem Guidance:")
        print(f"  {lunar.day_stem.strategic_application}")

        # Test reverse conversion
        gregorian_back = converter.lunar_to_gregorian(lunar)
        print(f"\nReverse conversion: {gregorian_back.strftime('%Y-%m-%d')}")
        print("-" * 60)


if __name__ == "__main__":
    main()