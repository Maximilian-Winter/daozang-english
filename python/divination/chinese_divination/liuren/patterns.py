"""
Da Liu Ren Pattern Recognition (大六壬課體識別)

Pattern identification for Liu Ren divination.

This module identifies:
- Lesson Body Patterns (課體) - The type of lesson configuration
- Special Patterns (特殊格局) - 伏吟, 反吟, etc.
- Transmission Patterns (傳體) - Patterns in the three transmissions

Reference: 六壬大全 (Ming Dynasty, 郭載騋)
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .constants import (
    BRANCH_ELEMENT, SIX_CLASHES, SIX_HARMONIES,
    does_control, does_generate,
    TWELVE_BRANCHES, get_branch_index,
)
from .components import SiKe, SanChuan, Ke, KeRelation
from .plates import LiuRenPlate


# =============================================================================
# Lesson Pattern Enumeration (課體枚舉)
# =============================================================================

class LessonPattern(Enum):
    """
    The nine standard lesson body patterns (九課體).

    Each pattern describes a specific configuration of the Four Lessons
    and determines the method of deriving the Three Transmissions.
    """
    # Primary patterns
    ZEI_KE = ("賊克課", "Lower attacks upper - 下賊上",
              "The controlled attacks the controller. Use the attacked.")
    KE_ZEI = ("克賊課", "Upper controls lower - 上克下",
              "The controller overcomes the controlled. Use the controller.")

    # Comparative patterns
    BI_YONG = ("比用課", "Same element comparison",
               "No 克 relationship, use elements of same type.")
    SHE_HAI = ("涉害課", "Harm/damage analysis",
               "Multiple 克, select by depth of harm.")
    YAO_KE = ("遙克課", "Distant control",
              "Control from distant positions.")

    # Special patterns
    MAO_XING = ("昴星課", "Pleiades star pattern",
                "Special star-influenced configuration.")
    BIE_ZE = ("別責課", "Distinguished separation",
              "Separate and assign responsibility.")
    BA_ZHUAN = ("八專課", "Eight specializations",
                "Specialized domain pattern.")

    # Extreme patterns
    FAN_YIN = ("返吟課", "Reverse chant",
               "All positions in opposition.")
    FU_YIN = ("伏吟課", "Hidden chant",
              "All positions identical.")

    def __init__(self, chinese: str, english: str, description: str):
        self.chinese = chinese
        self.english = english
        self.description = description


# =============================================================================
# Special Pattern Enumeration (特殊格局枚舉)
# =============================================================================

class SpecialPattern(Enum):
    """Special patterns that modify interpretation."""

    FU_YIN = ("伏吟", "Hidden/Buried",
              "Heaven and Earth branches identical. Stagnation, no movement.")
    FAN_YIN = ("反吟", "Reverse",
               "Heaven and Earth branches in opposition. Conflict, reversal.")

    CHONG_SHEN = ("重審", "Re-examination",
                  "Repeated positions require careful re-examination.")
    SAN_CHUAN_DI = ("三傳遞", "Sequential transmission",
                    "Transmissions follow natural sequence.")

    KONG_WANG = ("空亡", "Void/Empty",
                 "Position falls in void, indicates emptiness.")

    GUI_REN_SHUN = ("貴人順", "Noble Person forward",
                   "Noble Person moves forward - favorable.")
    GUI_REN_NI = ("貴人逆", "Noble Person backward",
                  "Noble Person moves backward - challenging.")

    def __init__(self, chinese: str, english: str, description: str):
        self.chinese = chinese
        self.english = english
        self.description = description


# =============================================================================
# Transmission Pattern Enumeration (傳體枚舉)
# =============================================================================

class TransmissionPattern(Enum):
    """Patterns identified in the Three Transmissions."""

    SHUN_CHUAN = ("順傳", "Forward transmission",
                  "Transmissions move in forward sequence.")
    NI_CHUAN = ("逆傳", "Backward transmission",
                "Transmissions move in backward sequence.")

    YUAN_TAI = ("元胎", "Origin embryo",
                "Transmissions return to origin.")
    GUI_YI = ("歸一", "Return to one",
              "All transmissions converge to single element.")

    QIAN_CHONG = ("前衝", "Forward clash",
                  "Transmissions clash with coming positions.")
    HOU_HE = ("後合", "Backward harmony",
              "Transmissions harmonize with past positions.")

    def __init__(self, chinese: str, english: str, description: str):
        self.chinese = chinese
        self.english = english
        self.description = description


# =============================================================================
# Pattern Identification Functions (模式識別函數)
# =============================================================================

def identify_lesson_pattern(si_ke: SiKe) -> LessonPattern:
    """
    Identify the lesson body pattern (課體) from the Four Lessons.

    Analysis order:
    1. Check for 賊克 (下賊上)
    2. Check for 克賊 (上克下)
    3. Check for multiple 克 (涉害)
    4. Check for 比用 (same element)
    5. Default to 遙克

    Args:
        si_ke: The Four Lessons

    Returns:
        The identified LessonPattern
    """
    zei_lessons = si_ke.get_zei_lessons()
    ke_lessons = si_ke.get_shang_ke_lessons()

    # Check for 賊克 (下賊上) - highest priority
    if zei_lessons:
        if len(zei_lessons) > 1:
            return LessonPattern.SHE_HAI  # Multiple賊 = 涉害
        return LessonPattern.ZEI_KE

    # Check for 克賊 (上克下)
    if ke_lessons:
        if len(ke_lessons) > 1:
            return LessonPattern.SHE_HAI  # Multiple克 = 涉害
        return LessonPattern.KE_ZEI

    # No克 relationship - check for 比用
    if _has_bi_yong(si_ke):
        return LessonPattern.BI_YONG

    # Check for八專 (special day-branch alignment)
    if _is_ba_zhuan(si_ke):
        return LessonPattern.BA_ZHUAN

    # Default to遙克
    return LessonPattern.YAO_KE


def _has_bi_yong(si_ke: SiKe) -> bool:
    """Check if 比用 pattern applies (same elements present)."""
    day_element = BRANCH_ELEMENT.get(si_ke.day_branch, '')

    for ke in si_ke.lessons:
        shang_element = BRANCH_ELEMENT.get(ke.shang, '')
        if shang_element == day_element:
            return True

    return False


def _is_ba_zhuan(si_ke: SiKe) -> bool:
    """
    Check for 八專 pattern.

    八專 occurs when day stem and branch are of the same element,
    creating a specialized configuration.
    """
    from .constants import STEM_ELEMENT

    stem_element = STEM_ELEMENT.get(si_ke.day_stem, '')
    branch_element = BRANCH_ELEMENT.get(si_ke.day_branch, '')

    return stem_element == branch_element


# =============================================================================
# Special Pattern Detection (特殊模式檢測)
# =============================================================================

def detect_special_patterns(plate: LiuRenPlate) -> List[SpecialPattern]:
    """
    Detect all special patterns in a Liu Ren plate.

    Args:
        plate: The complete LiuRenPlate

    Returns:
        List of detected SpecialPattern values
    """
    patterns = []

    # Check 伏吟
    if _is_fu_yin(plate):
        patterns.append(SpecialPattern.FU_YIN)

    # Check 反吟
    if _is_fan_yin(plate):
        patterns.append(SpecialPattern.FAN_YIN)

    # Check 重審
    if _has_chong_shen(plate):
        patterns.append(SpecialPattern.CHONG_SHEN)

    # Check 貴人順/逆
    gui_pattern = _check_gui_ren_direction(plate)
    if gui_pattern:
        patterns.append(gui_pattern)

    return patterns


def _is_fu_yin(plate: LiuRenPlate) -> bool:
    """
    Check for 伏吟 (Hidden Chant) pattern.

    Occurs when heaven plate exactly overlays earth plate
    (monthly general = query hour).
    """
    return plate.monthly_general_branch == plate.hour_branch


def _is_fan_yin(plate: LiuRenPlate) -> bool:
    """
    Check for 反吟 (Reverse Chant) pattern.

    Occurs when heaven plate is in complete opposition to earth plate
    (monthly general and query hour are in 六衝 relationship).
    """
    clash = SIX_CLASHES.get(plate.monthly_general_branch)
    return clash == plate.hour_branch


def _has_chong_shen(plate: LiuRenPlate) -> bool:
    """
    Check for 重審 pattern.

    Occurs when the same branch appears multiple times
    in the Four Lessons or Three Transmissions.
    """
    branches = []

    # Collect all branches from Four Lessons
    for ke in plate.si_ke.lessons:
        branches.extend([ke.shang, ke.xia])

    # Collect from Three Transmissions
    branches.extend(plate.san_chuan.get_all_branches())

    # Check for duplicates
    return len(branches) != len(set(branches))


def _check_gui_ren_direction(plate: LiuRenPlate) -> Optional[SpecialPattern]:
    """
    Check Noble Person (貴人) direction pattern.

    Determines if Noble Person moves forward (順) or backward (逆)
    relative to the query hour.
    """
    noble_idx = get_branch_index(plate.noble_person_branch)
    hour_idx = get_branch_index(plate.hour_branch)

    # Calculate direction
    diff = (noble_idx - hour_idx) % 12

    if diff <= 6:
        return SpecialPattern.GUI_REN_SHUN
    else:
        return SpecialPattern.GUI_REN_NI


# =============================================================================
# Transmission Pattern Detection (傳體檢測)
# =============================================================================

def identify_transmission_pattern(san_chuan: SanChuan) -> List[TransmissionPattern]:
    """
    Identify patterns in the Three Transmissions.

    Args:
        san_chuan: The Three Transmissions

    Returns:
        List of identified TransmissionPattern values
    """
    patterns = []

    # Check direction
    if _is_shun_chuan(san_chuan):
        patterns.append(TransmissionPattern.SHUN_CHUAN)
    elif _is_ni_chuan(san_chuan):
        patterns.append(TransmissionPattern.NI_CHUAN)

    # Check element convergence
    if san_chuan.is_all_same_element():
        patterns.append(TransmissionPattern.GUI_YI)

    # Check for clashes
    if _has_chuan_clash(san_chuan):
        patterns.append(TransmissionPattern.QIAN_CHONG)

    # Check for harmonies
    if _has_chuan_harmony(san_chuan):
        patterns.append(TransmissionPattern.HOU_HE)

    return patterns


def _is_shun_chuan(san_chuan: SanChuan) -> bool:
    """Check if transmissions follow forward sequence."""
    idx1 = get_branch_index(san_chuan.chu_chuan)
    idx2 = get_branch_index(san_chuan.zhong_chuan)
    idx3 = get_branch_index(san_chuan.mo_chuan)

    # Forward if each is 1-3 positions ahead
    diff1 = (idx2 - idx1) % 12
    diff2 = (idx3 - idx2) % 12

    return 1 <= diff1 <= 3 and 1 <= diff2 <= 3


def _is_ni_chuan(san_chuan: SanChuan) -> bool:
    """Check if transmissions follow backward sequence."""
    idx1 = get_branch_index(san_chuan.chu_chuan)
    idx2 = get_branch_index(san_chuan.zhong_chuan)
    idx3 = get_branch_index(san_chuan.mo_chuan)

    # Backward if each is 1-3 positions behind
    diff1 = (idx1 - idx2) % 12
    diff2 = (idx2 - idx3) % 12

    return 1 <= diff1 <= 3 and 1 <= diff2 <= 3


def _has_chuan_clash(san_chuan: SanChuan) -> bool:
    """Check if any transmissions clash with each other."""
    branches = san_chuan.get_all_branches()

    for i, b1 in enumerate(branches):
        for b2 in branches[i+1:]:
            if SIX_CLASHES.get(b1) == b2:
                return True

    return False


def _has_chuan_harmony(san_chuan: SanChuan) -> bool:
    """Check if any transmissions harmonize with each other."""
    branches = san_chuan.get_all_branches()

    for i, b1 in enumerate(branches):
        for b2 in branches[i+1:]:
            if SIX_HARMONIES.get(b1) == b2:
                return True

    return False


# =============================================================================
# Complete Pattern Analysis (完整模式分析)
# =============================================================================

@dataclass
class PatternAnalysis:
    """Complete pattern analysis results."""
    lesson_pattern: LessonPattern
    special_patterns: List[SpecialPattern]
    transmission_patterns: List[TransmissionPattern]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all patterns."""
        return {
            'lesson_pattern': {
                'chinese': self.lesson_pattern.chinese,
                'english': self.lesson_pattern.english,
                'description': self.lesson_pattern.description,
            },
            'special_patterns': [
                {
                    'chinese': p.chinese,
                    'english': p.english,
                    'description': p.description,
                }
                for p in self.special_patterns
            ],
            'transmission_patterns': [
                {
                    'chinese': p.chinese,
                    'english': p.english,
                    'description': p.description,
                }
                for p in self.transmission_patterns
            ],
        }


def analyze_patterns(plate: LiuRenPlate) -> PatternAnalysis:
    """
    Perform complete pattern analysis on a Liu Ren plate.

    Args:
        plate: The complete LiuRenPlate

    Returns:
        PatternAnalysis with all identified patterns
    """
    # Identify lesson pattern
    lesson_pattern = identify_lesson_pattern(plate.si_ke)

    # Override with伏吟/反吟 if applicable
    if _is_fu_yin(plate):
        lesson_pattern = LessonPattern.FU_YIN
    elif _is_fan_yin(plate):
        lesson_pattern = LessonPattern.FAN_YIN

    # Detect special patterns
    special_patterns = detect_special_patterns(plate)

    # Identify transmission patterns
    transmission_patterns = identify_transmission_pattern(plate.san_chuan)

    return PatternAnalysis(
        lesson_pattern=lesson_pattern,
        special_patterns=special_patterns,
        transmission_patterns=transmission_patterns,
    )


def apply_pattern_to_plate(plate: LiuRenPlate) -> None:
    """
    Apply pattern analysis to a plate, updating its pattern fields.

    This modifies the plate in place.

    Args:
        plate: The LiuRenPlate to update
    """
    analysis = analyze_patterns(plate)

    plate.lesson_pattern = analysis.lesson_pattern.chinese
    plate.special_patterns = [p.chinese for p in analysis.special_patterns]
