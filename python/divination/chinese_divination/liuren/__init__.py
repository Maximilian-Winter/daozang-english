"""
Da Liu Ren (大六壬) - The Great Six Ren Divination System

One of the Three Great Oracles (三式) of classical Chinese divination,
alongside Qimen Dunjia (奇門遁甲) and Taiyi Shenshu (太乙神數).

Da Liu Ren uses a 12-palace system based on the interaction between
Heaven Plate (天盤) and Earth Plate (地盤), deriving Four Lessons (四課)
and Three Transmissions (三傳) for interpretation.

Primary Reference: 六壬大全 (Ming Dynasty, 郭載騋)

Usage:
    from chinese_divination.liuren import LiuRen

    liuren = LiuRen()
    plate = liuren.calculate_now()
    print(liuren.get_summary(plate))

    # Query-specific divination
    from chinese_divination.liuren.analysis import QueryType
    result = liuren.query(datetime.now(), QueryType.CAREER)
"""

# Main API
from .liu_ren import LiuRen, divine

# Core components
from .components import (
    TianJiang,
    TwelveGenerals,
    get_twelve_generals,
    SiKe,
    SanChuan,
    Ke,
    GeneralPosition,
    GeneralNature,
    KeRelation,
)

# Plate structures
from .plates import (
    LiuRenPlate,
    HeavenPlate,
    EarthPlate,
    LiuRenPalace,
    TwelvePalaces,
)

# Calculator
from .calculator import LiuRenCalculator, calculate_liu_ren_plate

# Patterns
from .patterns import (
    LessonPattern,
    SpecialPattern,
    TransmissionPattern,
    PatternAnalysis,
    analyze_patterns,
    identify_lesson_pattern,
    detect_special_patterns,
)

# Analysis
from .analysis import (
    LiuRenAnalyzer,
    QueryType,
    Favorability,
    DirectionalAdvice,
)

__all__ = [
    # Main API
    'LiuRen',
    'divine',

    # Components
    'TianJiang',
    'TwelveGenerals',
    'get_twelve_generals',
    'SiKe',
    'SanChuan',
    'Ke',
    'GeneralPosition',
    'GeneralNature',
    'KeRelation',

    # Plates
    'LiuRenPlate',
    'HeavenPlate',
    'EarthPlate',
    'LiuRenPalace',
    'TwelvePalaces',

    # Calculator
    'LiuRenCalculator',
    'calculate_liu_ren_plate',

    # Patterns
    'LessonPattern',
    'SpecialPattern',
    'TransmissionPattern',
    'PatternAnalysis',
    'analyze_patterns',
    'identify_lesson_pattern',
    'detect_special_patterns',

    # Analysis
    'LiuRenAnalyzer',
    'QueryType',
    'Favorability',
    'DirectionalAdvice',
]
