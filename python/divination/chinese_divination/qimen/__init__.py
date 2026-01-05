"""
Qimen Dunjia (奇門遁甲) - Strategic Divination Framework

This package implements the complete Qimen Dunjia system,
one of the Three Styles (三式) of Chinese strategic divination.

The system uses five rotating plates to analyze temporal patterns:
- Earth Plate (地盤): Static base with Six Yi and Three Wonders
- Heaven Plate (天盤): Nine Stars rotation
- Human Plate (人盤): Eight Gates rotation
- Spirit Plate (神盤): Eight Spirits rotation
- Door Plate (門盤): Direction analysis

Usage:
    from prognostication.qimen import QimenDunjia

    qimen = QimenDunjia()
    plate = qimen.calculate(datetime.now())
    analysis = qimen.analyze(plate)

    # Export to beautiful HTML
    from prognostication.qimen import export_plate_to_html
    export_plate_to_html(plate, "qimen_reading.html")

Classical sources encoded:
    - 黄帝太一八门逆顺生死诀
    - 黄帝太一八门入式诀
    - 黄帝太一八门入式秘诀

九天玄碼女在此 - The Mysterious Code Lady of the Nine Heavens is here
"""

from .qimen_dunjia import QimenDunjia
from .components import NineStars, EightGates, EightSpirits, NineStar, EightGate, EightSpirit
from .palaces import Palace, NinePalaces, QimenPlate
from .calculator import QimenCalculator
from .analysis import QimenAnalyzer
from .html_export import QimenHTMLExporter, export_plate_to_html, quick_html_export
from .stem_patterns import (
    StemPattern, PatternCategory,
    get_stem_pattern, analyze_stem_combination,
    get_auspicious_patterns, get_inauspicious_patterns,
)

__all__ = [
    # Main API
    'QimenDunjia',
    'QimenCalculator',
    'QimenAnalyzer',

    # Components
    'NineStars',
    'EightGates',
    'EightSpirits',
    'NineStar',
    'EightGate',
    'EightSpirit',

    # Palace structures
    'Palace',
    'NinePalaces',
    'QimenPlate',

    # HTML Export
    'QimenHTMLExporter',
    'export_plate_to_html',
    'quick_html_export',

    # Stem Patterns
    'StemPattern',
    'PatternCategory',
    'get_stem_pattern',
    'analyze_stem_combination',
    'get_auspicious_patterns',
    'get_inauspicious_patterns',
]

__version__ = '1.1.0'
