"""
Taiyi Shenshu (太乙神數) - The Divine Calculation of the Supreme Unity

One of the Three Styles (三式) of Chinese divination, alongside Qimen Dunjia (奇門遁甲)
and Liuren (六壬). Taiyi focuses on grand cycles of fate, nations, and celestial patterns.

This module implements the classical Taiyi Shenshu system based on:
- 太乙金鏡式經 (Tang Dynasty, by Wang Ximing 王希明)
- 太乙秘書 (Song Dynasty, by Wang Zuo 王佐)
- 黃帝太乙八門入式訣 (Daoist Canon)

Core Components:
- Sixteen Spirits (十六神): The celestial entities that govern fate
- Nine Palaces (九宮): The spatial arrangement (Taiyi skips center palace 5)
- Five Generals (五將): Strategic commanders for divination
- Eight Gates (八門): Pathways of fortune and misfortune

Usage:
    from chinese_divination.taiyi import TaiyiShenshu

    taiyi = TaiyiShenshu()
    plate = taiyi.calculate_now()
    print(plate.format_summary())

碼道長存 — The Way of Code endures
"""

from .constants import (
    TaiyiConstants,
    TaiyiDunType,
    TaiyiEra,
    PalaceType,
    PalaceInfo,
    PALACE_DEFINITIONS,
)

from .spirits import (
    SpiritPosition,
    TaiyiSpirit,
    SixteenSpirits,
    SpiritElementRelations,
)

from .palaces import (
    TaiyiPalace,
    NinePalaces,
    TaiyiPlate,
)

from .calculator import TaiyiCalculator

from .generals import (
    GeneralElement,
    GeneralType,
    GeneralState,
    TaiyiGeneral,
    FiveGenerals,
    calculate_generals,
)

from .gates import (
    GateConstants,
    GateType,
    GateNature,
    TaiyiGate,
    GATE_DEFINITIONS,
    EightGates,
    calculate_gates,
)

from .taiyi_shenshu import (
    TaiyiShenshu,
    calculate_taiyi,
)

__all__ = [
    # Main API
    'TaiyiShenshu',
    'calculate_taiyi',
    # Calculator
    'TaiyiCalculator',
    # Constants
    'TaiyiConstants',
    'TaiyiDunType',
    'TaiyiEra',
    'PalaceType',
    'PalaceInfo',
    'PALACE_DEFINITIONS',
    # Spirits
    'SpiritPosition',
    'TaiyiSpirit',
    'SixteenSpirits',
    'SpiritElementRelations',
    # Palaces
    'TaiyiPalace',
    'NinePalaces',
    'TaiyiPlate',
    # Generals
    'GeneralElement',
    'GeneralType',
    'GeneralState',
    'TaiyiGeneral',
    'FiveGenerals',
    'calculate_generals',
    # Gates
    'GateConstants',
    'GateType',
    'GateNature',
    'TaiyiGate',
    'GATE_DEFINITIONS',
    'EightGates',
    'calculate_gates',
]

__version__ = '0.1.0'
__author__ = 'Nine Heavens Mysterious Code Lady (九天玄碼女)'
