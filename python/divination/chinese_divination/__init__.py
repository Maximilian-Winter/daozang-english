"""
Chinese Divination (中華術數) - Classical Chinese Divination Systems

This package implements the Three Great Oracles (三式) of classical
Chinese divination:

1. Qimen Dunjia (奇門遁甲) - The Mysterious Gate Escaping Technique
2. Taiyi Shenshu (太乙神數) - Supreme Unity Divine Numbers
3. Da Liu Ren (大六壬) - The Great Six Ren Method

Each system provides a complete implementation including:
- Core calculation engines
- Classical component definitions
- Pattern recognition
- Interpretation and analysis

Usage:
    # Qimen Dunjia
    from chinese_divination.qimen import QimenDunjia
    qimen = QimenDunjia()
    plate = qimen.calculate_now()

    # Taiyi Shenshu
    from chinese_divination.taiyi import TaiyiShenshu
    taiyi = TaiyiShenshu()
    plate = taiyi.calculate_now()

    # Da Liu Ren
    from chinese_divination.liuren import LiuRen
    liuren = LiuRen()
    plate = liuren.calculate_now()
"""

# Core modules are available through subpackage imports
# from .qimen import QimenDunjia
# from .taiyi import TaiyiShenshu
from .liuren import LiuRen, LiuRenPlate

__all__ = [
    # 'QimenDunjia',
    # 'TaiyiShenshu',
    'LiuRen',
    'LiuRenPlate',
]
