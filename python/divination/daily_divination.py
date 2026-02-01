#!/usr/bin/env python3
"""
三式合參 Daily Divination - Combined Three Oracles

Simply runs all three divination systems together.

碼道長存 — The Way of Code endures
"""

from datetime import datetime
import sys
import os

# Fix Windows encoding issue
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import bilingual formatters
from chinese_divination.qimen.bilingual_formatter import format_detailed_analysis as format_qimen
from chinese_divination.liuren.bilingual_formatter import format_detailed_analysis as format_liuren
from chinese_divination.taiyi.bilingual_formatter import format_detailed_analysis as format_taiyi


def main():
    print("=" * 70)
    print("三式合參 — The Three Oracles Combined")
    print(f"占時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print()

    # =========================================================================
    # 1. Qimen Dunjia (奇門遁甲)
    # =========================================================================
    print("\n" + "=" * 70)
    print("一、奇門遁甲 Qimen Dunjia")
    print("=" * 70 + "\n")

    from chinese_divination.qimen import QimenDunjia

    qimen = QimenDunjia()
    plate = qimen.calculate(datetime.now())
    print(plate.format_display())

    # Enhanced bilingual analysis
    analysis = qimen.analyze(plate)
    print(format_qimen(analysis))

    # =========================================================================
    # 2. Taiyi Shenshu (太乙神數)
    # =========================================================================
    print("\n" + "=" * 70)
    print("二、太乙神數 Taiyi Shenshu")
    print("=" * 70 + "\n")

    from chinese_divination.taiyi import TaiyiShenshu

    taiyi = TaiyiShenshu()
    plate = taiyi.calculate_now("day")
    print(plate.format_summary())

    # Enhanced bilingual analysis
    analysis = taiyi.analyze(plate)
    print(format_taiyi(analysis))

    # =========================================================================
    # 3. Da Liu Ren (大六壬)
    # =========================================================================
    print("\n" + "=" * 70)
    print("三、大六壬 Da Liu Ren")
    print("=" * 70 + "\n")

    from chinese_divination.liuren import LiuRen

    liuren = LiuRen()
    plate = liuren.calculate_now()
    print(plate.format_display())

    # Enhanced bilingual analysis
    analysis = liuren.analyze(plate)
    print(format_liuren(analysis))

    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 70)
    print("三式圓滿 — The Three Oracles Complete")
    print("碼道長存 — The Way of Code Endures")
    print("=" * 70)


if __name__ == "__main__":
    main()
