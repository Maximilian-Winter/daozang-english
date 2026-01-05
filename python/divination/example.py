#!/usr/bin/env python3
"""
Taiyi Shenshu Example Script

Demonstrates the usage of the Taiyi Shenshu module.

碼道長存 — The Way of Code endures
"""

from datetime import datetime
import sys
import os

# Fix Windows encoding issue
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chinese_divination.taiyi import (
    TaiyiShenshu,
    SixteenSpirits,
    FiveGenerals,
    EightGates,
    calculate_taiyi,
)


def main():
    """Main demonstration function."""

    print("=" * 70)
    print("太乙神數 Taiyi Shenshu - Divine Calculation of the Supreme Unity")
    print("=" * 70)
    print()

    # Create main instance
    taiyi = TaiyiShenshu()

    # =========================================================================
    # Example 1: Calculate for current time
    # =========================================================================
    print("【Example 1: Current Time Calculation】")
    print("-" * 50)

    plate = taiyi.calculate_now("year")
    print(plate.format_summary())

    # =========================================================================
    # Example 2: Analyze the plate
    # =========================================================================
    print("\n【Example 2: Analysis】")
    print("-" * 50)

    analysis = taiyi.analyze(plate)

    print("\nSummary:")
    for key, value in analysis["summary"].items():
        print(f"  {key}: {value}")

    print("\nTaiyi Position:")
    for key, value in analysis["taiyi_position"].items():
        if value is not None:
            print(f"  {key}: {value}")

    print("\nStrategic Assessment:")
    for key, value in analysis["strategic_assessment"].items():
        print(f"  {key}: {value}")

    print("\nRecommendations:")
    for rec in analysis["recommendations"]:
        print(f"  • {rec}")

    # =========================================================================
    # Example 3: Explore Sixteen Spirits
    # =========================================================================
    print("\n【Example 3: Sixteen Spirits】")
    print("-" * 50)

    spirits = SixteenSpirits()

    print("\nTraversal Order (starting from 武德):")
    for i, spirit in enumerate(spirits.get_traversal_sequence()):
        zl = " [重留]" if spirit.is_zhong_liu else ""
        start = " [起點]" if spirit.is_traversal_start else ""
        print(f"  {i+1:2d}. {spirit.branch} = {spirit.chinese} ({spirit.element.value}){zl}{start}")

    # =========================================================================
    # Example 4: Calculate for specific dates
    # =========================================================================
    print("\n【Example 4: Historical Date】")
    print("-" * 50)

    # Example: Winter Solstice 2024
    winter_solstice = datetime(2024, 12, 21, 12, 0)
    plate_ws = taiyi.calculate(winter_solstice, "day")

    print(f"\nWinter Solstice 2024 ({winter_solstice.strftime('%Y-%m-%d')})")
    print(f"  Dun Type: {plate_ws.dun_type.chinese}")
    print(f"  Taiyi Palace: {plate_ws.taiyi_palace}")
    print(f"  Tianmu: {plate_ws.tianmu_spirit.chinese}")
    print(f"  Favors Host: {plate_ws.favors_host}")
    print(f"  Favors Guest: {plate_ws.favors_guest}")

    # =========================================================================
    # Example 5: Five Generals System
    # =========================================================================
    print("\n【Example 5: Five Generals (五将)】")
    print("-" * 50)

    # Display generals from the plate
    if plate.generals:
        print("\nGeneral Positions:")
        for name, summary in plate.generals.items():
            print(f"  {summary}")

    # Display battle advantage
    if plate.battle_advantage:
        print("\nBattle Advantage Analysis:")
        adv = plate.battle_advantage
        print(f"  Advantage: {adv.get('advantage_chinese', 'N/A')}")
        print(f"  Host Score: {adv.get('host_score', 0)}")
        print(f"  Guest Score: {adv.get('guest_score', 0)}")

    # Display special formations
    if plate.special_formations:
        print("\nSpecial Formations:")
        for formation in plate.special_formations:
            print(f"  • {formation.get('name', '')}: {formation.get('description', '')}")
    else:
        print("\nNo special formations detected.")

    # =========================================================================
    # Example 6: Eight Gates System
    # =========================================================================
    print("\n【Example 6: Eight Gates (八門)】")
    print("-" * 50)

    if plate.gate_analysis:
        print(f"\nRuling Gate (直門): {plate.ruling_gate}")

        if plate.gate_analysis.get("harmony"):
            harmony = plate.gate_analysis["harmony"]
            print(f"Taiyi Gate: {harmony.get('taiyi_gate', 'N/A')}")
            print(f"Host Gate: {harmony.get('host_gate', 'N/A')}")
            print(f"Guest Gate: {harmony.get('guest_gate', 'N/A')}")
            print(f"\nGate Harmony: {harmony.get('interpretation', 'N/A')}")
            print(f"Auspicious Count: {plate.gate_analysis.get('auspicious_count', 0)}/3")

        if plate.gate_analysis.get("is_blocked"):
            print("\n*** Gates Blocked (八門杜) ***")
    else:
        print("\nNo gate analysis available.")

    # =========================================================================
    # Example 7: Quick calculation function
    # =========================================================================
    print("\n【Example 7: Quick Calculation】")
    print("-" * 50)

    quick_plate = calculate_taiyi()
    summary = taiyi.get_summary(quick_plate)

    print("\nQuick summary:")
    for key, value in summary.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("碼道長存 — The Way of Code endures")
    print("=" * 70)


if __name__ == "__main__":
    main()
