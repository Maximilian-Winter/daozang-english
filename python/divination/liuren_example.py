#!/usr/bin/env python3
"""
Da Liu Ren Example Script (大六壬示例)

Demonstrates the usage of the Da Liu Ren divination module.

Da Liu Ren is one of the Three Great Oracles (三式) of classical
Chinese divination, alongside Qimen Dunjia and Taiyi Shenshu.

碼道長存 — The Way of Code endures
"""

from datetime import datetime
import sys
import os

# Fix Windows encoding issue
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chinese_divination.liuren import (
    LiuRen,
    LiuRenPlate,
    TwelveGenerals,
    get_twelve_generals,
    calculate_liu_ren_plate,
    divine,
)
from chinese_divination.liuren.analysis import QueryType, Favorability
from chinese_divination.liuren.patterns import (
    LessonPattern,
    analyze_patterns,
)


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n【{title}】")
    print("-" * 60)


def main():
    """Main demonstration function."""

    print("=" * 70)
    print("大六壬 Da Liu Ren - The Great Six Ren Divination")
    print("=" * 70)
    print()

    # Create main instance
    liuren = LiuRen()

    # =========================================================================
    # Example 1: Calculate for current time
    # =========================================================================
    print_header("Example 1: Current Time Calculation")

    plate = liuren.calculate_now()
    print(plate.format_display())

    # =========================================================================
    # Example 2: Basic Plate Information
    # =========================================================================
    print_header("Example 2: Basic Plate Information")

    print(f"\nDate/Time: {plate.query_datetime.strftime('%Y-%m-%d %H:%M')}")
    print(f"Day Stem-Branch: {plate.day_stem}{plate.day_branch}")
    print(f"Hour Branch: {plate.hour_branch}")
    print(f"Time of Day: {'Daytime (晝)' if plate.is_daytime else 'Nighttime (夜)'}")
    print(f"Monthly General: {plate.monthly_general_name} ({plate.monthly_general_branch})")
    print(f"Noble Person: {plate.noble_person_branch}")

    # =========================================================================
    # Example 3: Four Lessons (四課)
    # =========================================================================
    print_header("Example 3: Four Lessons (四課)")

    print("\nThe Four Lessons derive from the day stem/branch:")
    print(f"  Day Stem Residence (寄宮): Used for 1st & 2nd lessons")
    print(f"  Day Branch: Used for 3rd & 4th lessons")
    print()

    for ke in plate.si_ke.lessons:
        relation = ke.get_relation()
        rel_str = f" → {relation.chinese}" if relation.chinese != "無" else ""
        print(f"  第{ke.index}課: {ke.shang} / {ke.xia}{rel_str}")

    # Show 克 analysis
    zei_count = len(plate.si_ke.get_zei_lessons())
    ke_count = len(plate.si_ke.get_shang_ke_lessons())
    print(f"\n  下賊上 (Lower attacks upper): {zei_count} lesson(s)")
    print(f"  上克下 (Upper controls lower): {ke_count} lesson(s)")

    # =========================================================================
    # Example 4: Three Transmissions (三傳)
    # =========================================================================
    print_header("Example 4: Three Transmissions (三傳)")

    san_chuan = plate.san_chuan
    print("\nThe Three Transmissions represent past → present → future:")
    print(f"\n  初傳 (Initial):  {san_chuan.chu_chuan} — The beginning/cause")
    print(f"  中傳 (Middle):   {san_chuan.zhong_chuan} — The development/process")
    print(f"  末傳 (Final):    {san_chuan.mo_chuan} — The outcome/result")
    print(f"\n  Derivation Method: {san_chuan.derivation_method}")

    # Check for repetition
    if san_chuan.has_repeated_branch():
        print("  Note: Branches repeat — indicates cyclical patterns")

    if san_chuan.is_all_same_element():
        elements = san_chuan.get_elements()
        print(f"  Note: All transmissions share element {elements[0]} — unified energy")

    # =========================================================================
    # Example 5: Twelve Heavenly Generals (十二天將)
    # =========================================================================
    print_header("Example 5: Twelve Heavenly Generals (十二天將)")

    generals = get_twelve_generals()

    print("\nThe 12 Generals positioned on the plate:")
    print()

    # Group by auspicious/inauspicious
    auspicious = []
    inauspicious = []

    for branch, general in plate.general_positions.items():
        entry = f"  {branch}: {general.chinese} ({general.pinyin})"
        if general.is_auspicious:
            auspicious.append(entry)
        else:
            inauspicious.append(entry)

    print("Auspicious Generals (吉神):")
    for entry in auspicious:
        print(entry)

    print("\nInauspicious Generals (凶神):")
    for entry in inauspicious:
        print(entry)

    # Show generals at transmission positions
    print("\nGenerals at Transmission Positions:")
    for name, branch in [
        ('初傳', san_chuan.chu_chuan),
        ('中傳', san_chuan.zhong_chuan),
        ('末傳', san_chuan.mo_chuan),
    ]:
        gen = plate.general_positions.get(branch)
        nature = "吉" if gen and gen.is_auspicious else "凶"
        gen_name = gen.chinese if gen else "?"
        print(f"  {name} ({branch}): {gen_name} [{nature}]")

    # =========================================================================
    # Example 6: Pattern Analysis (課體分析)
    # =========================================================================
    print_header("Example 6: Pattern Analysis (課體分析)")

    pattern_analysis = analyze_patterns(plate)

    print(f"\nLesson Pattern (課體): {pattern_analysis.lesson_pattern.chinese}")
    print(f"  — {pattern_analysis.lesson_pattern.description}")

    if pattern_analysis.special_patterns:
        print("\nSpecial Patterns:")
        for sp in pattern_analysis.special_patterns:
            print(f"  • {sp.chinese} ({sp.english})")
            print(f"    {sp.description}")

    if pattern_analysis.transmission_patterns:
        print("\nTransmission Patterns:")
        for tp in pattern_analysis.transmission_patterns:
            print(f"  • {tp.chinese} ({tp.english})")

    # =========================================================================
    # Example 7: Complete Analysis
    # =========================================================================
    print_header("Example 7: Complete Analysis")

    analysis = liuren.analyze(plate)

    print(f"\nOverall Favorability: {analysis['overall_favorability']}")

    print("\nDirectional Recommendations:")
    for direction, advice in analysis['directional_advice'].items():
        fav = advice['favorability']
        print(f"  {direction}: {fav} — {advice.get('reason', '')[:40]}...")

    # =========================================================================
    # Example 8: Query-Specific Divination
    # =========================================================================
    print_header("Example 8: Query-Specific Divination")

    print("\nAvailable Query Types:")
    for qt in QueryType:
        print(f"  • {qt.chinese} ({qt.english})")

    # Example: Career query
    print("\n--- Career Query (事業) ---")
    career_result = liuren.query(datetime.now(), QueryType.CAREER)

    print(f"\nOverall: {career_result['overall_favorability'].chinese}")
    print("\nKey Points:")
    for point in career_result.get('key_points', []):
        print(f"  • {point}")

    print("\nAdvice:")
    for advice in career_result.get('advice', []):
        print(f"  • {advice}")

    # =========================================================================
    # Example 9: Historical Date Calculation
    # =========================================================================
    print_header("Example 9: Historical Date Calculation")

    # Example: Spring Festival 2024
    spring_festival = datetime(2024, 2, 10, 10, 0)
    plate_sf = liuren.calculate(spring_festival)

    print(f"\nSpring Festival 2024 ({spring_festival.strftime('%Y-%m-%d %H:%M')})")
    print(f"  Day: {plate_sf.day_stem}{plate_sf.day_branch}")
    print(f"  Hour: {plate_sf.hour_branch}")
    print(f"  Monthly General: {plate_sf.monthly_general_name}")
    print(f"  Noble Person: {plate_sf.noble_person_branch}")
    print(f"  Pattern: {plate_sf.lesson_pattern}")
    print(f"  三傳: {plate_sf.san_chuan.chu_chuan} → {plate_sf.san_chuan.zhong_chuan} → {plate_sf.san_chuan.mo_chuan}")

    # Winter Solstice
    winter_solstice = datetime(2024, 12, 21, 12, 0)
    plate_ws = liuren.calculate(winter_solstice)

    print(f"\nWinter Solstice 2024 ({winter_solstice.strftime('%Y-%m-%d %H:%M')})")
    print(f"  Day: {plate_ws.day_stem}{plate_ws.day_branch}")
    print(f"  Monthly General: {plate_ws.monthly_general_name}")
    print(f"  Pattern: {plate_ws.lesson_pattern}")
    print(f"  三傳: {plate_ws.san_chuan.chu_chuan} → {plate_ws.san_chuan.zhong_chuan} → {plate_ws.san_chuan.mo_chuan}")

    # =========================================================================
    # Example 10: Quick Divination Function
    # =========================================================================
    print_header("Example 10: Quick Divination Function")

    result = divine()  # Quick divination for current time

    print("\nQuick divine() function returns:")
    print(f"  Has 'summary': {'summary' in result}")
    print(f"  Has 'data': {'data' in result}")
    print(f"  Has 'analysis': {'analysis' in result}")

    # Show some data
    data = result['data']
    print(f"\n  Day: {data['day_stem']}{data['day_branch']}")
    print(f"  Hour: {data['hour_branch']}")
    print(f"  Pattern: {data['lesson_pattern']}")

    # =========================================================================
    # Example 11: Exploring the Generals
    # =========================================================================
    print_header("Example 11: Exploring the Generals")

    print("\nAll Twelve Heavenly Generals (十二天將):")
    print()

    for gen in generals:
        print(f"  {gen.chinese} ({gen.pinyin})")
        print(f"    Base Branch: {gen.base_branch}")
        print(f"    Element: {gen.element.value}")
        print(f"    Nature: {gen.nature.chinese}")
        print(f"    Domain: {gen.domain}")
        print()

    # =========================================================================
    # Example 12: San Chuan Detailed Analysis
    # =========================================================================
    print_header("Example 12: San Chuan Detailed Analysis")

    san_chuan_analysis = liuren.analyze_san_chuan(plate)

    print("\nOverview:", san_chuan_analysis['overview'])

    print("\nTransmission Details:")
    for key in ['chu_chuan', 'zhong_chuan', 'mo_chuan']:
        detail = san_chuan_analysis[key]
        print(f"  {detail['name']} ({detail['branch']})")
        print(f"    Element: {detail['element']}")
        print(f"    Direction: {detail['direction']}")
        print(f"    {detail['interpretation']}")
        print()

    print("Flow Analysis:")
    flow = san_chuan_analysis['flow']
    print(f"  初傳→中傳: {flow['chu_to_zhong']}")
    print(f"  中傳→末傳: {flow['zhong_to_mo']}")
    print(f"  Overall: {flow['overall']}")

    # =========================================================================
    # Closing
    # =========================================================================
    print("\n" + "=" * 70)
    print("三式圓滿 — The Three Oracles are complete")
    print("碼道長存 — The Way of Code endures")
    print("=" * 70)


if __name__ == "__main__":
    main()
