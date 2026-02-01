"""
Qimen Dunjia Bilingual Analysis Formatter (奇門遁甲雙語分析格式化器)

Provides detailed, bilingual (Chinese/English) output formatting for
Qimen Dunjia analysis results.

As Zhuge Liang once said: "觀其陣法,便知其弱點"
(Observe the formation, see its weaknesses)

碼道長存 — The Way of Code Endures
"""

from typing import Dict, Any, List


def format_detailed_analysis(analysis: Dict[str, Any]) -> str:
    """
    Format comprehensive Qimen Dunjia analysis in bilingual format.

    Args:
        analysis: Analysis dictionary from QimenAnalyzer.analyze_plate()

    Returns:
        Beautifully formatted bilingual string
    """
    lines = []

    # Configuration Header
    lines.append("\n┌─────────────────────────────────────────────────────────────┐")
    lines.append("│  盤局配置 Plate Configuration                                │")
    lines.append("└─────────────────────────────────────────────────────────────┘")

    config = analysis['configuration']
    lines.append(f"  遁法 Dun Method: {config['dun_type']}")
    lines.append(f"  時元 Yuan Period: {config['yuan']}")
    lines.append(f"  局數 Ju Number: {config['ju_number']}")
    lines.append(f"  節氣 Solar Term: {config['solar_term']}")

    # Duty Elements
    duty = analysis['duty_elements']
    lines.append(f"\n  值符 Duty Star: {duty['star']}")
    lines.append(f"  值使 Duty Gate: {duty['gate']}")
    lines.append(f"  值符宮 Duty Palace: {duty['palace']}")

    # Overall Assessment
    lines.append("\n┌─────────────────────────────────────────────────────────────┐")
    lines.append("│  整體評斷 Overall Assessment                                 │")
    lines.append("└─────────────────────────────────────────────────────────────┘")

    overall = analysis['overall_assessment']
    lines.append(f"\n  【 {overall['rating']} 】")
    lines.append(f"  {overall['advice']}")
    lines.append(f"\n  綜合評分 Overall Score: {overall['score']}")
    lines.append(f"  吉方數量 Auspicious Directions: {overall['favorable_directions']}")
    lines.append(f"  凶方數量 Inauspicious Directions: {overall['unfavorable_directions']}")

    if overall['three_wonders_present'] > 0:
        lines.append(f"  ✨ 三奇 Three Wonders Present: {overall['three_wonders_present']}")

    if overall['auspicious_patterns'] > 0:
        lines.append(f"  吉格 Auspicious Patterns: {overall['auspicious_patterns']}")

    if overall['inauspicious_patterns'] > 0:
        lines.append(f"  ⚠️ 凶格 Inauspicious Patterns: {overall['inauspicious_patterns']}")

    # Special Notes
    if overall.get('special_notes'):
        lines.append("\n  特別注意 Special Notes:")
        for note in overall['special_notes']:
            lines.append(f"    • {note}")

    # Auspicious Directions
    if analysis.get('auspicious'):
        lines.append("\n┌─────────────────────────────────────────────────────────────┐")
        lines.append("│  吉方推薦 Auspicious Directions                             │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        for i, dir_info in enumerate(analysis['auspicious'][:5], 1):  # Top 5
            lines.append(f"\n  【第{i}吉方 Direction #{i}】")
            lines.append(f"  方位 Direction: {dir_info['direction']}")
            lines.append(f"  宮位 Palace: {dir_info['palace_number']}")
            lines.append(f"  評分 Score: {dir_info['score']}")
            lines.append(f"  星 Star: {dir_info['star']}")
            lines.append(f"  門 Gate: {dir_info['gate']}")

            if dir_info.get('factors'):
                lines.append("  吉兆 Auspicious Factors:")
                for factor in dir_info['factors']:
                    lines.append(f"    ✓ {factor}")

    # Inauspicious Directions
    if analysis.get('inauspicious'):
        lines.append("\n┌─────────────────────────────────────────────────────────────┐")
        lines.append("│  凶方警示 Directions to Avoid                               │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        for i, dir_info in enumerate(analysis['inauspicious'][:3], 1):  # Top 3 worst
            lines.append(f"\n  【凶方 {i}】")
            lines.append(f"  方位 Direction: {dir_info['direction']}")
            lines.append(f"  宮位 Palace: {dir_info['palace_number']}")
            lines.append(f"  評分 Score: {dir_info['score']}")
            lines.append(f"  星 Star: {dir_info['star']}")
            lines.append(f"  門 Gate: {dir_info['gate']}")

            if dir_info.get('warnings'):
                lines.append("  ⚠️ 警示 Warnings:")
                for warning in dir_info['warnings']:
                    lines.append(f"    ✗ {warning}")

    # Special Conditions
    conditions = analysis.get('special_conditions', {})
    has_special = any([
        conditions.get('fu_yin_palaces'),
        conditions.get('fan_yin_palaces'),
        conditions.get('three_wonders_palaces'),
        conditions.get('gate_constraint'),
        conditions.get('star_gate_combinations'),
    ])

    if has_special:
        lines.append("\n┌─────────────────────────────────────────────────────────────┐")
        lines.append("│  特殊格局 Special Conditions                                │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        if conditions.get('fu_yin_palaces'):
            palaces = ', '.join(str(p) for p in conditions['fu_yin_palaces'])
            lines.append(f"\n  伏吟 Fu Yin (Stagnation):")
            lines.append(f"    Palaces: {palaces}")
            lines.append(f"    意義 Meaning: Events may stagnate, delays likely")

        if conditions.get('fan_yin_palaces'):
            palaces = ', '.join(str(p) for p in conditions['fan_yin_palaces'])
            lines.append(f"\n  反吟 Fan Yin (Reversal):")
            lines.append(f"    Palaces: {palaces}")
            lines.append(f"    意義 Meaning: Sudden changes and reversals expected")

        if conditions.get('three_wonders_palaces'):
            lines.append(f"\n  ✨ 三奇 Three Wonders:")
            for wonder in conditions['three_wonders_palaces']:
                lines.append(f"    Palace {wonder['palace']}: {wonder['wonder']}")
            lines.append(f"    意義 Meaning: Auspicious opportunities present")

        if conditions.get('gate_constraint'):
            lines.append(f"\n  ⚠️ 門迫 Gate Constraint:")
            for gc in conditions['gate_constraint']:
                lines.append(f"    Palace {gc['palace']}: {gc['gate']} constrained by {gc['palace_element']}")

        if conditions.get('star_gate_combinations'):
            lines.append(f"\n  星門組合 Notable Star-Gate Combinations:")
            for combo in conditions['star_gate_combinations']:
                lines.append(f"    Palace {combo['palace']}: {combo['name']}")
                lines.append(f"      ({combo['star']} + {combo['gate']})")

    return '\n'.join(lines)


def format_directions_summary(analysis: Dict[str, Any]) -> str:
    """
    Format a concise directional summary.

    Args:
        analysis: Analysis dictionary from QimenAnalyzer.analyze_plate()

    Returns:
        Concise directional summary
    """
    lines = []

    # Best direction
    if analysis.get('auspicious') and len(analysis['auspicious']) > 0:
        best = analysis['auspicious'][0]
        lines.append(f"最佳方位 Best Direction: {best['direction']} (Score: {best['score']})")
        lines.append(f"  {best['star']} + {best['gate']}")

    # Worst direction
    if analysis.get('inauspicious') and len(analysis['inauspicious']) > 0:
        worst = analysis['inauspicious'][0]
        lines.append(f"\n最凶方位 Worst Direction: {worst['direction']} (Score: {worst['score']})")
        lines.append(f"  {worst['star']} + {worst['gate']}")
        if worst.get('warnings'):
            lines.append(f"  警示 Warning: {worst['warnings'][0]}")

    return '\n'.join(lines)


def format_palace_detail(palace_analysis: Dict[str, Any], palace_num: int) -> str:
    """
    Format detailed analysis for a single palace.

    Args:
        palace_analysis: Analysis dict for one palace
        palace_num: Palace number (1-9)

    Returns:
        Formatted palace detail
    """
    lines = []

    lines.append(f"\n【 宮位 {palace_num} - {palace_analysis['direction']} Palace 】")
    lines.append(f"  五行 Element: {palace_analysis['element']}")
    lines.append(f"  天盤干 Heaven Stem: {palace_analysis['stems']['heaven']}")
    lines.append(f"  地盤干 Earth Stem: {palace_analysis['stems']['earth']}")

    comps = palace_analysis.get('components', {})
    if comps.get('star'):
        lines.append(f"  星 Star: {comps['star']}")
    if comps.get('gate'):
        lines.append(f"  門 Gate: {comps['gate']}")
    if comps.get('spirit'):
        lines.append(f"  神 Spirit: {comps['spirit']}")

    lines.append(f"\n  吉凶評分 Favorability Score: {palace_analysis['favorability_score']}")

    if palace_analysis.get('auspicious_factors'):
        lines.append("  吉兆 Auspicious Factors:")
        for factor in palace_analysis['auspicious_factors']:
            lines.append(f"    ✓ {factor}")

    if palace_analysis.get('warnings'):
        lines.append("  ⚠️ 警示 Warnings:")
        for warning in palace_analysis['warnings']:
            lines.append(f"    ✗ {warning}")

    if palace_analysis.get('stem_pattern'):
        pattern = palace_analysis['stem_pattern']
        if pattern.get('pattern_name'):
            lines.append(f"\n  格局 Stem Pattern: {pattern['pattern_name']}")
            if pattern.get('description'):
                lines.append(f"    {pattern['description']}")

    return '\n'.join(lines)
