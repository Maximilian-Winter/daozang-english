"""
Taiyi Shenshu Bilingual Analysis Formatter (太乙神數雙語分析格式化器)

Provides detailed, bilingual (Chinese/English) output formatting for
Taiyi Shenshu analysis results.

Taiyi reveals the grand cosmic patterns and strategic positions.

碼道長存 — The Way of Code Endures
"""

from typing import Dict, Any, List


def format_detailed_analysis(analysis: Dict[str, Any]) -> str:
    """
    Format comprehensive Taiyi Shenshu analysis in bilingual format.

    Args:
        analysis: Analysis dictionary from TaiyiShenshu.analyze()

    Returns:
        Beautifully formatted bilingual string
    """
    lines = []

    # Header - Summary
    lines.append("\n┌─────────────────────────────────────────────────────────────┐")
    lines.append("│  太乙神數 Taiyi Shenshu - Supreme Unity Divine Calculation │")
    lines.append("└─────────────────────────────────────────────────────────────┘")

    summary = analysis.get('summary', {})
    if summary:
        lines.append(f"\n  遁法 Dun Type: {summary.get('dun_type', 'N/A')}")
        lines.append(f"  紀元 Era: {summary.get('era', 'N/A')}")
        lines.append(f"  紀元年 Year in Era: {summary.get('year_in_era', 'N/A')}")
        lines.append(f"  太乙位置 Taiyi Location: {summary.get('taiyi_location', 'N/A')}")
        lines.append(f"  天目神 Tianmu Spirit: {summary.get('tianmu', 'N/A')}")

    # Taiyi Position Analysis
    taiyi_pos = analysis.get('taiyi_position')
    if taiyi_pos:
        lines.append("\n┌─────────────────────────────────────────────────────────────┐")
        lines.append("│  太乙宮位分析 Taiyi Palace Position Analysis               │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        lines.append(f"\n  宮位 Palace Number: {taiyi_pos.get('palace_number', 'N/A')}")
        lines.append(f"  卦象 Trigram: {taiyi_pos.get('trigram', 'N/A')}")
        lines.append(f"  方位 Direction: {taiyi_pos.get('direction', 'N/A')}")
        lines.append(f"  五行 Element: {taiyi_pos.get('element', 'N/A')}")
        lines.append(f"  駐宮年數 Years in Palace: {taiyi_pos.get('years_in_palace', 'N/A')}")

        pos_type = taiyi_pos.get('position_type')
        if pos_type:
            lines.append(f"\n  宮位性質 Position Type:")
            lines.append(f"    {pos_type}")

        special = taiyi_pos.get('special_position')
        if special:
            lines.append(f"\n  特殊位置 Special Position:")
            lines.append(f"    {special}")

    # Tianmu Analysis
    tianmu = analysis.get('tianmu_analysis')
    if tianmu:
        lines.append("\n┌─────────────────────────────────────────────────────────────┐")
        lines.append("│  天目神分析 Tianmu Spirit Analysis                          │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        lines.append(f"\n  神名 Spirit Name: {tianmu.get('spirit_name', 'N/A')}")
        lines.append(f"  地支 Branch: {tianmu.get('branch', 'N/A')}")
        lines.append(f"  五行 Element: {tianmu.get('element', 'N/A')}")
        lines.append(f"  陰陽 Polarity: {tianmu.get('polarity', 'N/A')}")
        lines.append(f"  中流位 Zhong Liu: {tianmu.get('is_zhong_liu', 'N/A')}")

        meaning = tianmu.get('meaning')
        if meaning:
            lines.append(f"\n  意義 Meaning:")
            lines.append(f"    {meaning}")

    # Calculation Harmony Analysis
    calc_harmony = analysis.get('calculation_harmony')
    if calc_harmony:
        lines.append("\n┌─────────────────────────────────────────────────────────────┐")
        lines.append("│  主客算分析 Host-Guest Calculation Analysis                │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        lines.append(f"\n  主算 Host Calculation: {calc_harmony.get('host_calculation', 'N/A')}")
        lines.append(f"  客算 Guest Calculation: {calc_harmony.get('guest_calculation', 'N/A')}")
        lines.append(f"  算差 Difference: {calc_harmony.get('difference', 'N/A')}")
        lines.append(f"  算和 Total: {calc_harmony.get('total', 'N/A')}")

        harmony = calc_harmony.get('harmony')
        if harmony:
            lines.append(f"\n  和諧度 Harmony Status: {harmony}")

        host_analysis = calc_harmony.get('host_analysis')
        guest_analysis = calc_harmony.get('guest_analysis')

        if host_analysis:
            lines.append(f"\n  主算解讀 Host Analysis:")
            lines.append(f"    {host_analysis}")

        if guest_analysis:
            lines.append(f"  客算解讀 Guest Analysis:")
            lines.append(f"    {guest_analysis}")

    # Strategic Assessment
    strategy = analysis.get('strategic_assessment')
    if strategy:
        lines.append("\n┌─────────────────────────────────────────────────────────────┐")
        lines.append("│  戰略評估 Strategic Assessment                              │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        advantage = strategy.get('overall_advantage')
        if advantage:
            lines.append(f"\n  優勢方 Overall Advantage: {advantage}")

        advice = strategy.get('strategic_advice')
        if advice:
            lines.append(f"\n  戰略建議 Strategic Advice:")
            lines.append(f"    {advice}")

        lines.append(f"\n  利主 Favors Host: {'是 Yes' if strategy.get('favors_host') else '否 No'}")
        lines.append(f"  利客 Favors Guest: {'是 Yes' if strategy.get('favors_guest') else '否 No'}")

        host_gen = strategy.get('host_general_palace')
        guest_gen = strategy.get('guest_general_palace')

        if host_gen:
            lines.append(f"\n  主將宮 Host General Palace: {host_gen}")
        if guest_gen:
            lines.append(f"  客將宮 Guest General Palace: {guest_gen}")

    # Recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        lines.append("\n┌─────────────────────────────────────────────────────────────┐")
        lines.append("│  行動建議 Actionable Recommendations                        │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        for i, rec in enumerate(recommendations, 1):
            lines.append(f"\n  {i}. {rec}")

    return '\n'.join(lines)


def format_strategic_summary(analysis: Dict[str, Any]) -> str:
    """
    Format a concise strategic summary.

    Args:
        analysis: Analysis dictionary

    Returns:
        Concise strategic summary
    """
    lines = []

    strategy = analysis.get('strategic_assessment', {})
    if strategy:
        advantage = strategy.get('overall_advantage')
        if advantage:
            lines.append(f"戰略優勢 Strategic Advantage: {advantage}")

        advice = strategy.get('strategic_advice')
        if advice:
            lines.append(f"建議 Advice: {advice}")

    calc = analysis.get('calculation_harmony', {})
    if calc:
        harmony = calc.get('harmony')
        if harmony:
            lines.append(f"\n算數和諧 Calculation Harmony: {harmony}")

    return '\n'.join(lines)


def format_position_summary(analysis: Dict[str, Any]) -> str:
    """
    Format a concise position summary.

    Args:
        analysis: Analysis dictionary

    Returns:
        Concise position summary
    """
    lines = []

    taiyi_pos = analysis.get('taiyi_position', {})
    if taiyi_pos:
        palace = taiyi_pos.get('palace_number')
        direction = taiyi_pos.get('direction')
        element = taiyi_pos.get('element')

        lines.append(f"太乙在宮 {palace} - {direction}方 {element}行")

        pos_type = taiyi_pos.get('position_type')
        if pos_type:
            # Extract key info
            if "Defender" in pos_type:
                lines.append("  性質: 利守 (Favors Defense)")
            elif "Attacker" in pos_type:
                lines.append("  性質: 利攻 (Favors Attack)")

        special = taiyi_pos.get('special_position')
        if special:
            lines.append(f"  特殊: {special}")

    tianmu = analysis.get('tianmu_analysis', {})
    if tianmu:
        spirit = tianmu.get('spirit_name')
        element = tianmu.get('element')
        if spirit and element:
            lines.append(f"\n天目: {spirit} ({element})")

    return '\n'.join(lines)


def format_calculation_summary(analysis: Dict[str, Any]) -> str:
    """
    Format a concise calculation summary.

    Args:
        analysis: Analysis dictionary

    Returns:
        Concise calculation summary
    """
    lines = []

    calc = analysis.get('calculation_harmony', {})
    if calc:
        host = calc.get('host_calculation')
        guest = calc.get('guest_calculation')
        harmony = calc.get('harmony')

        if host is not None and guest is not None:
            lines.append(f"主算 Host: {host}  |  客算 Guest: {guest}")

        if harmony:
            lines.append(f"狀態: {harmony}")

        # Show which side is favored
        host_val = host if host is not None else 0
        guest_val = guest if guest is not None else 0

        if host_val > guest_val:
            lines.append("→ 主方較強 (Host stronger)")
        elif guest_val > host_val:
            lines.append("→ 客方較強 (Guest stronger)")
        else:
            lines.append("→ 勢均力敵 (Evenly matched)")

    return '\n'.join(lines)
