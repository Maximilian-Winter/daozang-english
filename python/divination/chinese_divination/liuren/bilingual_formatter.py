"""
Da Liu Ren Bilingual Analysis Formatter (大六壬雙語分析格式化器)

Provides detailed, bilingual (Chinese/English) output formatting for
Da Liu Ren analysis results.

The Three Transmissions reveal the flow of fate from cause to effect.

碼道長存 — The Way of Code Endures
"""

from typing import Dict, Any, List, Optional


def format_detailed_analysis(analysis: Dict[str, Any]) -> str:
    """
    Format comprehensive Da Liu Ren analysis in bilingual format.

    Args:
        analysis: Analysis dictionary from LiuRenAnalyzer.analyze()

    Returns:
        Beautifully formatted bilingual string
    """
    lines = []

    # Header
    lines.append("\n┌─────────────────────────────────────────────────────────────┐")
    lines.append("│  課式概要 Lesson Summary                                    │")
    lines.append("└─────────────────────────────────────────────────────────────┘")

    summary = analysis.get('plate_summary', {})
    if summary:
        lines.append(f"  日干 Day Stem: {summary.get('day_stem', 'N/A')}")
        lines.append(f"  日支 Day Branch: {summary.get('day_branch', 'N/A')}")
        lines.append(f"  時干 Hour Stem: {summary.get('hour_stem', 'N/A')}")
        lines.append(f"  時支 Hour Branch: {summary.get('hour_branch', 'N/A')}")

        if summary.get('lesson_pattern'):
            lines.append(f"\n  課式 Lesson Pattern: {summary['lesson_pattern']}")

        if summary.get('special_patterns'):
            patterns = ', '.join(summary['special_patterns'])
            lines.append(f"  特殊格局 Special Patterns: {patterns}")

    # Overall Favorability
    lines.append("\n┌─────────────────────────────────────────────────────────────┐")
    lines.append("│  整體評斷 Overall Favorability                              │")
    lines.append("└─────────────────────────────────────────────────────────────┘")

    overall_fav = analysis.get('overall_favorability', 'Unknown')
    lines.append(f"\n  【 {overall_fav} 】")

    # Patterns
    patterns_summary = analysis.get('patterns', {})
    if patterns_summary:
        lines.append("\n  格局分析 Pattern Analysis:")

        if patterns_summary.get('lesson_pattern'):
            lp = patterns_summary['lesson_pattern']
            lines.append(f"    課式: {lp.get('name', 'Unknown')}")
            if lp.get('description'):
                lines.append(f"      {lp['description']}")

        if patterns_summary.get('transmission_patterns'):
            lines.append(f"    傳送格局 Transmission Patterns:")
            for tp in patterns_summary['transmission_patterns']:
                lines.append(f"      • {tp}")

        if patterns_summary.get('special_patterns'):
            lines.append(f"    特殊格局 Special Patterns:")
            for sp in patterns_summary['special_patterns']:
                lines.append(f"      • {sp}")

    # Three Transmissions (San Chuan) Analysis
    san_chuan = analysis.get('san_chuan_analysis')
    if san_chuan:
        lines.append("\n┌─────────────────────────────────────────────────────────────┐")
        lines.append("│  三傳分析 Three Transmissions Analysis                      │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        # Overview
        if san_chuan.get('overview'):
            lines.append(f"\n  總評 Overview: {san_chuan['overview']}")

        # Derivation method
        if san_chuan.get('derivation'):
            lines.append(f"  起法 Derivation: {san_chuan['derivation']}")

        # Chu Chuan (Initial Transmission)
        if san_chuan.get('chu_chuan'):
            chu = san_chuan['chu_chuan']
            lines.append(f"\n  ▶ 初傳 CHU CHUAN (Beginning/Cause):")
            lines.append(f"      支 Branch: {chu['branch']}")
            lines.append(f"      五行 Element: {chu['element']}")
            lines.append(f"      方位 Direction: {chu['direction']}")
            lines.append(f"      {chu['interpretation']}")

            # Add general if available
            generals = san_chuan.get('generals', {})
            if generals.get('chu'):
                lines.append(f"      天將 General: {generals['chu'].chinese}")

        # Zhong Chuan (Middle Transmission)
        if san_chuan.get('zhong_chuan'):
            zhong = san_chuan['zhong_chuan']
            lines.append(f"\n  ▶ 中傳 ZHONG CHUAN (Development/Process):")
            lines.append(f"      支 Branch: {zhong['branch']}")
            lines.append(f"      五行 Element: {zhong['element']}")
            lines.append(f"      方位 Direction: {zhong['direction']}")
            lines.append(f"      {zhong['interpretation']}")

            generals = san_chuan.get('generals', {})
            if generals.get('zhong'):
                lines.append(f"      天將 General: {generals['zhong'].chinese}")

        # Mo Chuan (Final Transmission)
        if san_chuan.get('mo_chuan'):
            mo = san_chuan['mo_chuan']
            lines.append(f"\n  ▶ 末傳 MO CHUAN (Outcome/Result):")
            lines.append(f"      支 Branch: {mo['branch']}")
            lines.append(f"      五行 Element: {mo['element']}")
            lines.append(f"      方位 Direction: {mo['direction']}")
            lines.append(f"      {mo['interpretation']}")

            generals = san_chuan.get('generals', {})
            if generals.get('mo'):
                lines.append(f"      天將 General: {generals['mo'].chinese}")

        # Flow analysis
        if san_chuan.get('flow'):
            flow = san_chuan['flow']
            lines.append(f"\n  流轉分析 Transmission Flow:")
            lines.append(f"    初→中 Chu→Zhong: {flow.get('chu_to_zhong', 'N/A')}")
            lines.append(f"    中→末 Zhong→Mo: {flow.get('zhong_to_mo', 'N/A')}")
            lines.append(f"    總體 Overall: {flow.get('overall', 'N/A')}")

    # General Analysis
    general_analysis = analysis.get('general_analysis')
    if general_analysis:
        lines.append("\n┌─────────────────────────────────────────────────────────────┐")
        lines.append("│  天將分析 Heavenly Generals Analysis                        │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        # Noble Person
        if general_analysis.get('noble_person'):
            noble = general_analysis['noble_person']
            lines.append(f"\n  貴人 Noble Person:")
            lines.append(f"    {noble.get('interpretation', 'N/A')}")
            lines.append(f"    方位 Direction: {noble.get('direction', 'N/A')}")
            lines.append(f"    五行 Element: {noble.get('element', 'N/A')}")

        # Transmission Generals
        if general_analysis.get('transmission_generals'):
            lines.append(f"\n  三傳天將 Transmission Generals:")
            trans_gens = general_analysis['transmission_generals']

            for name in ['初傳', '中傳', '末傳']:
                if name in trans_gens:
                    tg = trans_gens[name]
                    status = "吉" if tg.get('is_auspicious') else "凶" if tg.get('is_auspicious') is False else "中"
                    lines.append(f"    {name}: {tg.get('general', 'N/A')} [{status}]")

                    keywords = tg.get('keywords', [])
                    if keywords:
                        lines.append(f"      關鍵 Keywords: {', '.join(keywords[:3])}")

        # Favorable positions
        if general_analysis.get('favorable_positions'):
            lines.append(f"\n  吉神方位 Auspicious Generals:")
            for pos in general_analysis['favorable_positions'][:4]:
                keywords = ', '.join(pos.get('keywords', [])[:2])
                lines.append(f"    {pos['branch']}: {pos['general']} - {keywords}")

        # Unfavorable positions
        if general_analysis.get('unfavorable_positions'):
            lines.append(f"\n  ⚠️ 凶神方位 Inauspicious Generals:")
            for pos in general_analysis['unfavorable_positions'][:4]:
                keywords = ', '.join(pos.get('keywords', [])[:2])
                lines.append(f"    {pos['branch']}: {pos['general']} - {keywords}")

    # Directional Advice
    directional = analysis.get('directional_advice')
    if directional:
        lines.append("\n┌─────────────────────────────────────────────────────────────┐")
        lines.append("│  方位建議 Directional Recommendations                       │")
        lines.append("└─────────────────────────────────────────────────────────────┘")

        # Group by favorability
        auspicious_dirs = []
        inauspicious_dirs = []
        neutral_dirs = []

        for direction, advice in directional.items():
            fav = advice.get('favorability', '中')
            entry = {
                'direction': direction,
                'favorability': fav,
                'reason': advice.get('reason', ''),
                'general': advice.get('general', '')
            }

            if '吉' in fav:
                auspicious_dirs.append(entry)
            elif '凶' in fav:
                inauspicious_dirs.append(entry)
            else:
                neutral_dirs.append(entry)

        if auspicious_dirs:
            lines.append(f"\n  吉方 Favorable Directions:")
            for dir_info in auspicious_dirs:
                lines.append(f"    {dir_info['direction']}: {dir_info['favorability']}")
                if dir_info['general']:
                    lines.append(f"      天將: {dir_info['general']}")
                if dir_info['reason']:
                    lines.append(f"      {dir_info['reason']}")

        if inauspicious_dirs:
            lines.append(f"\n  凶方 Unfavorable Directions:")
            for dir_info in inauspicious_dirs:
                lines.append(f"    {dir_info['direction']}: {dir_info['favorability']}")
                if dir_info['general']:
                    lines.append(f"      天將: {dir_info['general']}")
                if dir_info['reason']:
                    lines.append(f"      {dir_info['reason']}")

    return '\n'.join(lines)


def format_san_chuan_summary(san_chuan_analysis: Dict[str, Any]) -> str:
    """
    Format a concise Three Transmissions summary.

    Args:
        san_chuan_analysis: San Chuan analysis dict

    Returns:
        Concise summary
    """
    lines = []

    if san_chuan_analysis.get('overview'):
        lines.append(f"三傳概要: {san_chuan_analysis['overview']}")

    chu = san_chuan_analysis.get('chu_chuan', {})
    zhong = san_chuan_analysis.get('zhong_chuan', {})
    mo = san_chuan_analysis.get('mo_chuan', {})

    if chu and zhong and mo:
        lines.append(f"初傳 Chu: {chu.get('branch')} ({chu.get('element')})")
        lines.append(f"中傳 Zhong: {zhong.get('branch')} ({zhong.get('element')})")
        lines.append(f"末傳 Mo: {mo.get('branch')} ({mo.get('element')})")

    flow = san_chuan_analysis.get('flow', {})
    if flow.get('overall'):
        lines.append(f"\n流轉: {flow['overall']}")

    return '\n'.join(lines)


def format_generals_summary(general_analysis: Dict[str, Any]) -> str:
    """
    Format a concise generals summary.

    Args:
        general_analysis: General analysis dict

    Returns:
        Concise summary
    """
    lines = []

    # Noble person
    noble = general_analysis.get('noble_person', {})
    if noble.get('interpretation'):
        lines.append(noble['interpretation'])

    # Key favorable
    favorable = general_analysis.get('favorable_positions', [])
    if favorable:
        lines.append(f"\n主要吉神 Key Auspicious:")
        for pos in favorable[:2]:
            lines.append(f"  {pos['branch']}: {pos['general']}")

    # Key unfavorable
    unfavorable = general_analysis.get('unfavorable_positions', [])
    if unfavorable:
        lines.append(f"\n主要凶神 Key Inauspicious:")
        for pos in unfavorable[:2]:
            lines.append(f"  {pos['branch']}: {pos['general']}")

    return '\n'.join(lines)
