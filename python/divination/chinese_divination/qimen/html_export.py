"""
Qimen Dunjia HTML Export Module (奇門遁甲 HTML 輸出模組)

Generates beautiful, self-contained static HTML pages for Qimen Dunjia plates.
Features Chinese aesthetic design with traditional colors and styling.

九天玄碼女在此 - 碼道長存
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import html


from .palaces import QimenPlate, Palace
from .analysis import QimenAnalyzer


class QimenHTMLExporter:
    """
    Generates stylish HTML exports for Qimen Dunjia plates.

    Creates self-contained HTML files with embedded CSS and all plate
    information beautifully displayed in a traditional Chinese aesthetic.
    """

    def __init__(self):
        self.analyzer = QimenAnalyzer()

    def export_plate(self, plate: QimenPlate,
                     output_path: Optional[str] = None,
                     title: Optional[str] = None,
                     include_analysis: bool = True) -> str:
        """
        Export a Qimen plate to an HTML file.

        Args:
            plate: The QimenPlate to export
            output_path: Path for the HTML file (optional, returns string if None)
            title: Custom title for the page
            include_analysis: Whether to include detailed analysis

        Returns:
            HTML string if output_path is None, otherwise the output path
        """
        analysis = self.analyzer.analyze_plate(plate) if include_analysis else None
        html_content = self._generate_html(plate, analysis, title)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return output_path
        return html_content

    def _generate_html(self, plate: QimenPlate,
                       analysis: Optional[Dict[str, Any]],
                       title: Optional[str]) -> str:
        """Generate the complete HTML document."""

        page_title = title or f"奇門遁甲盤 - {plate.solar_term.chinese}"

        return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(page_title)}</title>
    {self._get_css()}
</head>
<body>
    <div class="container">
        {self._generate_header(plate)}
        {self._generate_info_section(plate)}
        {self._generate_palace_grid(plate)}
        {self._generate_analysis_section(plate, analysis) if analysis else ''}
        {self._generate_footer()}
    </div>
</body>
</html>"""

    def _get_css(self) -> str:
        """Return embedded CSS with Chinese aesthetic styling."""
        return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@400;600;700&family=Noto+Sans+TC:wght@300;400;500&display=swap');

        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --accent-gold: #d4af37;
            --accent-red: #c41e3a;
            --accent-jade: #00a86b;
            --text-primary: #eee8d5;
            --text-secondary: #b8b8b8;
            --text-muted: #888;
            --border-gold: #b8860b;
            --shadow-color: rgba(0, 0, 0, 0.5);

            /* Element colors */
            --wood: #228b22;
            --fire: #dc143c;
            --earth: #daa520;
            --metal: #c0c0c0;
            --water: #4169e1;

            /* Gate colors */
            --gate-auspicious: #00a86b;
            --gate-neutral: #daa520;
            --gate-inauspicious: #c41e3a;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Noto Sans TC', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Header */
        .header {
            text-align: center;
            padding: 40px 20px;
            border-bottom: 2px solid var(--border-gold);
            margin-bottom: 30px;
            position: relative;
        }

        .header::before,
        .header::after {
            content: '☰';
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            font-size: 2rem;
            color: var(--accent-gold);
            opacity: 0.5;
        }

        .header::before { left: 20px; }
        .header::after { right: 20px; }

        .header h1 {
            font-family: 'Noto Serif TC', serif;
            font-size: 2.5rem;
            color: var(--accent-gold);
            text-shadow: 2px 2px 4px var(--shadow-color);
            margin-bottom: 10px;
            letter-spacing: 8px;
        }

        .header .subtitle {
            font-size: 1.1rem;
            color: var(--text-secondary);
            letter-spacing: 2px;
        }

        .header .datetime {
            margin-top: 15px;
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        /* Info Section */
        .info-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .info-card {
            background: var(--bg-card);
            border: 1px solid var(--border-gold);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px var(--shadow-color);
        }

        .info-card .label {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .info-card .value {
            font-family: 'Noto Serif TC', serif;
            font-size: 1.5rem;
            color: var(--accent-gold);
        }

        .info-card .sub-value {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 5px;
        }

        /* Four Pillars */
        .pillars {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 40px;
            flex-wrap: wrap;
        }

        .pillar {
            background: var(--bg-card);
            border: 1px solid var(--border-gold);
            border-radius: 8px;
            padding: 15px 25px;
            text-align: center;
            min-width: 100px;
        }

        .pillar .pillar-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .pillar .stem-branch {
            font-family: 'Noto Serif TC', serif;
            font-size: 1.8rem;
            color: var(--text-primary);
            letter-spacing: 2px;
        }

        /* Palace Grid */
        .palace-grid-container {
            margin: 40px 0;
        }

        .palace-grid-title {
            text-align: center;
            font-family: 'Noto Serif TC', serif;
            font-size: 1.5rem;
            color: var(--accent-gold);
            margin-bottom: 25px;
            letter-spacing: 4px;
        }

        .palace-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            max-width: 900px;
            margin: 0 auto;
        }

        .palace {
            background: var(--bg-card);
            border: 2px solid var(--border-gold);
            border-radius: 12px;
            padding: 20px;
            min-height: 200px;
            position: relative;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .palace:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(212, 175, 55, 0.3);
        }

        .palace.center {
            background: linear-gradient(135deg, var(--bg-card) 0%, #1a3a5c 100%);
        }

        .palace-number {
            position: absolute;
            top: 10px;
            left: 10px;
            width: 28px;
            height: 28px;
            background: var(--accent-gold);
            color: var(--bg-primary);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9rem;
        }

        .palace-trigram {
            position: absolute;
            top: 10px;
            right: 10px;
            font-family: 'Noto Serif TC', serif;
            font-size: 1.5rem;
            color: var(--accent-gold);
        }

        .palace-direction {
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 35px;
            margin-bottom: 15px;
        }

        .palace-stems {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 15px;
        }

        .stem-box {
            text-align: center;
        }

        .stem-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-bottom: 3px;
        }

        .stem-value {
            font-family: 'Noto Serif TC', serif;
            font-size: 1.6rem;
            color: var(--text-primary);
        }

        .stem-value.wonder {
            color: var(--accent-jade);
            text-shadow: 0 0 10px rgba(0, 168, 107, 0.5);
        }

        .palace-components {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            margin-top: 10px;
        }

        .component {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.95rem;
        }

        .component-icon {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .component-icon.star { background: var(--accent-gold); }
        .component-icon.gate-good { background: var(--gate-auspicious); }
        .component-icon.gate-neutral { background: var(--gate-neutral); }
        .component-icon.gate-bad { background: var(--gate-inauspicious); }
        .component-icon.spirit { background: var(--water); }

        /* Element badge */
        .element-badge {
            position: absolute;
            bottom: 10px;
            right: 10px;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .element-wood { background: var(--wood); color: white; }
        .element-fire { background: var(--fire); color: white; }
        .element-earth { background: var(--earth); color: var(--bg-primary); }
        .element-metal { background: var(--metal); color: var(--bg-primary); }
        .element-water { background: var(--water); color: white; }

        /* Analysis Section */
        .analysis-section {
            margin-top: 50px;
        }

        .analysis-title {
            font-family: 'Noto Serif TC', serif;
            font-size: 1.5rem;
            color: var(--accent-gold);
            text-align: center;
            margin-bottom: 30px;
            letter-spacing: 4px;
        }

        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
        }

        .analysis-card {
            background: var(--bg-card);
            border: 1px solid var(--border-gold);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 15px var(--shadow-color);
        }

        .analysis-card h3 {
            font-family: 'Noto Serif TC', serif;
            color: var(--accent-gold);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-gold);
            font-size: 1.1rem;
        }

        .analysis-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(184, 134, 11, 0.2);
        }

        .analysis-item:last-child {
            border-bottom: none;
        }

        .analysis-label {
            color: var(--text-secondary);
        }

        .analysis-value {
            color: var(--text-primary);
            font-weight: 500;
        }

        .analysis-value.good { color: var(--gate-auspicious); }
        .analysis-value.warning { color: var(--accent-red); }

        /* Direction recommendations */
        .direction-list {
            list-style: none;
        }

        .direction-list li {
            padding: 12px 15px;
            margin: 8px 0;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.2);
        }

        .direction-list li.auspicious {
            border-left: 4px solid var(--gate-auspicious);
        }

        .direction-list li.inauspicious {
            border-left: 4px solid var(--gate-inauspicious);
        }

        .direction-name {
            font-weight: 600;
            color: var(--accent-gold);
        }

        .direction-details {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 5px;
        }

        /* Overall Rating */
        .overall-rating {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, var(--bg-card) 0%, #1a3a5c 100%);
            border: 2px solid var(--accent-gold);
            border-radius: 15px;
            margin: 30px 0;
        }

        .rating-label {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 10px;
            letter-spacing: 2px;
        }

        .rating-value {
            font-family: 'Noto Serif TC', serif;
            font-size: 2.5rem;
            color: var(--accent-gold);
            text-shadow: 0 0 20px rgba(212, 175, 55, 0.5);
        }

        .rating-advice {
            margin-top: 15px;
            font-size: 1rem;
            color: var(--text-secondary);
        }

        /* Special Conditions */
        .special-conditions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }

        .condition-tag {
            padding: 6px 15px;
            border-radius: 20px;
            font-size: 0.85rem;
            background: rgba(196, 30, 58, 0.3);
            border: 1px solid var(--accent-red);
            color: var(--text-primary);
        }

        .condition-tag.positive {
            background: rgba(0, 168, 107, 0.3);
            border-color: var(--accent-jade);
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 40px 20px;
            margin-top: 50px;
            border-top: 1px solid var(--border-gold);
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        .footer .brand {
            font-family: 'Noto Serif TC', serif;
            color: var(--accent-gold);
            font-size: 1rem;
            margin-bottom: 10px;
            letter-spacing: 4px;
        }

        .footer .verse {
            font-style: italic;
            margin-top: 15px;
            color: var(--text-secondary);
        }

        /* Responsive */
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8rem;
                letter-spacing: 4px;
            }

            .palace-grid {
                grid-template-columns: 1fr;
                max-width: 400px;
            }

            .pillars {
                flex-direction: column;
                align-items: center;
            }

            .header::before,
            .header::after {
                display: none;
            }
        }

        /* Print styles */
        @media print {
            body {
                background: white;
                color: black;
            }

            .palace, .info-card, .analysis-card {
                border: 1px solid #333;
                box-shadow: none;
            }
        }
    </style>"""

    def _generate_header(self, plate: QimenPlate) -> str:
        """Generate the page header."""
        dt = plate.datetime_info
        gregorian_str = f"{dt.year}年{dt.month}月{dt.day}日"

        return f"""
        <header class="header">
            <h1>奇門遁甲</h1>
            <div class="subtitle">QIMEN DUNJIA DIVINATION PLATE</div>
            <div class="datetime">{gregorian_str}</div>
        </header>"""

    def _generate_info_section(self, plate: QimenPlate) -> str:
        """Generate the configuration info section."""
        dt = plate.datetime_info

        # Four Pillars
        four_pillars = f"""
        <div class="pillars">
            <div class="pillar">
                <div class="pillar-label">Year 年柱</div>
                <div class="stem-branch">{dt.year_stem.chinese}{dt.year_branch.chinese}</div>
            </div>
            <div class="pillar">
                <div class="pillar-label">Month 月柱</div>
                <div class="stem-branch">{dt.month_stem.chinese}{dt.month_branch.chinese}</div>
            </div>
            <div class="pillar">
                <div class="pillar-label">Day 日柱</div>
                <div class="stem-branch">{dt.day_stem.chinese}{dt.day_branch.chinese}</div>
            </div>
            <div class="pillar">
                <div class="pillar-label">Hour 時柱</div>
                <div class="stem-branch">{dt.hour_stem.chinese}{dt.hour_branch.chinese}</div>
            </div>
        </div>"""

        # Info cards
        duty_star = plate.duty_star.chinese if plate.duty_star else '-'
        duty_gate = plate.duty_gate.chinese if plate.duty_gate else '-'

        info_cards = f"""
        <div class="info-section">
            <div class="info-card">
                <div class="label">遁法 Dun Type</div>
                <div class="value">{plate.dun_type.chinese}</div>
            </div>
            <div class="info-card">
                <div class="label">元 Yuan</div>
                <div class="value">{plate.yuan.chinese}</div>
            </div>
            <div class="info-card">
                <div class="label">局 Ju</div>
                <div class="value">{plate.ju_number}局</div>
            </div>
            <div class="info-card">
                <div class="label">節氣 Solar Term</div>
                <div class="value">{plate.solar_term.chinese}</div>
            </div>
            <div class="info-card">
                <div class="label">值符 Duty Star</div>
                <div class="value">{duty_star}</div>
                <div class="sub-value">落{plate.duty_palace}宮</div>
            </div>
            <div class="info-card">
                <div class="label">值使 Duty Gate</div>
                <div class="value">{duty_gate}</div>
            </div>
        </div>"""

        return four_pillars + info_cards

    def _generate_palace_grid(self, plate: QimenPlate) -> str:
        """Generate the Nine Palaces grid."""
        # Grid order: 4,9,2 / 3,5,7 / 8,1,6
        grid_order = [[4, 9, 2], [3, 5, 7], [8, 1, 6]]

        palaces_html = ""
        for row in grid_order:
            for palace_num in row:
                palace = plate.get_palace(palace_num)
                if palace:
                    palaces_html += self._generate_palace_cell(palace)

        return f"""
        <div class="palace-grid-container">
            <div class="palace-grid-title">九 宮 盤</div>
            <div class="palace-grid">
                {palaces_html}
            </div>
        </div>"""

    def _generate_palace_cell(self, palace: Palace) -> str:
        """Generate a single palace cell."""
        trigram = palace.trigram.chinese if palace.trigram else "中"
        direction = palace.direction.value if palace.direction else ""

        # Determine element class
        element_class = ""
        element_name = ""
        if palace.base_element:
            element_map = {
                '木': 'wood', '火': 'fire', '土': 'earth',
                '金': 'metal', '水': 'water'
            }
            element_name = palace.base_element.value
            element_class = element_map.get(element_name, '')

        # Check for Three Wonders
        three_wonders = ['乙', '丙', '丁']
        earth_class = 'wonder' if palace.earth_plate_stem in three_wonders else ''
        heaven_class = 'wonder' if palace.heaven_plate_stem in three_wonders else ''

        # Components
        star_html = ""
        gate_html = ""
        spirit_html = ""

        if palace.star:
            star_html = f'<div class="component"><span class="component-icon star"></span>{palace.star.chinese}</div>'

        if palace.gate:
            gate_class = self._get_gate_class(palace.gate.chinese)
            gate_html = f'<div class="component"><span class="component-icon {gate_class}"></span>{palace.gate.chinese}</div>'

        if palace.spirit:
            spirit_html = f'<div class="component"><span class="component-icon spirit"></span>{palace.spirit.chinese}</div>'

        center_class = "center" if palace.number == 5 else ""

        return f"""
            <div class="palace {center_class}">
                <div class="palace-number">{palace.number}</div>
                <div class="palace-trigram">{trigram}</div>
                <div class="palace-direction">{direction}</div>
                <div class="palace-stems">
                    <div class="stem-box">
                        <div class="stem-label">地</div>
                        <div class="stem-value {earth_class}">{palace.earth_plate_stem or '-'}</div>
                    </div>
                    <div class="stem-box">
                        <div class="stem-label">天</div>
                        <div class="stem-value {heaven_class}">{palace.heaven_plate_stem or '-'}</div>
                    </div>
                </div>
                <div class="palace-components">
                    {star_html}
                    {gate_html}
                    {spirit_html}
                </div>
                <div class="element-badge element-{element_class}">{element_name}</div>
            </div>"""

    def _get_gate_class(self, gate_chinese: str) -> str:
        """Get CSS class for gate favorability."""
        good_gates = ['開門', '休門', '生門']
        bad_gates = ['死門', '驚門']

        if gate_chinese in good_gates:
            return 'gate-good'
        elif gate_chinese in bad_gates:
            return 'gate-bad'
        return 'gate-neutral'

    def _generate_analysis_section(self, plate: QimenPlate, analysis: Dict[str, Any]) -> str:
        """Generate the analysis section."""
        overall = analysis.get('overall_assessment', {})
        rating = overall.get('rating', '平 - Neutral')
        advice = overall.get('advice', '')

        # Auspicious directions
        auspicious = analysis.get('auspicious', [])
        auspicious_html = ""
        for d in auspicious[:3]:  # Top 3
            auspicious_html += f"""
                <li class="auspicious">
                    <div class="direction-name">{d['direction']} ({d['palace_number']}宮)</div>
                    <div class="direction-details">{d['star']} + {d['gate']}</div>
                </li>"""

        # Inauspicious directions
        inauspicious = analysis.get('inauspicious', [])
        inauspicious_html = ""
        for d in inauspicious[:3]:  # Top 3 to avoid
            warnings = ', '.join(d.get('warnings', ['注意']))
            inauspicious_html += f"""
                <li class="inauspicious">
                    <div class="direction-name">{d['direction']} ({d['palace_number']}宮)</div>
                    <div class="direction-details">{warnings}</div>
                </li>"""

        # Special conditions
        conditions = analysis.get('special_conditions', {})
        conditions_html = ""

        three_wonders = conditions.get('three_wonders_palaces', [])
        for tw in three_wonders:
            conditions_html += f'<span class="condition-tag positive">三奇 {tw["wonder"]} @ {tw["palace"]}宮</span>'

        fu_yin = conditions.get('fu_yin_palaces', [])
        for p in fu_yin:
            conditions_html += f'<span class="condition-tag">伏吟 @ {p}宮</span>'

        fan_yin = conditions.get('fan_yin_palaces', [])
        for p in fan_yin:
            conditions_html += f'<span class="condition-tag">反吟 @ {p}宮</span>'

        return f"""
        <section class="analysis-section">
            <div class="analysis-title">盤 局 分 析</div>

            <div class="overall-rating">
                <div class="rating-label">OVERALL ASSESSMENT</div>
                <div class="rating-value">{rating}</div>
                <div class="rating-advice">{advice}</div>
                {f'<div class="special-conditions">{conditions_html}</div>' if conditions_html else ''}
            </div>

            <div class="analysis-grid">
                <div class="analysis-card">
                    <h3>吉方 Auspicious Directions</h3>
                    <ul class="direction-list">
                        {auspicious_html if auspicious_html else '<li>No highly auspicious directions found</li>'}
                    </ul>
                </div>

                <div class="analysis-card">
                    <h3>凶方 Directions to Avoid</h3>
                    <ul class="direction-list">
                        {inauspicious_html if inauspicious_html else '<li>No critical warnings</li>'}
                    </ul>
                </div>

                <div class="analysis-card">
                    <h3>值符值使 Duty Elements</h3>
                    <div class="analysis-item">
                        <span class="analysis-label">值符 (Duty Star)</span>
                        <span class="analysis-value">{analysis['duty_elements']['star'] or '-'}</span>
                    </div>
                    <div class="analysis-item">
                        <span class="analysis-label">值使 (Duty Gate)</span>
                        <span class="analysis-value">{analysis['duty_elements']['gate'] or '-'}</span>
                    </div>
                    <div class="analysis-item">
                        <span class="analysis-label">落宮 (Duty Palace)</span>
                        <span class="analysis-value">{analysis['duty_elements']['palace']}宮</span>
                    </div>
                </div>

                <div class="analysis-card">
                    <h3>統計 Statistics</h3>
                    <div class="analysis-item">
                        <span class="analysis-label">Auspicious Directions</span>
                        <span class="analysis-value good">{overall.get('favorable_directions', 0)}</span>
                    </div>
                    <div class="analysis-item">
                        <span class="analysis-label">Inauspicious Directions</span>
                        <span class="analysis-value warning">{overall.get('unfavorable_directions', 0)}</span>
                    </div>
                    <div class="analysis-item">
                        <span class="analysis-label">Three Wonders Present</span>
                        <span class="analysis-value">{overall.get('three_wonders_present', 0)}</span>
                    </div>
                </div>
            </div>
        </section>"""

    def _generate_footer(self) -> str:
        """Generate the page footer."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        return f"""
        <footer class="footer">
            <div class="brand">九天玄碼女</div>
            <div>Generated: {timestamp}</div>
            <div class="verse">碼道無形 生育萬程 — The Way of Code is formless, giving birth to all programs</div>
        </footer>"""


# =============================================================================
# Convenience Functions
# =============================================================================

def export_plate_to_html(plate: QimenPlate,
                         output_path: Optional[str] = None,
                         title: Optional[str] = None) -> str:
    """
    Convenience function to export a plate to HTML.

    Args:
        plate: QimenPlate to export
        output_path: File path for HTML output
        title: Optional page title

    Returns:
        HTML string or file path
    """
    exporter = QimenHTMLExporter()
    return exporter.export_plate(plate, output_path, title)


def quick_html_export(dt: Optional[datetime] = None,
                      output_path: Optional[str] = None) -> str:
    """
    Quick export: calculate plate and generate HTML in one step.

    Args:
        dt: Datetime for calculation (default: now)
        output_path: File path for output

    Returns:
        HTML string or file path
    """
    from .qimen_dunjia import QimenDunjia

    qimen = QimenDunjia()
    plate = qimen.calculate(dt or datetime.now())

    exporter = QimenHTMLExporter()
    return exporter.export_plate(plate, output_path)
