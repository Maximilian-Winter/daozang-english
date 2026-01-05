"""
Qimen Dunjia Analysis Module (奇門遁甲分析模組)

This module provides interpretation and analysis methods for
Qimen Dunjia plate configurations.

Analysis includes:
- Element relationship analysis (Five Elements interactions)
- Star-Gate combination assessment
- Direction favorability
- Special condition detection
- Question-specific divination

Classical references:
    - 黄帝太一八门逆顺生死诀
    - 黄帝太一八门入式诀

九天玄碼女在此 - 碼道長存
"""

import sys
import os
from typing import Dict, List, Optional, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.core import Element, Direction
from .palaces import QimenPlate, Palace
from .components import NineStar, EightGate, EightSpirit
from .constants import THREE_WONDERS, SIX_YI
from .stem_patterns import analyze_stem_combination, get_stem_element_interaction


class QimenAnalyzer:
    """
    Analysis engine for Qimen Dunjia plate interpretation.

    Provides methods for analyzing plate configurations, determining
    favorability, and generating divination interpretations.
    """

    def __init__(self):
        """Initialize the analyzer with element relationship mappings."""
        # Five Element generating cycle (相生)
        self.generating_cycle = {
            Element.WOOD: Element.FIRE,    # Wood generates Fire
            Element.FIRE: Element.EARTH,   # Fire generates Earth
            Element.EARTH: Element.METAL,  # Earth generates Metal
            Element.METAL: Element.WATER,  # Metal generates Water
            Element.WATER: Element.WOOD,   # Water generates Wood
        }

        # Five Element controlling cycle (相剋)
        self.controlling_cycle = {
            Element.WOOD: Element.EARTH,   # Wood controls Earth
            Element.EARTH: Element.WATER,  # Earth controls Water
            Element.WATER: Element.FIRE,   # Water controls Fire
            Element.FIRE: Element.METAL,   # Fire controls Metal
            Element.METAL: Element.WOOD,   # Metal controls Wood
        }

        # Gate favorability rankings (traditional assessment)
        self.gate_favorability = {
            '開門': 5,   # Kai - Most auspicious, openings
            '休門': 4,   # Xiu - Very good, rest and recuperation
            '生門': 4,   # Sheng - Good, growth and vitality
            '傷門': 2,   # Shang - Caution, injuries possible
            '杜門': 2,   # Du - Blocked, obstacles
            '景門': 3,   # Jing - Moderate, illumination
            '死門': 1,   # Si - Avoid, endings
            '驚門': 1,   # Jing - Avoid, shocks and fears
        }

        # Star favorability rankings
        self.star_favorability = {
            '天心': 5,   # Tianxin - Most auspicious
            '天任': 4,   # Tianren - Very good
            '天輔': 4,   # Tianfu - Good support
            '天沖': 3,   # Tianchong - Active energy
            '天禽': 3,   # Tianqin - Central, neutral
            '天英': 3,   # Tianying - Bright but intense
            '天芮': 2,   # Tianrui - Caution, illness star
            '天柱': 2,   # Tianzhu - Caution, destruction
            '天蓬': 1,   # Tianpeng - Most difficult
        }

    # =========================================================================
    # Element Analysis
    # =========================================================================

    def analyze_element_relationship(self, element1: Element, element2: Element) -> str:
        """
        Analyze the relationship between two elements.

        Args:
            element1: First element
            element2: Second element

        Returns:
            Relationship type: 'generating', 'controlling', 'controlled', 'draining', 'same'
        """
        if element1 == element2:
            return 'same'

        if self.generating_cycle.get(element1) == element2:
            return 'generating'  # element1 generates element2

        if self.generating_cycle.get(element2) == element1:
            return 'draining'  # element2 drains element1

        if self.controlling_cycle.get(element1) == element2:
            return 'controlling'  # element1 controls element2

        if self.controlling_cycle.get(element2) == element1:
            return 'controlled'  # element1 is controlled by element2

        return 'neutral'

    def get_element_relationship_meaning(self, relationship: str) -> Dict[str, Any]:
        """
        Get the meaning and favorability of an element relationship.

        Args:
            relationship: Relationship type from analyze_element_relationship

        Returns:
            Dictionary with meaning and favorability score
        """
        meanings = {
            'same': {
                'meaning': 'Harmony through similarity',
                'chinese': '比和',
                'favorability': 3,
                'advice': 'Stable but may lack dynamic energy'
            },
            'generating': {
                'meaning': 'Support and nourishment',
                'chinese': '相生',
                'favorability': 5,
                'advice': 'Favorable for growth and development'
            },
            'draining': {
                'meaning': 'Energy being drawn away',
                'chinese': '洩氣',
                'favorability': 2,
                'advice': 'May experience exhaustion or loss'
            },
            'controlling': {
                'meaning': 'Dominance and constraint',
                'chinese': '相剋',
                'favorability': 4,
                'advice': 'Good for overcoming obstacles, less good for harmony'
            },
            'controlled': {
                'meaning': 'Being restrained or opposed',
                'chinese': '受剋',
                'favorability': 1,
                'advice': 'Expect resistance and difficulties'
            },
            'neutral': {
                'meaning': 'No direct interaction',
                'chinese': '無關',
                'favorability': 3,
                'advice': 'Neither helps nor hinders'
            }
        }
        return meanings.get(relationship, meanings['neutral'])

    # =========================================================================
    # Palace Analysis
    # =========================================================================

    def analyze_palace(self, palace: Palace) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single palace.

        Args:
            palace: Palace to analyze

        Returns:
            Dictionary with palace analysis
        """
        analysis = {
            'number': palace.number,
            'direction': palace.direction.value if palace.direction else 'unknown',
            'element': palace.base_element.value if palace.base_element else 'unknown',
            'stems': {
                'earth': palace.earth_plate_stem,
                'heaven': palace.heaven_plate_stem,
            },
            'components': {},
            'favorability_score': 0,
            'warnings': [],
            'auspicious_factors': [],
        }

        score = 0

        # Analyze star
        if palace.star:
            analysis['components']['star'] = palace.star.chinese
            star_score = self.star_favorability.get(palace.star.chinese, 3)
            score += star_score
            if star_score >= 4:
                analysis['auspicious_factors'].append(f"{palace.star.chinese} - {palace.star.nature}")
            elif star_score <= 2:
                analysis['warnings'].append(f"{palace.star.chinese} is unfavorable")

        # Analyze gate
        if palace.gate:
            analysis['components']['gate'] = palace.gate.chinese
            gate_score = self.gate_favorability.get(palace.gate.chinese, 3)
            score += gate_score
            if gate_score >= 4:
                analysis['auspicious_factors'].append(f"{palace.gate.chinese} - auspicious gate")
            elif gate_score <= 1:
                analysis['warnings'].append(f"{palace.gate.chinese} - avoid this direction")

        # Analyze spirit
        if palace.spirit:
            analysis['components']['spirit'] = palace.spirit.chinese

        # Check for Three Wonders
        if palace.has_three_wonders():
            score += 2
            wonder = palace.heaven_plate_stem if palace.heaven_plate_stem in THREE_WONDERS else palace.earth_plate_stem
            analysis['auspicious_factors'].append(f"三奇 ({wonder}) present")

        # Check stem combinations
        stem_combo = palace.get_stem_combination_name()
        if stem_combo:
            analysis['stem_combination'] = stem_combo
            if stem_combo == '伏吟':
                analysis['warnings'].append('伏吟 - stagnation, delays likely')
                score -= 1
            elif stem_combo == '反吟':
                analysis['warnings'].append('反吟 - reversal, sudden changes')
                score -= 1

        # Analyze element relationships
        if palace.star and palace.base_element:
            star_palace_relation = self.analyze_element_relationship(
                palace.star.element, palace.base_element
            )
            analysis['star_palace_element'] = star_palace_relation

        # Analyze stem patterns (格局)
        if palace.heaven_plate_stem and palace.earth_plate_stem:
            stem_analysis = analyze_stem_combination(
                palace.heaven_plate_stem,
                palace.earth_plate_stem
            )
            analysis['stem_pattern'] = stem_analysis

            # Adjust score based on pattern
            if stem_analysis.get('category') == '吉格':
                score += 2
                if stem_analysis.get('pattern_name'):
                    analysis['auspicious_factors'].append(
                        f"格局: {stem_analysis['pattern_name']}"
                    )
            elif stem_analysis.get('category') == '凶格':
                score -= 2
                if stem_analysis.get('pattern_name'):
                    analysis['warnings'].append(
                        f"格局: {stem_analysis['pattern_name']}"
                    )

            # Element interaction
            element_interaction = get_stem_element_interaction(
                palace.heaven_plate_stem,
                palace.earth_plate_stem
            )
            analysis['element_interaction'] = element_interaction

        analysis['favorability_score'] = score
        return analysis

    # =========================================================================
    # Direction Analysis
    # =========================================================================

    def analyze_directions(self, plate: QimenPlate) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all directions in the plate.

        Args:
            plate: QimenPlate to analyze

        Returns:
            Dictionary mapping direction to analysis
        """
        directions = {}

        for palace_num in range(1, 10):
            palace = plate.get_palace(palace_num)
            if palace:
                analysis = self.analyze_palace(palace)
                direction_name = palace.direction.value if palace.direction else f"Palace {palace_num}"
                directions[direction_name] = analysis

        return directions

    def find_auspicious_directions(self, plate: QimenPlate) -> List[Dict[str, Any]]:
        """
        Find the most auspicious directions.

        Args:
            plate: QimenPlate to analyze

        Returns:
            List of auspicious directions sorted by favorability
        """
        results = []

        for palace_num in range(1, 10):
            palace = plate.get_palace(palace_num)
            if palace:
                analysis = self.analyze_palace(palace)
                if analysis['favorability_score'] >= 6:  # Threshold for "auspicious"
                    results.append({
                        'palace_number': palace_num,
                        'direction': analysis['direction'],
                        'score': analysis['favorability_score'],
                        'factors': analysis['auspicious_factors'],
                        'gate': analysis['components'].get('gate'),
                        'star': analysis['components'].get('star'),
                    })

        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def find_inauspicious_directions(self, plate: QimenPlate) -> List[Dict[str, Any]]:
        """
        Find directions to avoid.

        Args:
            plate: QimenPlate to analyze

        Returns:
            List of inauspicious directions with warnings
        """
        results = []

        for palace_num in range(1, 10):
            palace = plate.get_palace(palace_num)
            if palace:
                analysis = self.analyze_palace(palace)
                if analysis['favorability_score'] <= 4 or analysis['warnings']:
                    results.append({
                        'palace_number': palace_num,
                        'direction': analysis['direction'],
                        'score': analysis['favorability_score'],
                        'warnings': analysis['warnings'],
                        'gate': analysis['components'].get('gate'),
                        'star': analysis['components'].get('star'),
                    })

        # Sort by score ascending (worst first)
        results.sort(key=lambda x: x['score'])
        return results

    # =========================================================================
    # Special Conditions
    # =========================================================================

    def check_special_conditions(self, plate: QimenPlate) -> Dict[str, Any]:
        """
        Check for special conditions in the plate.

        Args:
            plate: QimenPlate to check

        Returns:
            Dictionary of special conditions found
        """
        conditions = {
            'fu_yin_palaces': [],      # 伏吟 - same stem
            'fan_yin_palaces': [],     # 反吟 - opposing stem
            'three_wonders_palaces': [],  # 三奇
            'gate_constraint': [],     # 門迫
            'star_gate_combinations': [],  # Special star-gate combos
        }

        for palace_num in range(1, 10):
            palace = plate.get_palace(palace_num)
            if not palace:
                continue

            # Check Fu Yin / Fan Yin
            stem_combo = palace.get_stem_combination_name()
            if stem_combo == '伏吟':
                conditions['fu_yin_palaces'].append(palace_num)
            elif stem_combo == '反吟':
                conditions['fan_yin_palaces'].append(palace_num)

            # Check Three Wonders
            if palace.has_three_wonders():
                conditions['three_wonders_palaces'].append({
                    'palace': palace_num,
                    'wonder': palace.heaven_plate_stem if palace.heaven_plate_stem in THREE_WONDERS else palace.earth_plate_stem
                })

            # Check gate constraint (門迫)
            # Gate constraint occurs when gate element is controlled by palace element
            if palace.gate and palace.base_element:
                gate_element = palace.gate.element
                relation = self.analyze_element_relationship(palace.base_element, gate_element)
                if relation == 'controlling':
                    conditions['gate_constraint'].append({
                        'palace': palace_num,
                        'gate': palace.gate.chinese,
                        'palace_element': palace.base_element.value,
                        'gate_element': gate_element.value,
                    })

            # Check notable star-gate combinations
            if palace.star and palace.gate:
                combo_name = self._get_star_gate_combo_name(palace.star, palace.gate)
                if combo_name:
                    conditions['star_gate_combinations'].append({
                        'palace': palace_num,
                        'name': combo_name,
                        'star': palace.star.chinese,
                        'gate': palace.gate.chinese,
                    })

        return conditions

    def _get_star_gate_combo_name(self, star: NineStar, gate: EightGate) -> Optional[str]:
        """
        Get the name of notable star-gate combinations.

        Args:
            star: The star in the palace
            gate: The gate in the palace

        Returns:
            Combination name or None if not notable
        """
        # Define notable combinations
        notable_combos = {
            ('天心', '開門'): '天心開門 - Most auspicious for new ventures',
            ('天任', '生門'): '天任生門 - Excellent for growth',
            ('天輔', '休門'): '天輔休門 - Good for recovery and rest',
            ('天蓬', '死門'): '天蓬死門 - Extremely unfavorable',
            ('天芮', '死門'): '天芮死門 - Avoid medical matters',
            ('天柱', '驚門'): '天柱驚門 - Risk of accidents',
        }

        key = (star.chinese, gate.chinese)
        return notable_combos.get(key)

    # =========================================================================
    # Query-Specific Analysis
    # =========================================================================

    def query_analysis(self, plate: QimenPlate, question_type: str) -> Dict[str, Any]:
        """
        Provide analysis tailored to a specific question type.

        Args:
            plate: QimenPlate to analyze
            question_type: Type of question (general, business, travel, etc.)

        Returns:
            Dictionary with question-specific analysis
        """
        base_analysis = self.analyze_plate(plate)

        # Add question-specific analysis
        if question_type == "business":
            return self._analyze_business(plate, base_analysis)
        elif question_type == "travel":
            return self._analyze_travel(plate, base_analysis)
        elif question_type == "health":
            return self._analyze_health(plate, base_analysis)
        elif question_type == "legal":
            return self._analyze_legal(plate, base_analysis)
        elif question_type == "relationship":
            return self._analyze_relationship(plate, base_analysis)
        elif question_type == "career":
            return self._analyze_career(plate, base_analysis)
        else:
            return base_analysis

    def _analyze_business(self, plate: QimenPlate, base: Dict) -> Dict[str, Any]:
        """Business-specific analysis."""
        analysis = base.copy()
        analysis['question_focus'] = 'business'

        # Look for Kai Gate (開門) and Sheng Gate (生門)
        kai_palace = plate.find_gate_palace('開門')
        sheng_palace = plate.find_gate_palace('生門')

        recommendations = []

        if kai_palace:
            palace = plate.get_palace(kai_palace)
            if palace:
                recommendations.append({
                    'direction': palace.direction.value,
                    'advice': 'Favorable for initiating new business deals',
                    'gate': '開門'
                })

        if sheng_palace:
            palace = plate.get_palace(sheng_palace)
            if palace:
                recommendations.append({
                    'direction': palace.direction.value,
                    'advice': 'Good for business growth and profit',
                    'gate': '生門'
                })

        analysis['business_recommendations'] = recommendations
        analysis['avoid_directions'] = [
            d for d in self.find_inauspicious_directions(plate)
            if d['gate'] in ['死門', '驚門']
        ]

        return analysis

    def _analyze_travel(self, plate: QimenPlate, base: Dict) -> Dict[str, Any]:
        """Travel-specific analysis."""
        analysis = base.copy()
        analysis['question_focus'] = 'travel'

        # Look for favorable travel gates
        favorable_gates = ['開門', '休門', '生門']
        unfavorable_gates = ['死門', '傷門', '驚門']

        safe_directions = []
        avoid_directions = []

        for palace_num in range(1, 10):
            palace = plate.get_palace(palace_num)
            if palace and palace.gate:
                gate_name = palace.gate.chinese
                if gate_name in favorable_gates:
                    safe_directions.append({
                        'direction': palace.direction.value,
                        'palace': palace_num,
                        'gate': gate_name,
                    })
                elif gate_name in unfavorable_gates:
                    avoid_directions.append({
                        'direction': palace.direction.value,
                        'palace': palace_num,
                        'gate': gate_name,
                        'warning': f'Avoid traveling {palace.direction.value}'
                    })

        analysis['safe_travel_directions'] = safe_directions
        analysis['avoid_travel_directions'] = avoid_directions

        return analysis

    def _analyze_health(self, plate: QimenPlate, base: Dict) -> Dict[str, Any]:
        """Health-specific analysis."""
        analysis = base.copy()
        analysis['question_focus'] = 'health'

        # Tianrui (天芮) is the illness star
        tianrui_palace = plate.find_star_palace('天芮')

        warnings = []
        recommendations = []

        if tianrui_palace:
            palace = plate.get_palace(tianrui_palace)
            if palace:
                warnings.append({
                    'element': palace.base_element.value,
                    'direction': palace.direction.value,
                    'advice': f'Illness energy concentrated in {palace.direction.value}'
                })

        # Look for Xiu Gate (休門) for recovery
        xiu_palace = plate.find_gate_palace('休門')
        if xiu_palace:
            palace = plate.get_palace(xiu_palace)
            if palace:
                recommendations.append({
                    'direction': palace.direction.value,
                    'advice': 'Favorable for rest and recovery'
                })

        analysis['health_warnings'] = warnings
        analysis['recovery_recommendations'] = recommendations

        return analysis

    def _analyze_legal(self, plate: QimenPlate, base: Dict) -> Dict[str, Any]:
        """Legal matters analysis."""
        analysis = base.copy()
        analysis['question_focus'] = 'legal'

        # Jing Gate (驚門) relates to legal/official matters
        jing_palace = plate.find_gate_palace('驚門')
        kai_palace = plate.find_gate_palace('開門')

        insights = []

        if jing_palace:
            palace = plate.get_palace(jing_palace)
            if palace:
                insights.append({
                    'type': 'caution',
                    'advice': 'Legal matters may bring unexpected developments'
                })

        if kai_palace:
            palace = plate.get_palace(kai_palace)
            if palace:
                insights.append({
                    'type': 'favorable',
                    'direction': palace.direction.value,
                    'advice': 'Good direction for filing documents or legal actions'
                })

        analysis['legal_insights'] = insights

        return analysis

    def _analyze_relationship(self, plate: QimenPlate, base: Dict) -> Dict[str, Any]:
        """Relationship analysis."""
        analysis = base.copy()
        analysis['question_focus'] = 'relationship'

        # Liu He (六合) spirit is favorable for relationships
        liuhe_palace = None
        for palace_num, spirit in plate.spirit_positions.items():
            if spirit and spirit.chinese == '六合':
                liuhe_palace = palace_num
                break

        insights = []

        if liuhe_palace:
            palace = plate.get_palace(liuhe_palace)
            if palace:
                insights.append({
                    'type': 'favorable',
                    'direction': palace.direction.value,
                    'advice': 'Favorable for relationships and partnerships'
                })

        # Look for favorable gates
        xiu_palace = plate.find_gate_palace('休門')
        if xiu_palace:
            palace = plate.get_palace(xiu_palace)
            if palace:
                insights.append({
                    'type': 'favorable',
                    'direction': palace.direction.value,
                    'advice': 'Good for peaceful resolution and harmony'
                })

        analysis['relationship_insights'] = insights

        return analysis

    def _analyze_career(self, plate: QimenPlate, base: Dict) -> Dict[str, Any]:
        """Career analysis."""
        analysis = base.copy()
        analysis['question_focus'] = 'career'

        # Kai Gate for new opportunities
        # Sheng Gate for growth
        # Tianxin star for wisdom in decisions

        recommendations = []

        kai_palace = plate.find_gate_palace('開門')
        if kai_palace:
            palace = plate.get_palace(kai_palace)
            if palace:
                recommendations.append({
                    'direction': palace.direction.value,
                    'type': 'new_opportunity',
                    'advice': 'Good for job interviews and new positions'
                })

        tianxin_palace = plate.find_star_palace('天心')
        if tianxin_palace:
            palace = plate.get_palace(tianxin_palace)
            if palace:
                recommendations.append({
                    'direction': palace.direction.value,
                    'type': 'wisdom',
                    'advice': 'Seek counsel or make important decisions'
                })

        analysis['career_recommendations'] = recommendations

        return analysis

    # =========================================================================
    # Comprehensive Plate Analysis
    # =========================================================================

    def analyze_plate(self, plate: QimenPlate) -> Dict[str, Any]:
        """
        Comprehensive analysis of an entire Qimen plate.

        Args:
            plate: QimenPlate to analyze

        Returns:
            Dictionary with complete plate analysis
        """
        analysis = {
            'configuration': {
                'dun_type': plate.dun_type.chinese,
                'yuan': plate.yuan.chinese,
                'ju_number': plate.ju_number,
                'solar_term': plate.solar_term.chinese,
            },
            'duty_elements': {
                'star': plate.duty_star.chinese if plate.duty_star else None,
                'gate': plate.duty_gate.chinese if plate.duty_gate else None,
                'palace': plate.duty_palace,
            },
            'directions': self.analyze_directions(plate),
            'auspicious': self.find_auspicious_directions(plate),
            'inauspicious': self.find_inauspicious_directions(plate),
            'special_conditions': self.check_special_conditions(plate),
            'overall_assessment': self._generate_overall_assessment(plate),
        }

        return analysis

    def _generate_overall_assessment(self, plate: QimenPlate) -> Dict[str, Any]:
        """
        Generate an overall assessment of the plate.

        Args:
            plate: QimenPlate to assess

        Returns:
            Dictionary with overall assessment
        """
        # Count favorable and unfavorable factors
        auspicious = self.find_auspicious_directions(plate)
        inauspicious = self.find_inauspicious_directions(plate)
        conditions = self.check_special_conditions(plate)

        favorable_count = len(auspicious)
        unfavorable_count = len(inauspicious)
        three_wonders_count = len(conditions['three_wonders_palaces'])

        # Analyze all stem patterns
        stem_pattern_summary = self.analyze_all_stem_patterns(plate)
        auspicious_patterns = stem_pattern_summary.get('auspicious_count', 0)
        inauspicious_patterns = stem_pattern_summary.get('inauspicious_count', 0)

        # Calculate overall score
        score = (favorable_count * 2) + (three_wonders_count * 3) + (auspicious_patterns * 2)
        score -= (unfavorable_count * 1.5) + (inauspicious_patterns * 2)

        # Generate overall rating
        if score >= 15:
            overall = "大吉 - Highly Auspicious"
            advice = "Excellent time for major undertakings. Heaven and Earth align in your favor."
        elif score >= 8:
            overall = "吉 - Auspicious"
            advice = "Favorable conditions. Proceed with confidence but remain mindful."
        elif score >= 3:
            overall = "小吉 - Moderately Auspicious"
            advice = "Generally favorable, though some caution is warranted."
        elif score >= -3:
            overall = "平 - Neutral"
            advice = "Mixed conditions. Exercise discernment and avoid major risks."
        elif score >= -8:
            overall = "小凶 - Moderately Inauspicious"
            advice = "Unfavorable trends present. Postpone major decisions if possible."
        else:
            overall = "凶 - Inauspicious"
            advice = "Avoid major undertakings. Focus on protection and preparation."

        return {
            'rating': overall,
            'advice': advice,
            'score': round(score, 1),
            'favorable_directions': favorable_count,
            'unfavorable_directions': unfavorable_count,
            'three_wonders_present': three_wonders_count,
            'auspicious_patterns': auspicious_patterns,
            'inauspicious_patterns': inauspicious_patterns,
            'special_notes': [
                n for n in [
                    "伏吟 present - expect delays and stagnation" if conditions['fu_yin_palaces'] else None,
                    "反吟 present - expect reversals and sudden changes" if conditions['fan_yin_palaces'] else None,
                    f"Gate constraint in palaces: {conditions['gate_constraint']}" if conditions['gate_constraint'] else None,
                ] if n
            ]
        }

    # =========================================================================
    # Stem Pattern Analysis
    # =========================================================================

    def analyze_all_stem_patterns(self, plate: QimenPlate) -> Dict[str, Any]:
        """
        Analyze all stem patterns across the entire plate.

        Args:
            plate: QimenPlate to analyze

        Returns:
            Dictionary with comprehensive stem pattern analysis
        """
        patterns = []
        auspicious_count = 0
        inauspicious_count = 0
        notable_patterns = []

        for palace_num in range(1, 10):
            palace = plate.get_palace(palace_num)
            if palace and palace.heaven_plate_stem and palace.earth_plate_stem:
                analysis = analyze_stem_combination(
                    palace.heaven_plate_stem,
                    palace.earth_plate_stem
                )

                if analysis.get('pattern_name'):
                    pattern_info = {
                        'palace': palace_num,
                        'direction': palace.direction.value if palace.direction else None,
                        'heaven_stem': palace.heaven_plate_stem,
                        'earth_stem': palace.earth_plate_stem,
                        'pattern_name': analysis['pattern_name'],
                        'category': analysis.get('category'),
                        'description': analysis.get('description'),
                    }
                    patterns.append(pattern_info)

                    if analysis.get('category') == '吉格':
                        auspicious_count += 1
                        notable_patterns.append(pattern_info)
                    elif analysis.get('category') == '凶格':
                        inauspicious_count += 1
                        notable_patterns.append(pattern_info)

        return {
            'all_patterns': patterns,
            'notable_patterns': notable_patterns,
            'auspicious_count': auspicious_count,
            'inauspicious_count': inauspicious_count,
            'total_named_patterns': len([p for p in patterns if p.get('pattern_name')]),
        }

    # =========================================================================
    # Spirit Analysis
    # =========================================================================

    def analyze_spirits(self, plate: QimenPlate) -> Dict[str, Any]:
        """
        Analyze the Eight Spirits positions and their influences.

        Args:
            plate: QimenPlate to analyze

        Returns:
            Dictionary with spirit analysis
        """
        spirit_interpretations = {
            '值符': {
                'nature': '吉神',
                'meaning': 'Chief deity, represents authority and protection',
                'favorable_for': ['Leadership', 'Official matters', 'Important decisions'],
            },
            '螣蛇': {
                'nature': '凶神',
                'meaning': 'Serpent spirit, represents deception and entanglement',
                'favorable_for': [],
                'warnings': ['Beware of deceit', 'Hidden enemies', 'Mental confusion'],
            },
            '太阴': {
                'nature': '吉神',
                'meaning': 'Great Yin, represents hidden support and secrets',
                'favorable_for': ['Secret matters', 'Female assistance', 'Concealment'],
            },
            '六合': {
                'nature': '吉神',
                'meaning': 'Six Harmonies, represents union and partnership',
                'favorable_for': ['Marriage', 'Partnerships', 'Negotiations', 'Mediation'],
            },
            '白虎': {
                'nature': '凶神',
                'meaning': 'White Tiger, represents violence and blood',
                'favorable_for': ['Military action'],
                'warnings': ['Injury risk', 'Violence', 'Blood-related matters'],
            },
            '玄武': {
                'nature': '凶神',
                'meaning': 'Dark Warrior, represents theft and loss',
                'favorable_for': [],
                'warnings': ['Theft', 'Deception', 'Loss of property'],
            },
            '九地': {
                'nature': '吉神',
                'meaning': 'Nine Earths, represents stability and grounding',
                'favorable_for': ['Defense', 'Hiding', 'Real estate', 'Agriculture'],
            },
            '九天': {
                'nature': '吉神',
                'meaning': 'Nine Heavens, represents rising and expansion',
                'favorable_for': ['Advancement', 'Promotion', 'Expansion', 'Travel upward'],
            },
        }

        analysis = {
            'spirit_positions': {},
            'auspicious_spirits': [],
            'inauspicious_spirits': [],
        }

        for palace_num, spirit in plate.spirit_positions.items():
            if spirit:
                palace = plate.get_palace(palace_num)
                direction = palace.direction.value if palace else None

                interp = spirit_interpretations.get(spirit.chinese, {})
                spirit_info = {
                    'palace': palace_num,
                    'direction': direction,
                    'spirit': spirit.chinese,
                    **interp
                }
                analysis['spirit_positions'][palace_num] = spirit_info

                if interp.get('nature') == '吉神':
                    analysis['auspicious_spirits'].append(spirit_info)
                else:
                    analysis['inauspicious_spirits'].append(spirit_info)

        return analysis

    # =========================================================================
    # Comprehensive Star-Gate Analysis
    # =========================================================================

    def analyze_star_gate_combinations(self, plate: QimenPlate) -> Dict[str, Any]:
        """
        Detailed analysis of all star-gate combinations.

        Args:
            plate: QimenPlate to analyze

        Returns:
            Dictionary with comprehensive star-gate analysis
        """
        combinations = []

        # Detailed star-gate interpretations
        combo_meanings = {
            ('天心', '開門'): {
                'rating': '大吉',
                'meaning': 'Supreme auspicious - Perfect for beginnings',
                'applications': ['New businesses', 'Important meetings', 'Medical treatment'],
            },
            ('天心', '休門'): {
                'rating': '吉',
                'meaning': 'Wisdom meets rest - Good for planning',
                'applications': ['Strategic planning', 'Rest', 'Recovery'],
            },
            ('天任', '生門'): {
                'rating': '大吉',
                'meaning': 'Benevolent growth - Excellent vitality',
                'applications': ['Health matters', 'New ventures', 'Childbirth'],
            },
            ('天輔', '景門'): {
                'rating': '吉',
                'meaning': 'Assistance illuminated - Good support',
                'applications': ['Education', 'Examinations', 'Cultural pursuits'],
            },
            ('天蓬', '死門'): {
                'rating': '大凶',
                'meaning': 'Double darkness - Extremely unfavorable',
                'applications': [],
                'warnings': ['Avoid all major actions', 'Risk of serious loss'],
            },
            ('天芮', '傷門'): {
                'rating': '凶',
                'meaning': 'Illness meets injury - Health risks',
                'applications': [],
                'warnings': ['Avoid medical procedures', 'Risk of complications'],
            },
            ('天柱', '驚門'): {
                'rating': '凶',
                'meaning': 'Destruction meets shock - Accidents likely',
                'applications': [],
                'warnings': ['Risk of accidents', 'Unexpected disasters'],
            },
            ('天沖', '傷門'): {
                'rating': '平',
                'meaning': 'Active conflict - Can be used for competition',
                'applications': ['Sports', 'Competitions'],
                'warnings': ['Risk of injury in physical activities'],
            },
            ('天英', '景門'): {
                'rating': '吉',
                'meaning': 'Double brightness - Fame and recognition',
                'applications': ['Public appearances', 'Marketing', 'Arts'],
            },
        }

        for palace_num in range(1, 10):
            palace = plate.get_palace(palace_num)
            if palace and palace.star and palace.gate:
                star_name = palace.star.chinese
                gate_name = palace.gate.chinese
                key = (star_name, gate_name)

                combo_info = {
                    'palace': palace_num,
                    'direction': palace.direction.value if palace.direction else None,
                    'star': star_name,
                    'gate': gate_name,
                    'star_nature': palace.star.nature if palace.star else None,
                    'gate_favorability': self.gate_favorability.get(gate_name, 3),
                }

                if key in combo_meanings:
                    combo_info.update(combo_meanings[key])
                else:
                    # Generate basic assessment
                    star_score = self.star_favorability.get(star_name, 3)
                    gate_score = self.gate_favorability.get(gate_name, 3)
                    total = star_score + gate_score

                    if total >= 8:
                        combo_info['rating'] = '吉'
                    elif total <= 4:
                        combo_info['rating'] = '凶'
                    else:
                        combo_info['rating'] = '平'

                combinations.append(combo_info)

        # Sort by rating
        rating_order = {'大吉': 0, '吉': 1, '平': 2, '凶': 3, '大凶': 4}
        combinations.sort(key=lambda x: rating_order.get(x.get('rating', '平'), 2))

        return {
            'combinations': combinations,
            'best_combinations': [c for c in combinations if c.get('rating') in ['大吉', '吉']][:3],
            'worst_combinations': [c for c in combinations if c.get('rating') in ['凶', '大凶']][:3],
        }

    # =========================================================================
    # Hour Analysis
    # =========================================================================

    def analyze_hour_influence(self, plate: QimenPlate) -> Dict[str, Any]:
        """
        Analyze the influence of the current hour.

        Args:
            plate: QimenPlate to analyze

        Returns:
            Dictionary with hour-specific analysis
        """
        hour_branch = plate.datetime_info.hour_branch
        hour_stem = plate.datetime_info.hour_stem

        hour_energy = {
            '子': {'period': '23:00-01:00', 'energy': 'Deep Yin', 'phase': 'Rest and regeneration'},
            '丑': {'period': '01:00-03:00', 'energy': 'Yin stabilizing', 'phase': 'Deep foundation work'},
            '寅': {'period': '03:00-05:00', 'energy': 'Yang birth', 'phase': 'New beginnings emerge'},
            '卯': {'period': '05:00-07:00', 'energy': 'Yang rising', 'phase': 'Gentle growth'},
            '辰': {'period': '07:00-09:00', 'energy': 'Transformation', 'phase': 'Dynamic change'},
            '巳': {'period': '09:00-11:00', 'energy': 'Wisdom', 'phase': 'Strategic thinking'},
            '午': {'period': '11:00-13:00', 'energy': 'Peak Yang', 'phase': 'Maximum activity'},
            '未': {'period': '13:00-15:00', 'energy': 'Yang declining', 'phase': 'Consolidation'},
            '申': {'period': '15:00-17:00', 'energy': 'Adaptation', 'phase': 'Problem solving'},
            '酉': {'period': '17:00-19:00', 'energy': 'Harvest', 'phase': 'Completion'},
            '戌': {'period': '19:00-21:00', 'energy': 'Protection', 'phase': 'Securing gains'},
            '亥': {'period': '21:00-23:00', 'energy': 'Return', 'phase': 'Preparation for rest'},
        }

        branch_info = hour_energy.get(hour_branch.chinese, {})

        return {
            'hour_stem': hour_stem.chinese,
            'hour_branch': hour_branch.chinese,
            'time_period': branch_info.get('period', 'Unknown'),
            'energy_quality': branch_info.get('energy', 'Unknown'),
            'current_phase': branch_info.get('phase', 'Unknown'),
            'duty_palace': plate.duty_palace,
            'recommendations': self._get_hour_recommendations(hour_branch.chinese),
        }

    def _get_hour_recommendations(self, branch: str) -> List[str]:
        """Get activity recommendations for an hour branch."""
        recommendations = {
            '子': ['Meditation', 'Planning', 'Accessing subconscious'],
            '丑': ['Detailed work', 'Persistent effort', 'Physical healing'],
            '寅': ['Spiritual practice', 'Bold initiatives', 'Exercise'],
            '卯': ['Gentle tasks', 'Diplomacy', 'Communication'],
            '辰': ['Transformative work', 'Creative projects', 'Change'],
            '巳': ['Strategic planning', 'Intellectual work', 'Negotiations'],
            '午': ['High-energy activities', 'Public matters', 'Leadership'],
            '未': ['Team activities', 'Consensus building', 'Nurturing'],
            '申': ['Problem-solving', 'Learning', 'Adaptable tasks'],
            '酉': ['Completing tasks', 'Quality control', 'Precision work'],
            '戌': ['Security matters', 'Protecting gains', 'Loyalty'],
            '亥': ['Gratitude', 'Enjoyment', 'Preparing for rest'],
        }
        return recommendations.get(branch, [])
