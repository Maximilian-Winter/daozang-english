"""
Da Liu Ren Analysis Module (大六壬分析模組)

Interpretation and analysis tools for Liu Ren divination results.

This module provides:
- San Chuan (Three Transmissions) interpretation
- General position analysis
- Directional recommendations
- Query-specific interpretations

Reference: 六壬大全 (Ming Dynasty, 郭載騋)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from .constants import (
    BRANCH_ELEMENT, BRANCH_DIRECTION, STEM_ELEMENT,
    ELEMENT_GENERATES, ELEMENT_CONTROLS,
    SIX_CLASHES, SIX_HARMONIES, THREE_HARMONIES,
    does_control, does_generate,
)
from .components import TianJiang, SiKe, SanChuan, GeneralNature
from .plates import LiuRenPlate
from .patterns import (
    LessonPattern, SpecialPattern, TransmissionPattern,
    analyze_patterns, PatternAnalysis,
)


# =============================================================================
# Query Types (占問類型)
# =============================================================================

class QueryType(Enum):
    """Types of questions that can be asked in Liu Ren divination."""

    GENERAL = ("綜合", "General fortune", "Overall situation analysis")
    CAREER = ("事業", "Career/Business", "Work and career matters")
    WEALTH = ("財運", "Wealth/Finance", "Financial matters")
    RELATIONSHIP = ("感情", "Relationships", "Love and relationships")
    HEALTH = ("健康", "Health", "Health and medical matters")
    TRAVEL = ("出行", "Travel", "Journey and movement")
    LEGAL = ("訟獄", "Legal matters", "Lawsuits and disputes")
    LOST_ITEM = ("失物", "Lost items", "Finding lost objects")
    PERSON = ("尋人", "Missing person", "Finding missing people")
    WEATHER = ("天氣", "Weather", "Weather prediction")
    EXAM = ("考試", "Examination", "Tests and examinations")
    MARRIAGE = ("婚姻", "Marriage", "Marriage and partnerships")

    def __init__(self, chinese: str, english: str, description: str):
        self.chinese = chinese
        self.english = english
        self._description = description


# =============================================================================
# Favorability Rating (吉凶評級)
# =============================================================================

class Favorability(Enum):
    """Overall favorability rating."""

    VERY_AUSPICIOUS = ("大吉", 5, "Highly favorable")
    AUSPICIOUS = ("吉", 4, "Favorable")
    NEUTRAL = ("平", 3, "Neutral")
    INAUSPICIOUS = ("凶", 2, "Unfavorable")
    VERY_INAUSPICIOUS = ("大凶", 1, "Highly unfavorable")

    def __init__(self, chinese: str, value: int, english: str):
        self.chinese = chinese
        self._value = value
        self.english = english


# =============================================================================
# Direction Analysis (方位分析)
# =============================================================================

@dataclass
class DirectionalAdvice:
    """Advice for a specific direction."""
    direction: str
    direction_chinese: str
    favorability: Favorability
    reason: str
    associated_branch: str
    associated_general: Optional[str] = None


# =============================================================================
# LiuRenAnalyzer Class (六壬分析器)
# =============================================================================

class LiuRenAnalyzer:
    """
    Analyzer for Liu Ren divination results.

    Provides interpretation and analysis of plate calculations.
    """

    def __init__(self):
        """Initialize the analyzer with reference data."""
        self._init_general_keywords()
        self._init_direction_map()

    def _init_general_keywords(self) -> None:
        """Initialize keyword associations for generals."""
        self.general_keywords: Dict[str, List[str]] = {
            '貴人': ['尊貴', '貴人相助', '升遷', '喜事'],
            '騰蛇': ['虛驚', '怪異', '夢境', '變化'],
            '朱雀': ['口舌', '文書', '消息', '是非'],
            '六合': ['婚姻', '合作', '交易', '媒介'],
            '勾陳': ['田土', '牢獄', '爭訟', '遲滯'],
            '青龍': ['財帛', '喜慶', '進益', '貴人'],
            '天后': ['婦女', '陰私', '暗昧', '內助'],
            '太陰': ['藏匿', '暗助', '女性', '陰謀'],
            '玄武': ['盜賊', '失亡', '欺詐', '曖昧'],
            '太常': ['飲食', '禮儀', '衣冠', '宴會'],
            '白虎': ['喪亡', '凶災', '血光', '軍旅'],
            '天空': ['空亡', '欺詐', '虛假', '不實'],
        }

    def _init_direction_map(self) -> None:
        """Initialize direction mappings."""
        self.direction_branches: Dict[str, List[str]] = {
            '北': ['子'],
            '東北': ['丑', '寅'],
            '東': ['卯'],
            '東南': ['辰', '巳'],
            '南': ['午'],
            '西南': ['未', '申'],
            '西': ['酉'],
            '西北': ['戌', '亥'],
        }

    # =========================================================================
    # San Chuan Analysis (三傳分析)
    # =========================================================================

    def analyze_san_chuan(self, san_chuan: SanChuan,
                         plate: Optional[LiuRenPlate] = None) -> Dict[str, Any]:
        """
        Analyze the Three Transmissions.

        The Three Transmissions represent:
        - 初傳: Beginning/cause
        - 中傳: Development/process
        - 末傳: Outcome/result

        Args:
            san_chuan: The Three Transmissions
            plate: Optional full plate for context

        Returns:
            Analysis dictionary
        """
        analysis = {
            'overview': self._get_san_chuan_overview(san_chuan),
            'chu_chuan': self._analyze_transmission(san_chuan.chu_chuan, '初傳', '開端'),
            'zhong_chuan': self._analyze_transmission(san_chuan.zhong_chuan, '中傳', '過程'),
            'mo_chuan': self._analyze_transmission(san_chuan.mo_chuan, '末傳', '結果'),
            'flow': self._analyze_transmission_flow(san_chuan),
            'derivation': san_chuan.derivation_method,
        }

        # Add general analysis if plate available
        if plate:
            analysis['generals'] = {
                'chu': plate.general_positions.get(san_chuan.chu_chuan),
                'zhong': plate.general_positions.get(san_chuan.zhong_chuan),
                'mo': plate.general_positions.get(san_chuan.mo_chuan),
            }

        return analysis

    def _get_san_chuan_overview(self, san_chuan: SanChuan) -> str:
        """Generate overview of Three Transmissions."""
        elements = san_chuan.get_elements()

        if san_chuan.is_all_same_element():
            return f"三傳同屬{elements[0]}行，氣勢專一"

        if san_chuan.has_repeated_branch():
            return "三傳有重見支，事有反覆"

        # Check element relationships
        if all(ELEMENT_GENERATES.get(elements[i]) == elements[i+1]
               for i in range(2)):
            return "三傳相生，順利發展"

        return "三傳各異，事態複雜"

    def _analyze_transmission(self, branch: str, name: str,
                             meaning: str) -> Dict[str, Any]:
        """Analyze a single transmission."""
        element = BRANCH_ELEMENT.get(branch, '')
        direction = BRANCH_DIRECTION.get(branch, '')

        return {
            'branch': branch,
            'element': element,
            'direction': direction,
            'name': name,
            'meaning': meaning,
            'interpretation': f"{name}落{branch}支，{direction}方，屬{element}行",
        }

    def _analyze_transmission_flow(self, san_chuan: SanChuan) -> Dict[str, Any]:
        """Analyze the flow between transmissions."""
        branches = san_chuan.get_all_branches()
        elements = san_chuan.get_elements()

        # Check relationships between consecutive transmissions
        relationships = []
        for i in range(2):
            rel = self._get_element_relationship(elements[i], elements[i+1])
            relationships.append(rel)

        return {
            'chu_to_zhong': relationships[0] if relationships else '無',
            'zhong_to_mo': relationships[1] if len(relationships) > 1 else '無',
            'overall': self._summarize_flow(relationships),
        }

    def _get_element_relationship(self, e1: str, e2: str) -> str:
        """Get the relationship between two elements."""
        if e1 == e2:
            return '比和'
        if ELEMENT_GENERATES.get(e1) == e2:
            return '相生'
        if ELEMENT_CONTROLS.get(e1) == e2:
            return '相剋'
        if ELEMENT_GENERATES.get(e2) == e1:
            return '被生'
        if ELEMENT_CONTROLS.get(e2) == e1:
            return '被剋'
        return '無關'

    def _summarize_flow(self, relationships: List[str]) -> str:
        """Summarize the transmission flow."""
        if all(r == '相生' for r in relationships):
            return '全程順暢，漸入佳境'
        if all(r == '相剋' for r in relationships):
            return '阻礙重重，困難不斷'
        if '被剋' in relationships:
            return '中途受阻，需謹慎行事'
        return '起伏交錯，順逆參半'

    # =========================================================================
    # General Analysis (天將分析)
    # =========================================================================

    def analyze_generals(self, plate: LiuRenPlate) -> Dict[str, Any]:
        """
        Analyze the Heavenly Generals' positions.

        Args:
            plate: The complete LiuRenPlate

        Returns:
            Analysis of general positions and their meanings
        """
        analysis = {
            'noble_person': self._analyze_noble_person(plate),
            'transmission_generals': self._analyze_transmission_generals(plate),
            'favorable_positions': [],
            'unfavorable_positions': [],
        }

        # Categorize positions by favorability
        for branch, general in plate.general_positions.items():
            if general.is_auspicious:
                analysis['favorable_positions'].append({
                    'branch': branch,
                    'general': general.chinese,
                    'keywords': self.general_keywords.get(general.chinese, []),
                })
            else:
                analysis['unfavorable_positions'].append({
                    'branch': branch,
                    'general': general.chinese,
                    'keywords': self.general_keywords.get(general.chinese, []),
                })

        return analysis

    def _analyze_noble_person(self, plate: LiuRenPlate) -> Dict[str, Any]:
        """Analyze the Noble Person position."""
        noble = plate.noble_person
        branch = plate.noble_person_branch

        return {
            'branch': branch,
            'direction': BRANCH_DIRECTION.get(branch, ''),
            'element': BRANCH_ELEMENT.get(branch, ''),
            'is_daytime': plate.is_daytime,
            'interpretation': (
                f"{'晝' if plate.is_daytime else '夜'}貴人在{branch}位，"
                f"主{BRANCH_DIRECTION.get(branch, '')}方貴人相助"
            ),
        }

    def _analyze_transmission_generals(self, plate: LiuRenPlate) -> Dict[str, Any]:
        """Analyze the generals at transmission positions."""
        result = {}

        for name, branch in [
            ('初傳', plate.san_chuan.chu_chuan),
            ('中傳', plate.san_chuan.zhong_chuan),
            ('末傳', plate.san_chuan.mo_chuan),
        ]:
            general = plate.general_positions.get(branch)
            if general:
                result[name] = {
                    'branch': branch,
                    'general': general.chinese,
                    'is_auspicious': general.is_auspicious,
                    'keywords': self.general_keywords.get(general.chinese, []),
                }
            else:
                result[name] = {
                    'branch': branch,
                    'general': None,
                    'is_auspicious': None,
                    'keywords': [],
                }

        return result

    # =========================================================================
    # Directional Advice (方位建議)
    # =========================================================================

    def get_directional_advice(self, plate: LiuRenPlate) -> Dict[str, DirectionalAdvice]:
        """
        Get directional recommendations.

        Analyzes which directions are favorable or unfavorable
        based on the plate configuration.

        Args:
            plate: The complete LiuRenPlate

        Returns:
            Dictionary of direction to DirectionalAdvice
        """
        advice: Dict[str, DirectionalAdvice] = {}

        for direction, branches in self.direction_branches.items():
            # Find the most significant branch in this direction
            primary_branch = branches[0]

            # Get the general at this position
            general = plate.general_positions.get(primary_branch)

            # Determine favorability
            favorability, reason = self._evaluate_direction(
                primary_branch, general, plate
            )

            advice[direction] = DirectionalAdvice(
                direction=direction,
                direction_chinese=direction,
                favorability=favorability,
                reason=reason,
                associated_branch=primary_branch,
                associated_general=general.chinese if general else None,
            )

        return advice

    def _evaluate_direction(self, branch: str,
                           general: Optional[TianJiang],
                           plate: LiuRenPlate) -> tuple:
        """Evaluate a direction's favorability."""
        reasons = []

        # Base on general
        if general:
            if general.is_auspicious:
                base = Favorability.AUSPICIOUS
                reasons.append(f"有{general.chinese}吉神")
            else:
                base = Favorability.INAUSPICIOUS
                reasons.append(f"有{general.chinese}凶神")
        else:
            base = Favorability.NEUTRAL
            reasons.append("無特殊神煞")

        # Check if in三傳
        if branch in plate.san_chuan.get_all_branches():
            if branch == plate.san_chuan.mo_chuan:
                reasons.append("為末傳結果方位")
            else:
                reasons.append("為三傳涉及方位")

        # Check clash with day
        if SIX_CLASHES.get(plate.day_branch) == branch:
            base = Favorability.INAUSPICIOUS
            reasons.append("與日支相衝")

        # Check harmony with day
        if SIX_HARMONIES.get(plate.day_branch) == branch:
            if base == Favorability.NEUTRAL:
                base = Favorability.AUSPICIOUS
            reasons.append("與日支六合")

        return (base, '，'.join(reasons))

    # =========================================================================
    # Query-Specific Analysis (專項分析)
    # =========================================================================

    def query(self, plate: LiuRenPlate,
             query_type: QueryType = QueryType.GENERAL) -> Dict[str, Any]:
        """
        Perform query-specific analysis.

        Args:
            plate: The complete LiuRenPlate
            query_type: The type of question being asked

        Returns:
            Query-specific analysis dictionary
        """
        # Base analysis
        analysis = {
            'query_type': query_type.chinese,
            'overall_favorability': self._get_overall_favorability(plate),
            'key_points': self._get_key_points(plate, query_type),
            'advice': self._get_specific_advice(plate, query_type),
        }

        # Add pattern analysis
        pattern_analysis = analyze_patterns(plate)
        analysis['patterns'] = pattern_analysis.get_summary()

        return analysis

    def _get_overall_favorability(self, plate: LiuRenPlate) -> Favorability:
        """Calculate overall favorability rating."""
        score = 3  # Start neutral

        # Check pattern
        if plate.lesson_pattern in ['伏吟課', '反吟課']:
            score -= 1

        # Check三傳 generals
        for branch in plate.san_chuan.get_all_branches():
            general = plate.general_positions.get(branch)
            if general:
                if general.is_auspicious:
                    score += 0.5
                else:
                    score -= 0.5

        # Check Noble Person
        noble = plate.noble_person
        if noble and noble.is_auspicious:
            score += 0.5

        # Convert to Favorability
        if score >= 4.5:
            return Favorability.VERY_AUSPICIOUS
        elif score >= 3.5:
            return Favorability.AUSPICIOUS
        elif score >= 2.5:
            return Favorability.NEUTRAL
        elif score >= 1.5:
            return Favorability.INAUSPICIOUS
        else:
            return Favorability.VERY_INAUSPICIOUS

    def _get_key_points(self, plate: LiuRenPlate,
                       query_type: QueryType) -> List[str]:
        """Get key analysis points for the query."""
        points = []

        # Pattern-based points
        if '伏吟' in plate.special_patterns:
            points.append("伏吟格局，事難推進，宜守不宜動")
        if '反吟' in plate.special_patterns:
            points.append("反吟格局，事多反覆，變化較大")

        # Transmission-based points
        mo_general = plate.general_positions.get(plate.san_chuan.mo_chuan)
        if mo_general:
            if mo_general.is_auspicious:
                points.append(f"末傳見{mo_general.chinese}，結果吉利")
            else:
                points.append(f"末傳見{mo_general.chinese}，需防不利")

        # Query-specific points
        if query_type == QueryType.WEALTH:
            qinglong = self._find_general_position(plate, '青龍')
            if qinglong:
                points.append(f"青龍在{qinglong}位，財運方位在{BRANCH_DIRECTION.get(qinglong, '')}")

        elif query_type == QueryType.RELATIONSHIP:
            liuhe = self._find_general_position(plate, '六合')
            tianhou = self._find_general_position(plate, '天后')
            if liuhe:
                points.append(f"六合在{liuhe}位，主婚姻和合")
            if tianhou:
                points.append(f"天后在{tianhou}位，主女性緣分")

        return points

    def _find_general_position(self, plate: LiuRenPlate,
                              general_name: str) -> Optional[str]:
        """Find which branch a specific general is at."""
        for branch, general in plate.general_positions.items():
            if general.chinese == general_name:
                return branch
        return None

    def _get_specific_advice(self, plate: LiuRenPlate,
                            query_type: QueryType) -> List[str]:
        """Get query-specific advice."""
        advice = []

        # General direction advice
        directions = self.get_directional_advice(plate)
        favorable_dirs = [
            d.direction for d in directions.values()
            if d.favorability in [Favorability.AUSPICIOUS, Favorability.VERY_AUSPICIOUS]
        ]
        if favorable_dirs:
            advice.append(f"有利方位：{', '.join(favorable_dirs)}")

        # Query-specific advice
        if query_type == QueryType.CAREER:
            advice.append(self._get_career_advice(plate))
        elif query_type == QueryType.WEALTH:
            advice.append(self._get_wealth_advice(plate))
        elif query_type == QueryType.RELATIONSHIP:
            advice.append(self._get_relationship_advice(plate))
        elif query_type == QueryType.TRAVEL:
            advice.append(self._get_travel_advice(plate))
        elif query_type == QueryType.HEALTH:
            advice.append(self._get_health_advice(plate))

        return [a for a in advice if a]  # Filter empty strings

    def _get_career_advice(self, plate: LiuRenPlate) -> str:
        """Get career-specific advice."""
        noble = plate.noble_person_branch
        return f"貴人在{noble}位，求職升遷宜往{BRANCH_DIRECTION.get(noble, '')}方"

    def _get_wealth_advice(self, plate: LiuRenPlate) -> str:
        """Get wealth-specific advice."""
        qinglong_pos = self._find_general_position(plate, '青龍')
        if qinglong_pos:
            return f"青龍在{qinglong_pos}，求財宜往{BRANCH_DIRECTION.get(qinglong_pos, '')}方"
        return "青龍不明顯，求財宜謹慎"

    def _get_relationship_advice(self, plate: LiuRenPlate) -> str:
        """Get relationship-specific advice."""
        liuhe_pos = self._find_general_position(plate, '六合')
        if liuhe_pos:
            return f"六合在{liuhe_pos}位，姻緣宜在{BRANCH_DIRECTION.get(liuhe_pos, '')}方尋覓"
        return "六合位置需詳察"

    def _get_travel_advice(self, plate: LiuRenPlate) -> str:
        """Get travel-specific advice."""
        # Check for白虎 (danger)
        baihu_pos = self._find_general_position(plate, '白虎')
        if baihu_pos:
            return f"白虎在{baihu_pos}位，出行宜避{BRANCH_DIRECTION.get(baihu_pos, '')}方"
        return "出行無大礙，依三傳方位擇吉"

    def _get_health_advice(self, plate: LiuRenPlate) -> str:
        """Get health-specific advice."""
        tengsha_pos = self._find_general_position(plate, '騰蛇')
        if tengsha_pos:
            return f"騰蛇在{tengsha_pos}位，主虛驚，注意{BRANCH_ELEMENT.get(tengsha_pos, '')}行所主臟腑"
        return "健康無大礙，依時令調養"

    # =========================================================================
    # Complete Analysis (完整分析)
    # =========================================================================

    def analyze(self, plate: LiuRenPlate) -> Dict[str, Any]:
        """
        Perform complete analysis of a Liu Ren plate.

        Args:
            plate: The complete LiuRenPlate

        Returns:
            Comprehensive analysis dictionary
        """
        # Get pattern analysis
        pattern_analysis = analyze_patterns(plate)

        return {
            'plate_summary': plate.get_summary(),
            'patterns': pattern_analysis.get_summary(),
            'san_chuan_analysis': self.analyze_san_chuan(plate.san_chuan, plate),
            'general_analysis': self.analyze_generals(plate),
            'directional_advice': {
                d: {
                    'favorability': advice.favorability.chinese,
                    'reason': advice.reason,
                    'general': advice.associated_general,
                }
                for d, advice in self.get_directional_advice(plate).items()
            },
            'overall_favorability': self._get_overall_favorability(plate).chinese,
        }
