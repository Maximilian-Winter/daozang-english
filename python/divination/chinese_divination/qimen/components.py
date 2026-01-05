"""
Qimen Dunjia Components: Nine Stars, Eight Gates, Eight Spirits

This module defines the three main rotating components of the Qimen system:
- Nine Stars (九星): Celestial influences from the Big Dipper
- Eight Gates (八門): Human realm manifestations
- Eight Spirits (八神): Spiritual/deity influences

Each component follows the container pattern established in lunar_calendar.py
with individual entity classes and container classes with lookup methods.
"""

from enum import IntEnum
from typing import Optional, List, Dict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.core import Element, Polarity


# =============================================================================
# Nine Stars (九星)
# =============================================================================

class StarType(IntEnum):
    """Nine Stars enumeration with palace numbers as values"""
    TIAN_PENG = 1   # 天蓬
    TIAN_RUI = 2    # 天芮
    TIAN_CHONG = 3  # 天冲
    TIAN_FU = 4     # 天辅
    TIAN_QIN = 5    # 天禽 (center)
    TIAN_XIN = 6    # 天心
    TIAN_ZHU = 7    # 天柱
    TIAN_REN = 8    # 天任
    TIAN_YING = 9   # 天英


class NineStar:
    """
    Represents a single Nine Star with all its attributes.

    The Nine Stars derive from the Big Dipper (北斗七星) plus two additional stars.
    Each star has specific elemental, directional, and auspicious qualities.
    """

    def __init__(self, star_type: StarType, chinese: str, pinyin: str,
                 element: Element, polarity: Polarity,
                 nature: str, auspiciousness: str,
                 description: str, strategic_meaning: str):
        self.star_type = star_type
        self.chinese = chinese
        self.pinyin = pinyin
        self.element = element
        self.polarity = polarity
        self.nature = nature  # 吉/凶/平
        self.auspiciousness = auspiciousness
        self.description = description
        self.strategic_meaning = strategic_meaning
        self.base_palace = star_type.value  # Original palace position

    def __str__(self) -> str:
        return f"{self.chinese} ({self.pinyin})"

    def __repr__(self) -> str:
        return f"NineStar({self.chinese}, Palace {self.base_palace}, {self.nature})"

    @property
    def is_auspicious(self) -> bool:
        return '吉' in self.nature


class NineStars:
    """
    Container class for all Nine Stars.

    Provides lookup methods by Chinese name, star type, or palace number.
    """

    def __init__(self):
        self.stars = [
            NineStar(
                star_type=StarType.TIAN_PENG,
                chinese="天蓬", pinyin="tiān péng",
                element=Element.WATER, polarity=Polarity.YANG,
                nature="凶星", auspiciousness="Major inauspicious",
                description="Primordial chaos star, associated with the element Water. "
                           "Represents hidden depths, dangerous waters, and covert activities.",
                strategic_meaning="Covert operations, espionage, hidden dangers, theft matters"
            ),
            NineStar(
                star_type=StarType.TIAN_RUI,
                chinese="天芮", pinyin="tiān ruì",
                element=Element.EARTH, polarity=Polarity.YIN,
                nature="凶星", auspiciousness="Inauspicious",
                description="Disease star, associated with Earth. Represents stagnation, "
                           "illness, and obstacles in undertakings.",
                strategic_meaning="Health matters, obstacles, delays, illness concerns"
            ),
            NineStar(
                star_type=StarType.TIAN_CHONG,
                chinese="天冲", pinyin="tiān chōng",
                element=Element.WOOD, polarity=Polarity.YANG,
                nature="吉星", auspiciousness="Auspicious",
                description="Movement star, associated with Wood. Represents bold action, "
                           "breakthrough energy, and competitive spirit.",
                strategic_meaning="Aggressive action, military affairs, competition, sports"
            ),
            NineStar(
                star_type=StarType.TIAN_FU,
                chinese="天辅", pinyin="tiān fǔ",
                element=Element.WOOD, polarity=Polarity.YIN,
                nature="吉星", auspiciousness="Very auspicious",
                description="Assistance star, associated with Wood. Represents support, "
                           "culture, literature, and receiving help from others.",
                strategic_meaning="Education, writing, scholarly pursuits, receiving assistance"
            ),
            NineStar(
                star_type=StarType.TIAN_QIN,
                chinese="天禽", pinyin="tiān qín",
                element=Element.EARTH, polarity=Polarity.YANG,
                nature="吉星", auspiciousness="Auspicious",
                description="Central star, associated with Earth. Represents stability, "
                           "balance, and central authority. Located at the pivot.",
                strategic_meaning="Central control, stability, mediation, coordination"
            ),
            NineStar(
                star_type=StarType.TIAN_XIN,
                chinese="天心", pinyin="tiān xīn",
                element=Element.METAL, polarity=Polarity.YANG,
                nature="吉星", auspiciousness="Very auspicious",
                description="Heart star, associated with Metal. Represents decisive leadership, "
                           "authority, medicine, and clear judgment.",
                strategic_meaning="Leadership decisions, medical matters, authority, judgment"
            ),
            NineStar(
                star_type=StarType.TIAN_ZHU,
                chinese="天柱", pinyin="tiān zhù",
                element=Element.METAL, polarity=Polarity.YIN,
                nature="凶星", auspiciousness="Minor inauspicious",
                description="Pillar star, associated with Metal. Represents destruction, "
                           "revelations, endings, and breaking down structures.",
                strategic_meaning="Demolition, secrets revealed, ending matters, exposure"
            ),
            NineStar(
                star_type=StarType.TIAN_REN,
                chinese="天任", pinyin="tiān rèn",
                element=Element.EARTH, polarity=Polarity.YANG,
                nature="吉星", auspiciousness="Auspicious",
                description="Benevolence star, associated with Earth. Represents "
                           "trustworthiness, reliability, and supporting others.",
                strategic_meaning="Recruitment, trust building, agriculture, real estate"
            ),
            NineStar(
                star_type=StarType.TIAN_YING,
                chinese="天英", pinyin="tiān yīng",
                element=Element.FIRE, polarity=Polarity.YIN,
                nature="凶星", auspiciousness="Minor inauspicious",
                description="Brilliance star, associated with Fire. Represents fire hazards, "
                           "exposure, blood, and legal matters.",
                strategic_meaning="Fire hazards, blood matters, lawsuits, public exposure"
            )
        ]

        # Create lookup dictionaries
        self.by_chinese = {star.chinese: star for star in self.stars}
        self.by_type = {star.star_type: star for star in self.stars}
        self.by_palace = {star.base_palace: star for star in self.stars}

    def get_by_chinese(self, chinese: str) -> Optional[NineStar]:
        """Get star by Chinese name"""
        return self.by_chinese.get(chinese)

    def get_by_type(self, star_type: StarType) -> Optional[NineStar]:
        """Get star by StarType enum"""
        return self.by_type.get(star_type)

    def get_by_palace(self, palace: int) -> Optional[NineStar]:
        """Get star by its base palace number (1-9)"""
        return self.by_palace.get(palace)

    def get_auspicious_stars(self) -> List[NineStar]:
        """Get list of all auspicious stars"""
        return [star for star in self.stars if star.is_auspicious]

    def get_inauspicious_stars(self) -> List[NineStar]:
        """Get list of all inauspicious stars"""
        return [star for star in self.stars if not star.is_auspicious]


# =============================================================================
# Eight Gates (八門)
# =============================================================================

class GateType(IntEnum):
    """Eight Gates enumeration with palace numbers as values"""
    XIU = 1     # 休门 - Rest Gate
    SI = 2      # 死门 - Death Gate
    SHANG = 3   # 伤门 - Injury Gate
    DU = 4      # 杜门 - Block Gate
    # No gate at center (5)
    KAI = 6     # 开门 - Open Gate
    JING_ALARM = 7   # 惊门 - Fear/Alarm Gate
    SHENG = 8   # 生门 - Life Gate
    JING_VIEW = 9    # 景门 - View/Scenery Gate


class EightGate:
    """
    Represents a single Eight Gate with all its attributes.

    The Eight Gates represent different aspects of human affairs and activities.
    Each gate has specific favorable and unfavorable uses.
    """

    def __init__(self, gate_type: GateType, chinese: str, pinyin: str,
                 element: Element, nature: str,
                 description: str, favorable_for: List[str],
                 unfavorable_for: List[str]):
        self.gate_type = gate_type
        self.chinese = chinese
        self.pinyin = pinyin
        self.element = element
        self.nature = nature  # 吉门/凶门/平门
        self.description = description
        self.favorable_for = favorable_for
        self.unfavorable_for = unfavorable_for
        self.base_palace = gate_type.value

    def __str__(self) -> str:
        return f"{self.chinese} ({self.pinyin})"

    def __repr__(self) -> str:
        return f"EightGate({self.chinese}, Palace {self.base_palace}, {self.nature})"

    @property
    def is_auspicious(self) -> bool:
        return '吉' in self.nature


class EightGates:
    """
    Container class for all Eight Gates.

    Note: The center palace (5) has no gate.
    """

    def __init__(self):
        self.gates = [
            EightGate(
                gate_type=GateType.XIU,
                chinese="休门", pinyin="xiū mén",
                element=Element.WATER, nature="吉门",
                description="Rest Gate - Associated with rest, recuperation, and audience "
                           "with nobility. One of the three auspicious gates.",
                favorable_for=[
                    "Seeking positions", "meeting superiors", "rest and recovery",
                    "seeking favors", "peaceful negotiations", "vacation"
                ],
                unfavorable_for=["Aggressive actions", "litigation", "confrontation"]
            ),
            EightGate(
                gate_type=GateType.SI,
                chinese="死门", pinyin="sǐ mén",
                element=Element.EARTH, nature="凶门",
                description="Death Gate - Associated with death, endings, and burial matters. "
                           "Generally inauspicious but useful for specific purposes.",
                favorable_for=[
                    "Funerals", "burial", "hunting", "fishing", "executions",
                    "ending relationships", "pest control"
                ],
                unfavorable_for=[
                    "All positive endeavors", "travel", "starting projects",
                    "medical treatment", "celebrations"
                ]
            ),
            EightGate(
                gate_type=GateType.SHANG,
                chinese="伤门", pinyin="shāng mén",
                element=Element.WOOD, nature="凶门",
                description="Injury Gate - Associated with injury, conflict, and aggressive "
                           "action. Useful for competitive or confrontational matters.",
                favorable_for=[
                    "Debt collection", "competition", "arrests", "hunting",
                    "sports competitions", "confronting enemies"
                ],
                unfavorable_for=[
                    "Peaceful activities", "negotiations", "medical treatment",
                    "marriage", "partnerships"
                ]
            ),
            EightGate(
                gate_type=GateType.DU,
                chinese="杜门", pinyin="dù mén",
                element=Element.WOOD, nature="平门",
                description="Block Gate - Associated with blockage, concealment, and hiding. "
                           "Neutral gate useful for avoiding detection.",
                favorable_for=[
                    "Escaping danger", "hiding", "avoiding disaster",
                    "secret affairs", "covert operations", "retreat"
                ],
                unfavorable_for=[
                    "Opening new ventures", "public activities", "seeking help",
                    "making announcements", "expansion"
                ]
            ),
            EightGate(
                gate_type=GateType.KAI,
                chinese="开门", pinyin="kāi mén",
                element=Element.METAL, nature="吉门",
                description="Open Gate - Associated with opening, beginning, and official "
                           "matters. One of the three auspicious gates.",
                favorable_for=[
                    "Starting enterprises", "official petitions", "opening stores",
                    "travel", "business expansion", "job seeking", "appointments"
                ],
                unfavorable_for=["Concealment", "burial matters", "hiding", "retreat"]
            ),
            EightGate(
                gate_type=GateType.JING_ALARM,
                chinese="惊门", pinyin="jīng mén",
                element=Element.METAL, nature="凶门",
                description="Alarm Gate - Associated with fear, alarm, and litigation. "
                           "Generally inauspicious but useful for legal matters.",
                favorable_for=[
                    "Lawsuits", "verbal disputes", "competitive speech",
                    "debate", "arguments", "legal proceedings"
                ],
                unfavorable_for=[
                    "All peaceful matters", "rest", "meditation",
                    "partnerships", "celebrations"
                ]
            ),
            EightGate(
                gate_type=GateType.SHENG,
                chinese="生门", pinyin="shēng mén",
                element=Element.EARTH, nature="吉门",
                description="Life Gate - Associated with life, growth, and wealth generation. "
                           "One of the three auspicious gates, especially for business.",
                favorable_for=[
                    "Business ventures", "seeking wealth", "construction",
                    "marriage", "having children", "investments", "purchases"
                ],
                unfavorable_for=["Burial", "funerals", "endings", "letting go"]
            ),
            EightGate(
                gate_type=GateType.JING_VIEW,
                chinese="景门", pinyin="jǐng mén",
                element=Element.FIRE, nature="平门",
                description="View Gate - Associated with views, displays, and documents. "
                           "Neutral gate useful for public presentation.",
                favorable_for=[
                    "Examinations", "publishing", "artistic works", "advertising",
                    "public speaking", "performances", "celebrations"
                ],
                unfavorable_for=["Secret matters", "covert operations", "hiding"]
            )
        ]

        # Create lookup dictionaries
        self.by_chinese = {gate.chinese: gate for gate in self.gates}
        self.by_type = {gate.gate_type: gate for gate in self.gates}
        self.by_palace = {gate.base_palace: gate for gate in self.gates}

    def get_by_chinese(self, chinese: str) -> Optional[EightGate]:
        """Get gate by Chinese name"""
        return self.by_chinese.get(chinese)

    def get_by_type(self, gate_type: GateType) -> Optional[EightGate]:
        """Get gate by GateType enum"""
        return self.by_type.get(gate_type)

    def get_by_palace(self, palace: int) -> Optional[EightGate]:
        """Get gate by its base palace number (1-9, excluding 5)"""
        return self.by_palace.get(palace)

    def get_three_auspicious(self) -> List[EightGate]:
        """Get the three auspicious gates: 休门, 开门, 生门"""
        return [self.by_type[GateType.XIU],
                self.by_type[GateType.KAI],
                self.by_type[GateType.SHENG]]

    def get_auspicious_gates(self) -> List[EightGate]:
        """Get all auspicious gates"""
        return [gate for gate in self.gates if gate.is_auspicious]

    def get_inauspicious_gates(self) -> List[EightGate]:
        """Get all inauspicious gates"""
        return [gate for gate in self.gates if not gate.is_auspicious]


# =============================================================================
# Eight Spirits (八神)
# =============================================================================

class SpiritType(IntEnum):
    """Eight Spirits enumeration"""
    ZHI_FU = 1      # 值符 - Duty Spirit
    TENG_SHE = 2    # 螣蛇 - Serpent
    TAI_YIN = 3     # 太阴 - Greater Yin
    LIU_HE = 4      # 六合 - Six Harmony
    # No spirit at center (5)
    BAI_HU = 6      # 白虎 - White Tiger
    XUAN_WU = 7     # 玄武 - Dark Warrior
    JIU_DI = 8      # 九地 - Nine Earth
    JIU_TIAN = 9    # 九天 - Nine Heaven


class EightSpirit:
    """
    Represents a single Eight Spirit with all its attributes.

    The Eight Spirits represent different spiritual and cosmic influences
    that affect the outcome of events and actions.
    """

    def __init__(self, spirit_type: SpiritType, chinese: str, pinyin: str,
                 nature: str, description: str,
                 indicates: List[str], warnings: List[str]):
        self.spirit_type = spirit_type
        self.chinese = chinese
        self.pinyin = pinyin
        self.nature = nature  # 吉神/凶神
        self.description = description
        self.indicates = indicates
        self.warnings = warnings

    def __str__(self) -> str:
        return f"{self.chinese} ({self.pinyin})"

    def __repr__(self) -> str:
        return f"EightSpirit({self.chinese}, {self.nature})"

    @property
    def is_auspicious(self) -> bool:
        return '吉' in self.nature


class EightSpirits:
    """
    Container class for all Eight Spirits.

    Note: The center palace (5) has no spirit.
    Spirits rotate following the Duty Star (值符).
    """

    def __init__(self):
        self.spirits = [
            EightSpirit(
                spirit_type=SpiritType.ZHI_FU,
                chinese="值符", pinyin="zhí fú",
                nature="吉神",
                description="Duty Spirit - The leader of the eight spirits, representing "
                           "authority, leadership, and noble assistance.",
                indicates=[
                    "Noble person's help", "authority support", "leadership",
                    "official assistance", "protection from above"
                ],
                warnings=["Position may be challenged if afflicted by inauspicious stars"]
            ),
            EightSpirit(
                spirit_type=SpiritType.TENG_SHE,
                chinese="螣蛇", pinyin="téng shé",
                nature="凶神",
                description="Flying Serpent - Represents illusion, deception, and strange "
                           "occurrences. Associated with dreams and mental disturbance.",
                indicates=[
                    "Dreams and visions", "illusions", "weird events",
                    "deception", "confusion", "nightmares"
                ],
                warnings=["Fire hazards", "mental disturbance", "being deceived", "lies"]
            ),
            EightSpirit(
                spirit_type=SpiritType.TAI_YIN,
                chinese="太阴", pinyin="tài yīn",
                nature="吉神",
                description="Greater Yin - Represents hidden support, female assistance, "
                           "and concealed help. Good for secret matters.",
                indicates=[
                    "Female assistance", "secret help", "concealment",
                    "hidden benefactors", "mother figures"
                ],
                warnings=["Hidden enemies if afflicted", "secret opposition"]
            ),
            EightSpirit(
                spirit_type=SpiritType.LIU_HE,
                chinese="六合", pinyin="liù hé",
                nature="吉神",
                description="Six Harmony - Represents marriage, partnerships, and harmonious "
                           "agreements. Excellent for relationship matters.",
                indicates=[
                    "Marriage", "partnerships", "negotiations", "intermediaries",
                    "matchmaking", "agreements", "contracts"
                ],
                warnings=["Failed agreements if afflicted", "broken promises"]
            ),
            EightSpirit(
                spirit_type=SpiritType.BAI_HU,
                chinese="白虎", pinyin="bái hǔ",
                nature="凶神",
                description="White Tiger - Represents violence, metal, and aggressive energy. "
                           "Associated with injury, surgery, and military matters.",
                indicates=[
                    "Violence", "accidents", "surgery", "military affairs",
                    "metal-related injuries", "conflict"
                ],
                warnings=["Injury risk", "bloodshed", "death", "accidents"]
            ),
            EightSpirit(
                spirit_type=SpiritType.XUAN_WU,
                chinese="玄武", pinyin="xuán wǔ",
                nature="凶神",
                description="Dark Warrior - Represents theft, treachery, and hidden dangers. "
                           "Associated with loss and betrayal.",
                indicates=[
                    "Theft", "treachery", "secrets", "escape", "water dangers",
                    "hidden matters"
                ],
                warnings=["Betrayal", "loss of property", "deception by trusted people"]
            ),
            EightSpirit(
                spirit_type=SpiritType.JIU_DI,
                chinese="九地", pinyin="jiǔ dì",
                nature="吉神",
                description="Nine Earth - Represents stability, concealment, and foundation. "
                           "Good for defense and hidden activities.",
                indicates=[
                    "Hidden activities", "defense", "real estate", "foundation building",
                    "underground matters", "patience"
                ],
                warnings=["Stagnation if overused", "being too passive"]
            ),
            EightSpirit(
                spirit_type=SpiritType.JIU_TIAN,
                chinese="九天", pinyin="jiǔ tiān",
                nature="吉神",
                description="Nine Heaven - Represents high aspirations, authority, and expansion. "
                           "Good for public and elevated matters.",
                indicates=[
                    "Expansion", "high aims", "authority", "public matters",
                    "elevation", "fame", "growth"
                ],
                warnings=["Overreach if afflicted", "hubris", "falling from height"]
            )
        ]

        # Create lookup dictionaries
        self.by_chinese = {spirit.chinese: spirit for spirit in self.spirits}
        self.by_type = {spirit.spirit_type: spirit for spirit in self.spirits}

    def get_by_chinese(self, chinese: str) -> Optional[EightSpirit]:
        """Get spirit by Chinese name"""
        return self.by_chinese.get(chinese)

    def get_by_type(self, spirit_type: SpiritType) -> Optional[EightSpirit]:
        """Get spirit by SpiritType enum"""
        return self.by_type.get(spirit_type)

    def get_sequence(self) -> List[EightSpirit]:
        """Get spirits in rotation sequence starting from Zhi Fu"""
        sequence_order = [
            SpiritType.ZHI_FU, SpiritType.TENG_SHE, SpiritType.TAI_YIN,
            SpiritType.LIU_HE, SpiritType.BAI_HU, SpiritType.XUAN_WU,
            SpiritType.JIU_DI, SpiritType.JIU_TIAN
        ]
        return [self.by_type[st] for st in sequence_order]

    def get_auspicious_spirits(self) -> List[EightSpirit]:
        """Get all auspicious spirits"""
        return [spirit for spirit in self.spirits if spirit.is_auspicious]

    def get_inauspicious_spirits(self) -> List[EightSpirit]:
        """Get all inauspicious spirits"""
        return [spirit for spirit in self.spirits if not spirit.is_auspicious]
