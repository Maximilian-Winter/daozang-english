"""
Taiyi Shenshu Five Generals System (太乙五将)

The Five Generals (五将) are the strategic commanders in Taiyi divination,
representing the five planets and their influences on military and state affairs.

From 太乙金鏡式經:
"五将者，太乙监将，并上下二目、主客大小将也。
 监将者，东方岁星之精，受木德之正气，王在春三月。
 上目者，南方荧感之精，受火德之正气，在天为阳，号始击将，属客。
 下目者，中宫镇星之精，受土德之正气，在地为阴，号文昌将，属主。
 客大将者，北方辰星之精，受水德之正气，主兵革。
 主大将者，西方太白之精，受金德之正气，主战斗。"

碼道長存 — The Way of Code endures
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from .spirits import TaiyiSpirit, SixteenSpirits


# =============================================================================
# Five Elements (五行) - For General Correspondences
# =============================================================================

class GeneralElement(Enum):
    """Five Elements associated with the Five Generals."""
    WOOD = ("木", "Jupiter", "岁星", "Spring")
    FIRE = ("火", "Mars", "荧惑", "Summer")
    EARTH = ("土", "Saturn", "镇星", "Four Seasons")
    METAL = ("金", "Venus", "太白", "Autumn")
    WATER = ("水", "Mercury", "辰星", "Winter")

    def __init__(self, chinese: str, planet: str, chinese_planet: str, season: str):
        self.chinese = chinese
        self.planet = planet
        self.chinese_planet = chinese_planet
        self.dominant_season = season


# =============================================================================
# General Type Enumeration (将类型)
# =============================================================================

class GeneralType(Enum):
    """
    Types of generals in the Taiyi system.

    监将 - Supervisor General (Taiyi itself)
    上目 - Upper Eye (始击将, belongs to Guest)
    下目 - Lower Eye (文昌将, belongs to Host)
    主大将 - Host Grand General
    客大将 - Guest Grand General
    主参将 - Host Deputy General
    客参将 - Guest Deputy General
    """
    SUPERVISOR = ("监将", "Supervisor", GeneralElement.WOOD, None)
    UPPER_EYE = ("上目/始击将", "Upper Eye / Shiji", GeneralElement.FIRE, "guest")
    LOWER_EYE = ("下目/文昌将", "Lower Eye / Wenchang", GeneralElement.EARTH, "host")
    HOST_GRAND = ("主大将", "Host Grand General", GeneralElement.METAL, "host")
    GUEST_GRAND = ("客大将", "Guest Grand General", GeneralElement.WATER, "guest")
    HOST_DEPUTY = ("主参将", "Host Deputy General", None, "host")
    GUEST_DEPUTY = ("客参将", "Guest Deputy General", None, "guest")

    def __init__(self, chinese: str, english: str,
                 element: Optional[GeneralElement], allegiance: Optional[str]):
        self.chinese_name = chinese
        self.english_name = english
        self.element = element
        self.allegiance = allegiance  # "host", "guest", or None

    @property
    def favors_host(self) -> bool:
        return self.allegiance == "host"

    @property
    def favors_guest(self) -> bool:
        return self.allegiance == "guest"


# =============================================================================
# General Position State (将位状态)
# =============================================================================

class GeneralState(Enum):
    """
    Positional states of generals affecting their fortune.

    From 太乙金鏡式經:
    - 发 (Fa): Prosperous, general is in a favorable position
    - 囚 (Qiu): Imprisoned, general is in an unfavorable position
    - 格 (Ge): Blocked, general faces opposition
    - 迫 (Po): Pressed, general is under pressure
    - 掩 (Yan): Covered, general is overwhelmed
    """
    PROSPEROUS = ("发", "Prosperous", "General in advantageous position")
    IMPRISONED = ("囚", "Imprisoned", "General in disadvantageous position")
    BLOCKED = ("格", "Blocked", "General faces opposition from Taiyi")
    PRESSED_INNER = ("内迫", "Inner Pressed", "General pressed from within")
    PRESSED_OUTER = ("外迫", "Outer Pressed", "General pressed from outside")
    COVERED = ("掩", "Covered", "General overwhelmed by opponent")
    NEUTRAL = ("和", "Neutral", "General in balanced position")

    def __init__(self, chinese: str, english: str, description: str):
        self.chinese_name = chinese
        self.english_name = english
        self.description = description

    @property
    def is_favorable(self) -> bool:
        return self in {GeneralState.PROSPEROUS, GeneralState.NEUTRAL}

    @property
    def is_unfavorable(self) -> bool:
        return self in {
            GeneralState.IMPRISONED, GeneralState.BLOCKED,
            GeneralState.PRESSED_INNER, GeneralState.PRESSED_OUTER,
            GeneralState.COVERED
        }


# =============================================================================
# General Data Class (将军数据)
# =============================================================================

@dataclass
class TaiyiGeneral:
    """
    Represents a general's complete position and state in a calculation.

    Attributes:
        general_type: Type of general (e.g., HOST_GRAND, GUEST_DEPUTY)
        palace: Palace number (1-9) where the general is located
        spirit: Associated spirit (for Upper/Lower Eye)
        state: Current positional state
        calculation: Associated calculation value (主算/客算)
    """
    general_type: GeneralType
    palace: int
    spirit: Optional[TaiyiSpirit] = None
    state: GeneralState = GeneralState.NEUTRAL
    calculation: Optional[int] = None

    @property
    def chinese_name(self) -> str:
        return self.general_type.chinese_name

    @property
    def element(self) -> Optional[GeneralElement]:
        return self.general_type.element

    @property
    def favors_host(self) -> bool:
        return self.general_type.favors_host

    @property
    def favors_guest(self) -> bool:
        return self.general_type.favors_guest

    @property
    def is_prosperous(self) -> bool:
        return self.state == GeneralState.PROSPEROUS

    @property
    def is_imprisoned(self) -> bool:
        return self.state == GeneralState.IMPRISONED

    def format_summary(self) -> str:
        """Format general information for display."""
        state_str = f" [{self.state.chinese_name}]" if self.state != GeneralState.NEUTRAL else ""
        calc_str = f" 算={self.calculation}" if self.calculation else ""
        spirit_str = f" ({self.spirit.chinese})" if self.spirit else ""
        return f"{self.chinese_name}: 宫{self.palace}{spirit_str}{state_str}{calc_str}"


# =============================================================================
# Five Generals System (五将系统)
# =============================================================================

class FiveGenerals:
    """
    Manages the Five Generals system for Taiyi calculations.

    The Five Generals form a strategic hierarchy:
    1. 监将 (Supervisor) - Taiyi itself, the supreme commander
    2. 上目/始击将 (Upper Eye/Shiji) - Represents guest offensive power
    3. 下目/文昌将 (Lower Eye/Wenchang) - Represents host defensive power
    4. 主大将 + 主参将 (Host Generals) - Host military commanders
    5. 客大将 + 客参将 (Guest Generals) - Guest military commanders
    """

    # Palace opposition pairs for determining 格 (blocked) state
    OPPOSITION_PAIRS = {1: 9, 9: 1, 2: 8, 8: 2, 3: 7, 7: 3, 4: 6, 6: 4}

    # Adjacent palaces for determining 迫 (pressed) state
    ADJACENT_PALACES = {
        1: [2, 6], 2: [1, 3, 7, 9], 3: [2, 4], 4: [3, 9],
        6: [1, 7], 7: [2, 6, 8], 8: [3, 7, 9], 9: [2, 4, 8]
    }

    def __init__(self):
        self.spirits = SixteenSpirits()

    # =========================================================================
    # Calculation Methods
    # =========================================================================

    def calculate_shiji_position(self, tianmu: TaiyiSpirit,
                                  accumulated: int) -> TaiyiSpirit:
        """
        Calculate the Shiji (始击将 / Upper Eye) position.

        From 太乙金鏡式經:
        始击将 follows the Tianmu through the sixteen spirits.

        Args:
            tianmu: Current Tianmu spirit
            accumulated: Accumulated time units

        Returns:
            Spirit representing Shiji position
        """
        # Shiji is typically derived from Tianmu's traversal
        # The exact algorithm varies by source; this follows one tradition
        shiji_steps = (tianmu.traversal_index + (accumulated % 8)) % 16
        return self.spirits.get_by_traversal_index(shiji_steps)

    def calculate_wenchang_position(self, tianmu: TaiyiSpirit,
                                     taiyi_palace: int) -> int:
        """
        Calculate the Wenchang (文昌将 / Lower Eye) palace position.

        From 太乙金鏡式經:
        "下目者，中宫镇星之精，受土德之正气，在地为阴，号文昌将，属主"

        Args:
            tianmu: Current Tianmu spirit
            taiyi_palace: Palace where Taiyi is located

        Returns:
            Palace number for Wenchang
        """
        # Wenchang often relates to the Tianmu's position in the nine palaces
        # This calculation maps spirit position to palace
        base_position = tianmu.traversal_index % 9
        if base_position == 5:
            base_position = 6  # Skip center palace
        if base_position == 0:
            base_position = 9
        return base_position

    def calculate_host_grand_general(self, host_calculation: int,
                                      palace_sequence: List[int]) -> int:
        """
        Calculate Host Grand General (主大将) palace position.

        Args:
            host_calculation: Host calculation value (主算)
            palace_sequence: Palace traversal sequence for current Dun

        Returns:
            Palace number for Host Grand General
        """
        palace_index = (host_calculation - 1) % 8
        return palace_sequence[palace_index]

    def calculate_guest_grand_general(self, guest_calculation: int,
                                        palace_sequence: List[int]) -> int:
        """
        Calculate Guest Grand General (客大将) palace position.

        Args:
            guest_calculation: Guest calculation value (客算)
            palace_sequence: Palace traversal sequence for current Dun

        Returns:
            Palace number for Guest Grand General
        """
        palace_index = (guest_calculation - 1) % 8
        return palace_sequence[palace_index]

    def calculate_host_deputy(self, host_calculation: int,
                               palace_sequence: List[int]) -> int:
        """
        Calculate Host Deputy General (主参将) palace position.

        From classical texts: Deputy is often at 3× the calculation mod cycle.

        Args:
            host_calculation: Host calculation value
            palace_sequence: Palace traversal sequence

        Returns:
            Palace number for Host Deputy General
        """
        palace_index = ((host_calculation * 3) - 1) % 8
        return palace_sequence[palace_index]

    def calculate_guest_deputy(self, guest_calculation: int,
                                palace_sequence: List[int]) -> int:
        """
        Calculate Guest Deputy General (客参将) palace position.

        Args:
            guest_calculation: Guest calculation value
            palace_sequence: Palace traversal sequence

        Returns:
            Palace number for Guest Deputy General
        """
        palace_index = ((guest_calculation * 3) - 1) % 8
        return palace_sequence[palace_index]

    # =========================================================================
    # State Determination Methods
    # =========================================================================

    def determine_general_state(self, general_palace: int,
                                 taiyi_palace: int,
                                 opponent_palace: Optional[int] = None,
                                 is_host: bool = True) -> GeneralState:
        """
        Determine the state of a general based on positional relationships.

        States:
        - 发 (Prosperous): Not in special negative relationship
        - 囚 (Imprisoned): In Taiyi's palace
        - 格 (Blocked): Opposite to Taiyi
        - 迫 (Pressed): Adjacent to Taiyi
        - 掩 (Covered): Opponent general in same palace

        Args:
            general_palace: Palace where the general is located
            taiyi_palace: Palace where Taiyi is located
            opponent_palace: Opponent general's palace (for 掩)
            is_host: Whether this is a host general

        Returns:
            GeneralState indicating the general's condition
        """
        # Check for 囚 (Imprisoned) - in Taiyi's palace
        if general_palace == taiyi_palace:
            return GeneralState.IMPRISONED

        # Check for 格 (Blocked) - opposite to Taiyi
        if self.OPPOSITION_PAIRS.get(general_palace) == taiyi_palace:
            return GeneralState.BLOCKED

        # Check for 掩 (Covered) - opponent in same palace
        if opponent_palace and general_palace == opponent_palace:
            return GeneralState.COVERED

        # Check for 迫 (Pressed) - adjacent to Taiyi
        adjacent = self.ADJACENT_PALACES.get(taiyi_palace, [])
        if general_palace in adjacent:
            # Determine inner vs outer based on palace numbers
            if general_palace < taiyi_palace:
                return GeneralState.PRESSED_INNER
            else:
                return GeneralState.PRESSED_OUTER

        # Default to prosperous
        return GeneralState.PROSPEROUS

    # =========================================================================
    # Complete Calculation
    # =========================================================================

    def calculate_all_generals(self,
                                taiyi_palace: int,
                                tianmu: TaiyiSpirit,
                                host_calculation: int,
                                guest_calculation: int,
                                palace_sequence: List[int],
                                accumulated: int) -> Dict[str, TaiyiGeneral]:
        """
        Calculate positions and states for all generals.

        Args:
            taiyi_palace: Palace where Taiyi is located
            tianmu: Current Tianmu spirit
            host_calculation: Host calculation value (主算)
            guest_calculation: Guest calculation value (客算)
            palace_sequence: Palace traversal sequence for current Dun
            accumulated: Accumulated time units

        Returns:
            Dictionary of all generals with their positions and states
        """
        # Calculate Shiji (始击将 / Upper Eye)
        shiji_spirit = self.calculate_shiji_position(tianmu, accumulated)

        # Calculate Wenchang (文昌将 / Lower Eye) palace
        wenchang_palace = self.calculate_wenchang_position(tianmu, taiyi_palace)

        # Calculate Grand Generals
        host_grand_palace = self.calculate_host_grand_general(
            host_calculation, palace_sequence
        )
        guest_grand_palace = self.calculate_guest_grand_general(
            guest_calculation, palace_sequence
        )

        # Calculate Deputy Generals
        host_deputy_palace = self.calculate_host_deputy(
            host_calculation, palace_sequence
        )
        guest_deputy_palace = self.calculate_guest_deputy(
            guest_calculation, palace_sequence
        )

        # Determine states
        host_grand_state = self.determine_general_state(
            host_grand_palace, taiyi_palace, guest_grand_palace, is_host=True
        )
        guest_grand_state = self.determine_general_state(
            guest_grand_palace, taiyi_palace, host_grand_palace, is_host=False
        )
        host_deputy_state = self.determine_general_state(
            host_deputy_palace, taiyi_palace, guest_deputy_palace, is_host=True
        )
        guest_deputy_state = self.determine_general_state(
            guest_deputy_palace, taiyi_palace, host_deputy_palace, is_host=False
        )

        # Create general objects
        generals = {
            "supervisor": TaiyiGeneral(
                general_type=GeneralType.SUPERVISOR,
                palace=taiyi_palace,
                state=GeneralState.NEUTRAL
            ),
            "upper_eye": TaiyiGeneral(
                general_type=GeneralType.UPPER_EYE,
                palace=shiji_spirit.traversal_index % 9 or 9,
                spirit=shiji_spirit,
                state=GeneralState.NEUTRAL
            ),
            "lower_eye": TaiyiGeneral(
                general_type=GeneralType.LOWER_EYE,
                palace=wenchang_palace,
                state=GeneralState.NEUTRAL
            ),
            "host_grand": TaiyiGeneral(
                general_type=GeneralType.HOST_GRAND,
                palace=host_grand_palace,
                state=host_grand_state,
                calculation=host_calculation
            ),
            "guest_grand": TaiyiGeneral(
                general_type=GeneralType.GUEST_GRAND,
                palace=guest_grand_palace,
                state=guest_grand_state,
                calculation=guest_calculation
            ),
            "host_deputy": TaiyiGeneral(
                general_type=GeneralType.HOST_DEPUTY,
                palace=host_deputy_palace,
                state=host_deputy_state
            ),
            "guest_deputy": TaiyiGeneral(
                general_type=GeneralType.GUEST_DEPUTY,
                palace=guest_deputy_palace,
                state=guest_deputy_state
            ),
        }

        return generals

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def analyze_battle_advantage(self,
                                  generals: Dict[str, TaiyiGeneral]) -> Dict[str, any]:
        """
        Analyze which side has the advantage based on general states.

        Args:
            generals: Dictionary of calculated generals

        Returns:
            Analysis dictionary with advantage assessment
        """
        host_score = 0
        guest_score = 0

        # Score based on general states
        for key, general in generals.items():
            if general.favors_host:
                if general.is_prosperous:
                    host_score += 2
                elif general.state.is_unfavorable:
                    host_score -= 1
            elif general.favors_guest:
                if general.is_prosperous:
                    guest_score += 2
                elif general.state.is_unfavorable:
                    guest_score -= 1

        # Determine advantage
        if host_score > guest_score:
            advantage = "host"
            advantage_chinese = "主人有利"
        elif guest_score > host_score:
            advantage = "guest"
            advantage_chinese = "客人有利"
        else:
            advantage = "neutral"
            advantage_chinese = "主客相当"

        return {
            "advantage": advantage,
            "advantage_chinese": advantage_chinese,
            "host_score": host_score,
            "guest_score": guest_score,
            "host_prosperous_count": sum(
                1 for g in generals.values()
                if g.favors_host and g.is_prosperous
            ),
            "guest_prosperous_count": sum(
                1 for g in generals.values()
                if g.favors_guest and g.is_prosperous
            ),
            "host_imprisoned_count": sum(
                1 for g in generals.values()
                if g.favors_host and g.is_imprisoned
            ),
            "guest_imprisoned_count": sum(
                1 for g in generals.values()
                if g.favors_guest and g.is_imprisoned
            ),
        }

    def check_special_formations(self,
                                  generals: Dict[str, TaiyiGeneral],
                                  taiyi_palace: int) -> List[Dict[str, str]]:
        """
        Check for special formations mentioned in classical texts.

        Special formations include:
        - 掩 (Cover): One general covers another
        - 迫 (Press): Generals in adjacent positions
        - 囚 (Imprison): General trapped in Taiyi's palace
        - 格 (Block): General opposite to Taiyi
        - 四郭固 (Four Walls Solid): Defensive formation

        Args:
            generals: Dictionary of calculated generals
            taiyi_palace: Palace where Taiyi is located

        Returns:
            List of detected special formations
        """
        formations = []

        # Check for 掩 (Cover) - Shiji covers Host Grand General
        if generals["upper_eye"].palace == generals["host_grand"].palace:
            formations.append({
                "name": "客目掩主大将",
                "english": "Guest Eye Covers Host Grand General",
                "effect": "unfavorable_host",
                "description": "始击将临主大将宫，主人不利"
            })

        # Check for 四郭固 (Four Walls Solid)
        host_palaces = {
            generals["host_grand"].palace,
            generals["host_deputy"].palace,
            generals["lower_eye"].palace
        }
        if len(host_palaces) >= 3 and all(p != taiyi_palace for p in host_palaces):
            formations.append({
                "name": "四郭固",
                "english": "Four Walls Solid",
                "effect": "favorable_host",
                "description": "主人防守坚固"
            })

        # Check for mutual blocking
        if (generals["host_grand"].state == GeneralState.BLOCKED and
            generals["guest_grand"].state == GeneralState.BLOCKED):
            formations.append({
                "name": "双方格",
                "english": "Mutual Blocking",
                "effect": "stalemate",
                "description": "双方相格，僵持不下"
            })

        return formations


# =============================================================================
# Convenience Function
# =============================================================================

def calculate_generals(taiyi_palace: int,
                        tianmu: TaiyiSpirit,
                        host_calculation: int,
                        guest_calculation: int,
                        palace_sequence: List[int],
                        accumulated: int) -> Dict[str, TaiyiGeneral]:
    """
    Convenience function to calculate all generals.

    Args:
        taiyi_palace: Palace where Taiyi is located
        tianmu: Current Tianmu spirit
        host_calculation: Host calculation value
        guest_calculation: Guest calculation value
        palace_sequence: Palace traversal sequence
        accumulated: Accumulated time units

    Returns:
        Dictionary of all calculated generals
    """
    system = FiveGenerals()
    return system.calculate_all_generals(
        taiyi_palace, tianmu, host_calculation, guest_calculation,
        palace_sequence, accumulated
    )


# End of generals module
