"""
Taiyi Shenshu Eight Gates System (太乙八門)

The Eight Gates (八門) are the pathways of fortune and misfortune in Taiyi divination.
Three are auspicious (三吉門), five are inauspicious.

From 太乙金鏡式經:
"推八門用法：開門、休門、生門謂之三吉門"
"李淳風云：常以開門加大乙，即太乙之八門也。
 又以開門加主大將，即主大將之八門也。
 又以開門加客大將，即客大將之八門也。"

Gate Calculation Algorithm:
"置上元甲子以来距所求积年，求岁计八门，
 以大游纪法七百二十去之；不尽，以三分纪法二百四十除之；
 余以三十约之，为直门数"

碼道長存 — The Way of Code endures
"""

from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from .constants import TaiyiConstants


# =============================================================================
# Gate Constants
# =============================================================================

class GateConstants:
    """Constants for gate calculations."""
    DA_YOU_JI_FA = 720      # 大游紀法 - Great Wandering Era Law
    SAN_FEN_JI_FA = 240     # 三分紀法 - Three Division Era Law
    GATE_CYCLE = 30         # 門周期 - Gate Cycle (30 years per gate)
    NUM_GATES = 8           # 八門 - Number of gates


# =============================================================================
# Gate Type Enumeration (門類型)
# =============================================================================

class GateType(IntEnum):
    """
    Eight Gates in their traversal order.

    Traversal sequence: 開 → 休 → 生 → 傷 → 杜 → 景 → 死 → 驚
    This follows the Luoshu palace arrangement clockwise.
    """
    OPEN = 1        # 開門 - Palace 1 (Kan/North)
    REST = 2        # 休門 - Palace 8 (Gen/Northeast)
    LIFE = 3        # 生門 - Palace 3 (Zhen/East)
    WOUND = 4       # 傷門 - Palace 4 (Xun/Southeast)
    BLOCK = 5       # 杜門 - Palace 9 (Li/South)
    VIEW = 6        # 景門 - Palace 2 (Kun/Southwest)
    DEATH = 7       # 死門 - Palace 7 (Dui/West)
    FEAR = 8        # 驚門 - Palace 6 (Qian/Northwest)


class GateNature(Enum):
    """Classification of gates by auspiciousness."""
    AUSPICIOUS = ("吉門", "Auspicious")
    INAUSPICIOUS = ("凶門", "Inauspicious")
    NEUTRAL = ("中門", "Neutral")

    def __init__(self, chinese: str, english: str):
        self.chinese = chinese
        self.english = english


# =============================================================================
# Gate Data Structure
# =============================================================================

@dataclass
class TaiyiGate:
    """
    Represents a single gate with its attributes.

    Attributes:
        gate_type: The type of gate (OPEN, REST, etc.)
        chinese: Chinese name
        pinyin: Romanized pronunciation
        native_palace: Natural palace position (1-9, excluding 5)
        element: Five Element association
        nature: Auspicious/Inauspicious classification
        meaning: Brief meaning of the gate
        classical_use: Classical strategic use
    """
    gate_type: GateType
    chinese: str
    pinyin: str
    native_palace: int
    element: str
    nature: GateNature
    meaning: str
    classical_use: str

    @property
    def is_auspicious(self) -> bool:
        """Check if this is one of the three auspicious gates."""
        return self.nature == GateNature.AUSPICIOUS

    @property
    def is_inauspicious(self) -> bool:
        return self.nature == GateNature.INAUSPICIOUS


# =============================================================================
# Gate Definitions
# =============================================================================

GATE_DEFINITIONS: Dict[GateType, TaiyiGate] = {
    GateType.OPEN: TaiyiGate(
        gate_type=GateType.OPEN,
        chinese="開門",
        pinyin="kāimén",
        native_palace=1,
        element="水",
        nature=GateNature.AUSPICIOUS,
        meaning="Opening, beginning, opportunity",
        classical_use="Favorable for all undertakings, starting ventures"
    ),
    GateType.REST: TaiyiGate(
        gate_type=GateType.REST,
        chinese="休門",
        pinyin="xiūmén",
        native_palace=8,
        element="土",
        nature=GateNature.AUSPICIOUS,
        meaning="Rest, recuperation, peace",
        classical_use="Favorable for rest, negotiation, seeking audiences"
    ),
    GateType.LIFE: TaiyiGate(
        gate_type=GateType.LIFE,
        chinese="生門",
        pinyin="shēngmén",
        native_palace=3,
        element="木",
        nature=GateNature.AUSPICIOUS,
        meaning="Life, growth, creation",
        classical_use="Favorable for commerce, seeking wealth, creation"
    ),
    GateType.WOUND: TaiyiGate(
        gate_type=GateType.WOUND,
        chinese="傷門",
        pinyin="shāngmén",
        native_palace=4,
        element="木",
        nature=GateNature.INAUSPICIOUS,
        meaning="Injury, conflict, harm",
        classical_use="Favorable only for hunting, combat, competitions"
    ),
    GateType.BLOCK: TaiyiGate(
        gate_type=GateType.BLOCK,
        chinese="杜門",
        pinyin="dùmén",
        native_palace=9,
        element="火",
        nature=GateNature.INAUSPICIOUS,
        meaning="Obstruction, hiding, secrecy",
        classical_use="Favorable for hiding, concealment, retreat"
    ),
    GateType.VIEW: TaiyiGate(
        gate_type=GateType.VIEW,
        chinese="景門",
        pinyin="jǐngmén",
        native_palace=2,
        element="土",
        nature=GateNature.NEUTRAL,
        meaning="View, fame, visibility",
        classical_use="Favorable for examinations, fame-seeking, announcements"
    ),
    GateType.DEATH: TaiyiGate(
        gate_type=GateType.DEATH,
        chinese="死門",
        pinyin="sǐmén",
        native_palace=7,
        element="金",
        nature=GateNature.INAUSPICIOUS,
        meaning="Death, end, finality",
        classical_use="Favorable only for funerals, burials, endings"
    ),
    GateType.FEAR: TaiyiGate(
        gate_type=GateType.FEAR,
        chinese="驚門",
        pinyin="jīngmén",
        native_palace=6,
        element="金",
        nature=GateNature.INAUSPICIOUS,
        meaning="Fear, shock, alarm",
        classical_use="Favorable only for legal matters, accusations"
    ),
}

# Traversal sequence of gates (following palace order)
GATE_SEQUENCE: List[GateType] = [
    GateType.OPEN,   # 1 - 開
    GateType.REST,   # 8 - 休
    GateType.LIFE,   # 3 - 生
    GateType.WOUND,  # 4 - 傷
    GateType.BLOCK,  # 9 - 杜
    GateType.VIEW,   # 2 - 景
    GateType.DEATH,  # 7 - 死
    GateType.FEAR,   # 6 - 驚
]

# Palace to gate mapping (native positions)
PALACE_TO_GATE: Dict[int, GateType] = {
    1: GateType.OPEN,
    8: GateType.REST,
    3: GateType.LIFE,
    4: GateType.WOUND,
    9: GateType.BLOCK,
    2: GateType.VIEW,
    7: GateType.DEATH,
    6: GateType.FEAR,
}


# =============================================================================
# Eight Gates System (八門系統)
# =============================================================================

class EightGates:
    """
    Manages the Eight Gates system for Taiyi calculations.

    The gates rotate around the nine palaces, with the ruling gate
    (直門/直使) determined by the calculation type and accumulated time.

    From 太乙金鏡式經:
    "命起開門，以次休、生門，左行八門，周而復始"
    """

    def __init__(self):
        self.gates = GATE_DEFINITIONS

    # =========================================================================
    # Gate Information Methods
    # =========================================================================

    def get_gate(self, gate_type: GateType) -> TaiyiGate:
        """Get gate information by type."""
        return self.gates[gate_type]

    def get_gate_by_palace(self, palace: int) -> Optional[TaiyiGate]:
        """Get the native gate for a palace."""
        gate_type = PALACE_TO_GATE.get(palace)
        if gate_type:
            return self.gates[gate_type]
        return None

    def get_auspicious_gates(self) -> List[TaiyiGate]:
        """Get the three auspicious gates (三吉門)."""
        return [g for g in self.gates.values() if g.is_auspicious]

    def get_inauspicious_gates(self) -> List[TaiyiGate]:
        """Get the inauspicious gates."""
        return [g for g in self.gates.values() if g.is_inauspicious]

    # =========================================================================
    # Gate Calculation Methods
    # =========================================================================

    def calculate_ruling_gate(self, accumulated: int) -> Tuple[GateType, int]:
        """
        Calculate the ruling gate (直門/直使) for a given accumulated time.

        Implements the classical algorithm:
        "以大游纪法七百二十去之；不尽，以三分纪法二百四十除之；
         余以三十约之，为直门数"

        Args:
            accumulated: Accumulated years/months/days

        Returns:
            Tuple of (ruling_gate_type, years_in_gate)
        """
        # Step 1: Remove complete Da You cycles (720)
        remainder = accumulated % GateConstants.DA_YOU_JI_FA

        # Step 2: Remove complete San Fen cycles (240)
        remainder = remainder % GateConstants.SAN_FEN_JI_FA

        # Step 3: Divide by gate cycle (30) to get gate index
        gate_index = remainder // GateConstants.GATE_CYCLE
        years_in_gate = remainder % GateConstants.GATE_CYCLE

        # Step 4: Map to gate sequence (starting from Open Gate)
        gate_type = GATE_SEQUENCE[gate_index % GateConstants.NUM_GATES]

        return (gate_type, years_in_gate)

    def calculate_gates_for_position(self, ruling_gate: GateType,
                                       start_palace: int) -> Dict[int, TaiyiGate]:
        """
        Calculate gate positions by adding ruling gate to each palace.

        From 太乙金鏡式經:
        "以開門加大乙，即太乙之八門也"

        This rotates the gates so the ruling gate lands on the start palace.

        Args:
            ruling_gate: The current ruling gate
            start_palace: Palace to place the ruling gate (e.g., Taiyi's palace)

        Returns:
            Dictionary mapping palace numbers to gates
        """
        # Find offset needed
        ruling_index = GATE_SEQUENCE.index(ruling_gate)

        # Palace sequence (excluding center palace 5)
        palace_sequence = [1, 8, 3, 4, 9, 2, 7, 6]

        # Find where start_palace is in the sequence
        try:
            start_index = palace_sequence.index(start_palace)
        except ValueError:
            start_index = 0  # Default if palace not found

        # Calculate offset
        offset = start_index

        # Place gates
        gate_positions = {}
        for i, palace in enumerate(palace_sequence):
            gate_idx = (ruling_index + i - offset) % 8
            gate_type = GATE_SEQUENCE[gate_idx]
            gate_positions[palace] = self.gates[gate_type]

        return gate_positions

    def calculate_taiyi_gates(self, ruling_gate: GateType,
                               taiyi_palace: int) -> Dict[int, TaiyiGate]:
        """
        Calculate Taiyi's Eight Gates (太乙八門).

        From classical text: "以開門加大乙，即太乙之八門也"

        Args:
            ruling_gate: Current ruling gate
            taiyi_palace: Palace where Taiyi is located

        Returns:
            Dictionary mapping palaces to gates
        """
        return self.calculate_gates_for_position(ruling_gate, taiyi_palace)

    def calculate_host_general_gates(self, ruling_gate: GateType,
                                       host_general_palace: int) -> Dict[int, TaiyiGate]:
        """
        Calculate Host Grand General's Eight Gates (主大將八門).

        From classical text: "以開門加主大將，即主大將之八門也"
        """
        return self.calculate_gates_for_position(ruling_gate, host_general_palace)

    def calculate_guest_general_gates(self, ruling_gate: GateType,
                                        guest_general_palace: int) -> Dict[int, TaiyiGate]:
        """
        Calculate Guest Grand General's Eight Gates (客大將八門).

        From classical text: "以開門加客大將，即客大將之八門也"
        """
        return self.calculate_gates_for_position(ruling_gate, guest_general_palace)

    # =========================================================================
    # Gate State Analysis
    # =========================================================================

    def is_gates_blocked(self, host_general_palace: int,
                          guest_general_palace: int) -> bool:
        """
        Check if gates are blocked (八門杜).

        Gates are blocked when generals cannot exit the center palace.

        Args:
            host_general_palace: Host general's palace
            guest_general_palace: Guest general's palace

        Returns:
            True if gates are blocked
        """
        # Gates blocked if either general is in center or specific positions
        return host_general_palace == 5 or guest_general_palace == 5

    def check_gate_harmony(self, taiyi_gate: TaiyiGate,
                            host_gate: TaiyiGate,
                            guest_gate: TaiyiGate) -> Dict[str, any]:
        """
        Check harmony between Taiyi's gate and the generals' gates.

        From 太乙金鏡式經:
        "客、主八門與太乙八門開、休、生合者，大利"

        Args:
            taiyi_gate: Gate at Taiyi's position
            host_gate: Gate at host general's position
            guest_gate: Gate at guest general's position

        Returns:
            Dictionary with harmony analysis
        """
        auspicious_gates = {GateType.OPEN, GateType.REST, GateType.LIFE}

        taiyi_auspicious = taiyi_gate.gate_type in auspicious_gates
        host_auspicious = host_gate.gate_type in auspicious_gates
        guest_auspicious = guest_gate.gate_type in auspicious_gates

        # Check for harmony (合)
        host_harmony = taiyi_auspicious and host_auspicious
        guest_harmony = taiyi_auspicious and guest_auspicious

        return {
            "taiyi_gate": taiyi_gate.chinese,
            "host_gate": host_gate.chinese,
            "guest_gate": guest_gate.chinese,
            "taiyi_auspicious": taiyi_auspicious,
            "host_auspicious": host_auspicious,
            "guest_auspicious": guest_auspicious,
            "host_harmony": host_harmony,
            "guest_harmony": guest_harmony,
            "three_gate_harmony": taiyi_auspicious and host_auspicious and guest_auspicious,
            "interpretation": self._interpret_gate_harmony(
                host_harmony, guest_harmony, taiyi_auspicious
            )
        }

    def _interpret_gate_harmony(self, host_harmony: bool,
                                 guest_harmony: bool,
                                 taiyi_auspicious: bool) -> str:
        """Generate interpretation of gate harmony."""
        if host_harmony and guest_harmony:
            return "三門皆合 - All three gates harmonious, greatly auspicious"
        elif host_harmony and not guest_harmony:
            return "主門合 - Host gate harmonious, favors defender"
        elif guest_harmony and not host_harmony:
            return "客門合 - Guest gate harmonious, favors attacker"
        elif taiyi_auspicious:
            return "太乙吉門 - Taiyi in auspicious gate, moderate fortune"
        else:
            return "諸門不合 - No gate harmony, proceed with caution"

    # =========================================================================
    # Complete Gate Calculation
    # =========================================================================

    def calculate_all_gates(self,
                            accumulated: int,
                            taiyi_palace: int,
                            host_general_palace: int,
                            guest_general_palace: int) -> Dict[str, any]:
        """
        Calculate complete gate positions and analysis.

        Args:
            accumulated: Accumulated time units
            taiyi_palace: Palace where Taiyi is located
            host_general_palace: Host general's palace
            guest_general_palace: Guest general's palace

        Returns:
            Dictionary with complete gate data
        """
        # Calculate ruling gate
        ruling_gate, years_in_gate = self.calculate_ruling_gate(accumulated)
        ruling_gate_info = self.gates[ruling_gate]

        # Calculate gate positions
        taiyi_gates = self.calculate_taiyi_gates(ruling_gate, taiyi_palace)
        host_gates = self.calculate_host_general_gates(ruling_gate, host_general_palace)
        guest_gates = self.calculate_guest_general_gates(ruling_gate, guest_general_palace)

        # Get gates at key positions
        taiyi_position_gate = taiyi_gates.get(taiyi_palace, ruling_gate_info)
        host_position_gate = host_gates.get(host_general_palace, ruling_gate_info)
        guest_position_gate = guest_gates.get(guest_general_palace, ruling_gate_info)

        # Check harmony
        harmony = self.check_gate_harmony(
            taiyi_position_gate, host_position_gate, guest_position_gate
        )

        # Check if blocked
        is_blocked = self.is_gates_blocked(host_general_palace, guest_general_palace)

        return {
            "ruling_gate": ruling_gate_info.chinese,
            "ruling_gate_type": ruling_gate,
            "years_in_gate": years_in_gate,
            "taiyi_gate": taiyi_position_gate.chinese,
            "host_gate": host_position_gate.chinese,
            "guest_gate": guest_position_gate.chinese,
            "is_blocked": is_blocked,
            "harmony": harmony,
            "taiyi_gates": {p: g.chinese for p, g in taiyi_gates.items()},
            "auspicious_count": sum(1 for g in [taiyi_position_gate, host_position_gate, guest_position_gate] if g.is_auspicious),
        }


# =============================================================================
# Convenience Function
# =============================================================================

def calculate_gates(accumulated: int,
                    taiyi_palace: int,
                    host_general_palace: int,
                    guest_general_palace: int) -> Dict[str, any]:
    """
    Convenience function to calculate gate positions.

    Args:
        accumulated: Accumulated time units
        taiyi_palace: Taiyi's palace
        host_general_palace: Host general's palace
        guest_general_palace: Guest general's palace

    Returns:
        Complete gate calculation results
    """
    system = EightGates()
    return system.calculate_all_gates(
        accumulated, taiyi_palace, host_general_palace, guest_general_palace
    )


# End of gates module
