"""
Finite automaton for stable road-surface state tracking.

Formal definition (Moore machine with debouncing)
-------------------------------------------------
Input alphabet  Sigma : {dry, wet, snow, gravel, damaged, unk}
                        (CNN prediction per frame; 'unk' = confidence too low)
States          Q     : (current, candidate, count)
                        current   in {INIT} U classes
                        candidate in {None} U classes
                        count     in {0 .. K-1}
Start state     q0    : (INIT, None, 0)
Transition      delta :
    - symbol == 'unk'                 -> state unchanged
    - symbol == current               -> (current, None, 0)           (noise cleared)
    - symbol != current:
         if symbol == candidate:  count += 1
         else:                    candidate = symbol, count = 1
         if count == K:           -> (symbol, None, 0)                (confirmed change)
Output          lambda: warning message attached to the *current* state (Moore output).

Because count is bounded by K, the number of states is finite, so this is a
legitimate finite automaton. K = confirm_frames removes frame-to-frame flicker.
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple

UNKNOWN = "unk"
INIT = "INIT"

WARNINGS = {
    INIT: "Initialising... waiting for stable road reading.",
    "dry": "Road dry. Normal driving.",
    "wet": "Wet road. Reduce speed and increase braking distance.",
    "snow": "Snow/ice detected. Drive slowly, avoid sudden braking.",
    "gravel": "Loose gravel. Reduce speed and avoid sharp turns.",
    "damaged": "Damaged road (pothole/cracks). Slow down and watch for hazards.",
}

# Recommended speed limit (km/h) per state -- illustrative values for the demo.
SPEED_LIMIT = {INIT: None, "dry": 80, "wet": 60, "snow": 30, "gravel": 40, "damaged": 30}


@dataclass
class FSMState:
    current: str = INIT
    candidate: Optional[str] = None
    count: int = 0


class RoadStateFSM:
    def __init__(self, classes: List[str], confirm_frames: int = 5):
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        self.classes = list(classes)
        self.alphabet = self.classes + [UNKNOWN]
        self.k = confirm_frames
        self.state = FSMState()
        self.history: List[Tuple[str, str]] = []   # (input_symbol, state_after)

    def reset(self):
        self.state = FSMState()
        self.history.clear()

    def step(self, symbol: str) -> str:
        """Feed one input symbol, return the current (stable) road state."""
        if symbol not in self.alphabet:
            raise ValueError(f"Symbol '{symbol}' not in alphabet {self.alphabet}")
        s = self.state
        if symbol != UNKNOWN:
            if symbol == s.current:
                s.candidate, s.count = None, 0
            else:
                if symbol == s.candidate:
                    s.count += 1
                else:
                    s.candidate, s.count = symbol, 1
                if s.count >= self.k:
                    s.current, s.candidate, s.count = symbol, None, 0
        self.history.append((symbol, s.current))
        return s.current

    @property
    def warning(self) -> str:
        return WARNINGS.get(self.state.current, "")

    @property
    def speed_limit(self):
        return SPEED_LIMIT.get(self.state.current)

    def run(self, symbols: List[str]) -> List[str]:
        return [self.step(x) for x in symbols]

    def transition_table(self) -> str:
        """Human-readable summary of the automaton (for the report)."""
        lines = [f"K = {self.k} consecutive frames required to change state",
                 f"Alphabet: {self.alphabet}",
                 f"States  : {[INIT] + self.classes} (+ internal candidate counter 0..{self.k-1})",
                 "Rules   :"]
        lines += ["  unk            -> stay in current state",
                  "  same as state  -> stay, reset candidate counter",
                  f"  different      -> count candidate; after {self.k} in a row, move to it"]
        return "\n".join(lines)


def mermaid_diagram(classes: List[str]) -> str:
    """Mermaid state diagram of the stable-state level of the automaton."""
    out = ["stateDiagram-v2", "    [*] --> INIT"]
    for c in classes:
        out.append(f"    INIT --> {c}: K x {c}")
    for a in classes:
        for b in classes:
            if a != b:
                out.append(f"    {a} --> {b}: K x {b}")
    return "\n".join(out)
