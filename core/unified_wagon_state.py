"""UnifiedWagonState -- one physical wagon, fully fused across cameras.

This is the canonical record consumed by reporting/.  It carries:
    - identity         (global_id, classification, OCR)
    - per-side door state
    - load status
    - damage status
    - provenance (which cameras contributed) + an overall confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

from . import constants as C


@dataclass
class UnifiedWagonState:
    global_id: str
    wagon_index: int

    # Stage-0 authoritative
    classification: str = C.CLASS_UNKNOWN
    classification_confidence: float = 0.0

    # Identity
    wagon_identifier: str = C.NO_DATA
    wagon_identifier_confidence: float = 0.0

    # Doors (side cameras)
    left_door: str = C.NO_DATA
    left_door_confidence: float = 0.0
    right_door: str = C.NO_DATA
    right_door_confidence: float = 0.0
    # Additive: every DISTINCT door of this wagon, each with its own state and
    # snapshot reference. A wagon side can show two doors in different states
    # (door 1 CLOSED, door 2 OPEN), which the two per-side fields above cannot
    # represent. Empty for a door payload that predates the field, in which
    # case consumers fall back to the per-side values exactly as before.
    doors: List[Dict[str, Any]] = field(default_factory=list)

    # Load (top cameras)
    load_status: str = C.NO_DATA
    load_confidence: float = 0.0

    # Damage
    top_damage: str = C.NO_DATA
    top_damage_details: List[Dict[str, Any]] = field(default_factory=list)
    side_damage: str = C.NO_DATA
    side_damage_details: List[Dict[str, Any]] = field(default_factory=list)

    # Provenance
    supporting_cameras: List[str] = field(default_factory=list)
    missing_cameras: List[str] = field(default_factory=list)
    confidence: float = 0.0          # 0..1 combined
    anomalies: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ----------------------------------------------------------------
    # convenience predicates
    # ----------------------------------------------------------------

    @property
    def has_open_door(self) -> bool:
        if any(d.get("state") == C.DOOR_OPEN for d in self.doors):
            return True
        return self.left_door == C.DOOR_OPEN or self.right_door == C.DOOR_OPEN

    @property
    def door_status(self) -> str:
        """Wagon-level Door status: OPEN when ANY of its doors is open.

        Prefers the per-door list; falls back to the two per-side fields, whose
        picker already biases to OPEN within a side.
        """
        states = [str(d.get("state") or "") for d in self.doors]
        if not states:
            states = [self.left_door, self.right_door]
        for wanted in (C.DOOR_DAMAGED, C.DOOR_OPEN, C.DOOR_PARTIAL,
                       C.DOOR_CLOSED):
            if wanted in states:
                return wanted
        return C.NO_DATA

    @property
    def has_damage(self) -> bool:
        return (self.top_damage == C.DAMAGE_PRESENT
                or self.side_damage == C.DAMAGE_PRESENT)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_wagons(wagons: List[UnifiedWagonState]) -> Dict[str, Any]:
    """Train-level summary: counts of every flagged condition."""
    return {
        "total_wagons":   len(wagons),
        "engine_count":   sum(1 for w in wagons if w.classification == C.CLASS_ENGINE),
        "wagon_count":    sum(1 for w in wagons if w.classification == C.CLASS_WAGON),
        "brake_van_count":sum(1 for w in wagons if w.classification == C.CLASS_BRAKE_VAN),
        "left_doors_open":  sum(1 for w in wagons if w.left_door == C.DOOR_OPEN),
        "right_doors_open": sum(1 for w in wagons if w.right_door == C.DOOR_OPEN),
        "loaded":           sum(1 for w in wagons if w.load_status == C.LOAD_LOADED),
        "empty":            sum(1 for w in wagons if w.load_status == C.LOAD_EMPTY),
        "top_damaged":      sum(1 for w in wagons if w.top_damage == C.DAMAGE_PRESENT),
        "side_damaged":     sum(1 for w in wagons if w.side_damage == C.DAMAGE_PRESENT),
        "ocr_captured":     sum(1 for w in wagons if w.wagon_identifier != C.NO_DATA),
    }
