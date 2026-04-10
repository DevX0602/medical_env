"""
Typed models for the Hospital Scheduling OpenEnv environment.
Uses stdlib dataclasses — works with Python 3.9+ and zero dependencies.
Pydantic v2 BaseModel is the preferred production choice (see requirements.txt).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Priority(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class Department(str, Enum):
    GENERAL     = "general"
    CARDIOLOGY  = "cardiology"
    ORTHOPEDICS = "orthopedics"
    EMERGENCY   = "emergency"
    PEDIATRICS  = "pediatrics"


class RoomType(str, Enum):
    CONSULTATION = "consultation"
    SURGERY      = "surgery"
    ICU          = "icu"
    WARD         = "ward"


class EventType(str, Enum):
    """Mid-episode dynamic events — what makes this environment truly novel."""
    PATIENT_ARRIVAL       = "patient_arrival"       # Surprise patient walks in
    DOCTOR_SICK           = "doctor_sick"            # Doctor calls in sick
    ROOM_EQUIPMENT_FAIL   = "room_equipment_fail"    # Equipment breaks down
    PATIENT_DETERIORATED  = "patient_deteriorated"   # Patient priority escalates
    SURGE                 = "surge"                  # Mass casualty surge


@dataclass
class Patient:
    id: str
    name: str
    priority: Priority
    required_department: Department
    required_room_type: RoomType
    appointment_duration: int
    earliest_slot: int = 0
    deadline_slot: Optional[int] = None
    assigned_doctor_id: Optional[str] = None
    assigned_room_id: Optional[str] = None
    assigned_start_slot: Optional[int] = None
    # Novelty fields
    arrived_at_step: int = 0          # Which step this patient became visible
    is_walk_in: bool = False          # Walk-in (surprise) vs pre-scheduled


@dataclass
class Doctor:
    id: str
    name: str
    department: Department
    available_slots: List[int] = field(default_factory=list)
    max_patients_per_day: int = 8
    # Fatigue mechanics: doctors slow down and need rest after long shifts
    fatigue_level: float = 0.0        # 0.0=fresh, 1.0=exhausted
    patients_seen_today: int = 0
    is_available: bool = True         # False if called in sick mid-episode


@dataclass
class Room:
    id: str
    room_type: RoomType
    department: Department
    available_slots: List[int] = field(default_factory=list)
    is_operational: bool = True       # False if equipment failure


@dataclass
class DynamicEvent:
    """A mid-episode event the agent must respond to."""
    event_type: EventType
    occurs_at_step: int
    description: str
    affected_id: Optional[str] = None   # doctor_id / room_id / patient_id
    payload: Dict[str, Any] = field(default_factory=dict)
    was_applied: bool = False


@dataclass
class ScheduleEntry:
    patient_id: str
    doctor_id: str
    room_id: str
    start_slot: int
    end_slot: int


@dataclass
class Action:
    patient_id: str
    doctor_id: str
    room_id: str
    start_slot: int


@dataclass
class Observation:
    patients: List[Patient]
    doctors: List[Doctor]
    rooms: List[Room]
    schedule: List[ScheduleEntry]
    current_step: int
    max_steps: int
    scheduled_count: int
    total_patients: int
    reward_so_far: float
    done: bool
    # Novelty fields
    active_events: List[DynamicEvent] = field(default_factory=list)   # events that just fired
    avg_doctor_fatigue: float = 0.0
    disruption_count: int = 0          # total disruptions this episode
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)