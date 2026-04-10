"""
Task definitions for HospitalSchedulingEnv.

task_easy   — 3 patients, 1 department, no deadlines
task_medium — 6 patients, 3 departments, soft deadlines
task_hard   — 10 patients, all departments, hard deadlines + emergencies
"""
from __future__ import annotations
from typing import List

from ..models import Department, Doctor, Patient, Priority, Room, RoomType


def _all_slots(total: int = 16) -> List[int]:
    return list(range(total))


# ── EASY ──────────────────────────────────────────────────────────────────

def task_easy():
    """
    3 patients, 1 doctor, 2 rooms, general department.
    No deadlines. Agent just needs to avoid conflicts.
    Max score ≈ 1.0
    """
    patients = [
        Patient(id="P1", name="Alice",   priority=Priority.LOW,    required_department=Department.GENERAL,
                required_room_type=RoomType.CONSULTATION, appointment_duration=1, earliest_slot=0),
        Patient(id="P2", name="Bob",     priority=Priority.MEDIUM, required_department=Department.GENERAL,
                required_room_type=RoomType.CONSULTATION, appointment_duration=2, earliest_slot=0),
        Patient(id="P3", name="Charlie", priority=Priority.HIGH,   required_department=Department.GENERAL,
                required_room_type=RoomType.CONSULTATION, appointment_duration=1, earliest_slot=2),
    ]
    doctors = [
        Doctor(id="D1", name="Dr. Smith", department=Department.GENERAL,
               available_slots=_all_slots(), max_patients_per_day=8),
    ]
    rooms = [
        Room(id="R1", room_type=RoomType.CONSULTATION, department=Department.GENERAL,
             available_slots=_all_slots()),
        Room(id="R2", room_type=RoomType.CONSULTATION, department=Department.GENERAL,
             available_slots=_all_slots()),
    ]
    return patients, doctors, rooms


# ── MEDIUM ────────────────────────────────────────────────────────────────

def task_medium():
    """
    6 patients across 3 departments, 3 doctors, 4 rooms, soft deadlines.
    Agent must match departments and respect slot constraints.
    Max score ≈ 1.0
    """
    patients = [
        Patient(id="P1", name="Diana",  priority=Priority.HIGH,   required_department=Department.CARDIOLOGY,
                required_room_type=RoomType.CONSULTATION, appointment_duration=2,
                earliest_slot=0, deadline_slot=8),
        Patient(id="P2", name="Eve",    priority=Priority.MEDIUM, required_department=Department.GENERAL,
                required_room_type=RoomType.CONSULTATION, appointment_duration=1,
                earliest_slot=0, deadline_slot=12),
        Patient(id="P3", name="Frank",  priority=Priority.LOW,    required_department=Department.ORTHOPEDICS,
                required_room_type=RoomType.CONSULTATION, appointment_duration=2,
                earliest_slot=4, deadline_slot=None),
        Patient(id="P4", name="Grace",  priority=Priority.HIGH,   required_department=Department.CARDIOLOGY,
                required_room_type=RoomType.CONSULTATION, appointment_duration=1,
                earliest_slot=0, deadline_slot=6),
        Patient(id="P5", name="Henry",  priority=Priority.MEDIUM, required_department=Department.GENERAL,
                required_room_type=RoomType.WARD,         appointment_duration=4,
                earliest_slot=2, deadline_slot=None),
        Patient(id="P6", name="Iris",   priority=Priority.LOW,    required_department=Department.ORTHOPEDICS,
                required_room_type=RoomType.CONSULTATION, appointment_duration=1,
                earliest_slot=0, deadline_slot=None),
    ]
    doctors = [
        Doctor(id="D1", name="Dr. Patel",    department=Department.CARDIOLOGY,
               available_slots=_all_slots()),
        Doctor(id="D2", name="Dr. Johnson",  department=Department.GENERAL,
               available_slots=_all_slots()),
        Doctor(id="D3", name="Dr. Williams", department=Department.ORTHOPEDICS,
               available_slots=_all_slots()),
    ]
    rooms = [
        Room(id="R1", room_type=RoomType.CONSULTATION, department=Department.CARDIOLOGY,
             available_slots=_all_slots()),
        Room(id="R2", room_type=RoomType.CONSULTATION, department=Department.GENERAL,
             available_slots=_all_slots()),
        Room(id="R3", room_type=RoomType.WARD,         department=Department.GENERAL,
             available_slots=_all_slots()),
        Room(id="R4", room_type=RoomType.CONSULTATION, department=Department.ORTHOPEDICS,
             available_slots=_all_slots()),
    ]
    return patients, doctors, rooms


# ── HARD ──────────────────────────────────────────────────────────────────

def task_hard():
    """
    10 patients across all departments including EMERGENCY and PEDIATRICS.
    Hard deadlines, scarce resources, room conflicts.
    Agent must triage critically ill patients first.
    Max score ≈ 1.0
    """
    patients = [
        # Emergencies
        Patient(id="P01", name="Jack",    priority=Priority.CRITICAL, required_department=Department.EMERGENCY,
                required_room_type=RoomType.ICU,          appointment_duration=4,
                earliest_slot=0,  deadline_slot=4),
        Patient(id="P02", name="Kate",    priority=Priority.CRITICAL, required_department=Department.EMERGENCY,
                required_room_type=RoomType.ICU,          appointment_duration=2,
                earliest_slot=0,  deadline_slot=3),
        # Cardiology
        Patient(id="P03", name="Liam",    priority=Priority.HIGH,     required_department=Department.CARDIOLOGY,
                required_room_type=RoomType.CONSULTATION, appointment_duration=2,
                earliest_slot=2,  deadline_slot=8),
        Patient(id="P04", name="Mia",     priority=Priority.HIGH,     required_department=Department.CARDIOLOGY,
                required_room_type=RoomType.SURGERY,      appointment_duration=3,
                earliest_slot=4,  deadline_slot=10),
        # Orthopedics
        Patient(id="P05", name="Noah",    priority=Priority.MEDIUM,   required_department=Department.ORTHOPEDICS,
                required_room_type=RoomType.SURGERY,      appointment_duration=2,
                earliest_slot=2,  deadline_slot=12),
        Patient(id="P06", name="Olivia",  priority=Priority.MEDIUM,   required_department=Department.ORTHOPEDICS,
                required_room_type=RoomType.CONSULTATION, appointment_duration=1,
                earliest_slot=0,  deadline_slot=None),
        # Pediatrics
        Patient(id="P07", name="Peter",   priority=Priority.HIGH,     required_department=Department.PEDIATRICS,
                required_room_type=RoomType.CONSULTATION, appointment_duration=1,
                earliest_slot=0,  deadline_slot=6),
        Patient(id="P08", name="Quinn",   priority=Priority.LOW,      required_department=Department.PEDIATRICS,
                required_room_type=RoomType.CONSULTATION, appointment_duration=2,
                earliest_slot=4,  deadline_slot=None),
        # General
        Patient(id="P09", name="Rachel",  priority=Priority.LOW,      required_department=Department.GENERAL,
                required_room_type=RoomType.CONSULTATION, appointment_duration=1,
                earliest_slot=0,  deadline_slot=None),
        Patient(id="P10", name="Samuel",  priority=Priority.MEDIUM,   required_department=Department.GENERAL,
                required_room_type=RoomType.WARD,         appointment_duration=4,
                earliest_slot=6,  deadline_slot=None),
    ]
    doctors = [
        Doctor(id="D1", name="Dr. ER-Chen",     department=Department.EMERGENCY,   available_slots=_all_slots()),
        Doctor(id="D2", name="Dr. Cardio-Kim",  department=Department.CARDIOLOGY,  available_slots=_all_slots()),
        Doctor(id="D3", name="Dr. Ortho-Ray",   department=Department.ORTHOPEDICS, available_slots=_all_slots()),
        Doctor(id="D4", name="Dr. Peds-Lee",    department=Department.PEDIATRICS,  available_slots=_all_slots()),
        Doctor(id="D5", name="Dr. Gen-Brown",   department=Department.GENERAL,     available_slots=_all_slots()),
    ]
    rooms = [
        Room(id="R1",  room_type=RoomType.ICU,          department=Department.EMERGENCY,   available_slots=_all_slots()),
        Room(id="R2",  room_type=RoomType.CONSULTATION, department=Department.EMERGENCY,   available_slots=_all_slots()),
        Room(id="R3",  room_type=RoomType.CONSULTATION, department=Department.CARDIOLOGY,  available_slots=_all_slots()),
        Room(id="R4",  room_type=RoomType.SURGERY,      department=Department.CARDIOLOGY,  available_slots=_all_slots()),
        Room(id="R5",  room_type=RoomType.CONSULTATION, department=Department.ORTHOPEDICS, available_slots=_all_slots()),
        Room(id="R6",  room_type=RoomType.SURGERY,      department=Department.ORTHOPEDICS, available_slots=_all_slots()),
        Room(id="R7",  room_type=RoomType.CONSULTATION, department=Department.PEDIATRICS,  available_slots=_all_slots()),
        Room(id="R8",  room_type=RoomType.CONSULTATION, department=Department.GENERAL,     available_slots=_all_slots()),
        Room(id="R9",  room_type=RoomType.WARD,         department=Department.GENERAL,     available_slots=_all_slots()),
    ]
    return patients, doctors, rooms


# ── CHAOS (bonus 4th task) ─────────────────────────────────────────────────

def task_chaos():
    """
    The ultimate test — 8 base patients + 4 surprise walk-ins mid-episode.
    
    Dynamic events fire throughout:
      Step 3  → Dr. ER-Chen calls in sick (lose your best emergency doctor)
      Step 5  → ICU equipment fails (Room R1 goes offline)
      Step 7  → Stable patient Peter deteriorates to CRITICAL
      Step 9  → Mass casualty surge (3 critical patients arrive simultaneously)
      Step 12 → Surprise walk-in (rare blood type surgical patient)
    
    The agent must:
      - Triage CRITICAL patients before their tight deadlines
      - Redistribute after losing a doctor mid-shift
      - Handle the ICU room failure (reschedule or use fallback room)
      - Balance staff fatigue across 5 doctors
      - Respond to surge without dropping earlier patients
    
    A random or naive agent scores ~0.35. A good LLM agent scores ~0.65+.
    Max theoretical score: ~1.0 (requires perfect triage + load balancing)
    """
    from ..models import DynamicEvent, EventType

    patients = [
        # Pre-scheduled patients (arrive at step 0)
        Patient(id="P01", name="Jack",    priority=Priority.CRITICAL, required_department=Department.EMERGENCY,
                required_room_type=RoomType.ICU,          appointment_duration=4,
                earliest_slot=0, deadline_slot=5),
        Patient(id="P02", name="Kate",    priority=Priority.HIGH,     required_department=Department.CARDIOLOGY,
                required_room_type=RoomType.CONSULTATION, appointment_duration=2,
                earliest_slot=0, deadline_slot=8),
        Patient(id="P03", name="Liam",    priority=Priority.HIGH,     required_department=Department.ORTHOPEDICS,
                required_room_type=RoomType.SURGERY,      appointment_duration=3,
                earliest_slot=2, deadline_slot=10),
        Patient(id="P04", name="Mia",     priority=Priority.MEDIUM,   required_department=Department.GENERAL,
                required_room_type=RoomType.CONSULTATION, appointment_duration=1,
                earliest_slot=0, deadline_slot=None),
        Patient(id="P05", name="Noah",    priority=Priority.LOW,      required_department=Department.PEDIATRICS,
                required_room_type=RoomType.CONSULTATION, appointment_duration=2,
                earliest_slot=4, deadline_slot=None),
        Patient(id="P06", name="Olivia",  priority=Priority.MEDIUM,   required_department=Department.GENERAL,
                required_room_type=RoomType.WARD,         appointment_duration=4,
                earliest_slot=6, deadline_slot=None),
        Patient(id="P07", name="Peter",   priority=Priority.LOW,      required_department=Department.CARDIOLOGY,
                required_room_type=RoomType.CONSULTATION, appointment_duration=1,
                earliest_slot=0, deadline_slot=None),
        Patient(id="P08", name="Quinn",   priority=Priority.MEDIUM,   required_department=Department.ORTHOPEDICS,
                required_room_type=RoomType.CONSULTATION, appointment_duration=2,
                earliest_slot=0, deadline_slot=None),
    ]

    doctors = [
        Doctor(id="D1", name="Dr. ER-Chen",     department=Department.EMERGENCY,   available_slots=_all_slots()),
        Doctor(id="D2", name="Dr. Cardio-Kim",  department=Department.CARDIOLOGY,  available_slots=_all_slots()),
        Doctor(id="D3", name="Dr. Ortho-Ray",   department=Department.ORTHOPEDICS, available_slots=_all_slots()),
        Doctor(id="D4", name="Dr. Peds-Lee",    department=Department.PEDIATRICS,  available_slots=_all_slots()),
        Doctor(id="D5", name="Dr. Gen-Brown",   department=Department.GENERAL,     available_slots=_all_slots()),
    ]

    rooms = [
        Room(id="R1",  room_type=RoomType.ICU,          department=Department.EMERGENCY,   available_slots=_all_slots()),
        Room(id="R2",  room_type=RoomType.CONSULTATION, department=Department.EMERGENCY,   available_slots=_all_slots()),
        Room(id="R3",  room_type=RoomType.CONSULTATION, department=Department.CARDIOLOGY,  available_slots=_all_slots()),
        Room(id="R4",  room_type=RoomType.SURGERY,      department=Department.ORTHOPEDICS, available_slots=_all_slots()),
        Room(id="R5",  room_type=RoomType.CONSULTATION, department=Department.ORTHOPEDICS, available_slots=_all_slots()),
        Room(id="R6",  room_type=RoomType.CONSULTATION, department=Department.PEDIATRICS,  available_slots=_all_slots()),
        Room(id="R7",  room_type=RoomType.CONSULTATION, department=Department.GENERAL,     available_slots=_all_slots()),
        Room(id="R8",  room_type=RoomType.WARD,         department=Department.GENERAL,     available_slots=_all_slots()),
    ]

    # ── Dynamic events that fire mid-episode ────────────────────────────────
    walk_in_surgical = Patient(
        id="W01", name="Surge-Victor", priority=Priority.HIGH,
        required_department=Department.ORTHOPEDICS, required_room_type=RoomType.SURGERY,
        appointment_duration=3, earliest_slot=0, deadline_slot=14,
    )

    surge_patients = [
        Patient(id="S01", name="Surge-Ana",   priority=Priority.CRITICAL, required_department=Department.EMERGENCY,
                required_room_type=RoomType.ICU,          appointment_duration=3, earliest_slot=0, deadline_slot=13),
        Patient(id="S02", name="Surge-Ben",   priority=Priority.CRITICAL, required_department=Department.EMERGENCY,
                required_room_type=RoomType.CONSULTATION, appointment_duration=2, earliest_slot=0, deadline_slot=14),
        Patient(id="S03", name="Surge-Carol", priority=Priority.HIGH,     required_department=Department.CARDIOLOGY,
                required_room_type=RoomType.CONSULTATION, appointment_duration=2, earliest_slot=0, deadline_slot=15),
    ]

    events = [
        DynamicEvent(
            event_type=EventType.DOCTOR_SICK,
            occurs_at_step=3,
            description="Dr. ER-Chen called in sick — emergency department now short-staffed",
            affected_id="D1",
        ),
        DynamicEvent(
            event_type=EventType.ROOM_EQUIPMENT_FAIL,
            occurs_at_step=5,
            description="ICU ventilator malfunction — Room R1 taken offline",
            affected_id="R1",
        ),
        DynamicEvent(
            event_type=EventType.PATIENT_DETERIORATED,
            occurs_at_step=7,
            description="Peter (P07) had a cardiac event — priority escalated to CRITICAL",
            affected_id="P07",
            payload={"new_deadline": 11},
        ),
        DynamicEvent(
            event_type=EventType.SURGE,
            occurs_at_step=9,
            description="Mass casualty event — 3 critical patients incoming",
            payload={"patients": surge_patients},
        ),
        DynamicEvent(
            event_type=EventType.PATIENT_ARRIVAL,
            occurs_at_step=12,
            description="Walk-in surgical patient requires urgent orthopedic procedure",
            payload={"patient": walk_in_surgical},
        ),
    ]

    return patients, doctors, rooms, events