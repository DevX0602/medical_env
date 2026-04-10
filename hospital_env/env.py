"""
HospitalSchedulingEnv — OpenEnv-compatible environment.
step() / reset() / state() API

What makes this environment novel:
  1. Staff fatigue       — doctors slow down after consecutive appointments; agent
                           must spread load or face increasing time-slot costs.
  2. Dynamic arrivals    — surprise walk-in patients appear mid-episode, some CRITICAL.
  3. Equipment failures  — rooms go offline unexpectedly; agent must reschedule.
  4. Priority escalation — stable patients can deteriorate and jump priority mid-episode.
  5. Mass casualty surge — rare event floods the system with critical patients at once.
"""
from __future__ import annotations
import copy, random
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action, Department, Doctor, DynamicEvent, EventType,
    Observation, Patient, Priority, Room, RoomType,
    ScheduleEntry, StepResult,
)


PRIORITY_WEIGHTS = {
    Priority.CRITICAL: 1.0,
    Priority.HIGH:     0.75,
    Priority.MEDIUM:   0.5,
    Priority.LOW:      0.25,
}

DEADLINE_PENALTY    = -0.3
CONFLICT_PENALTY    = -0.5
FATIGUE_THRESHOLD   = 0.7      # above this, doctor needs a break
FATIGUE_PENALTY     = -0.15    # per step when using an exhausted doctor
FATIGUE_PER_PATIENT = 0.18     # fatigue gained per appointment
FATIGUE_DECAY       = 0.05     # fatigue lost per idle slot (rest)


class HospitalSchedulingEnv:
    """
    OpenEnv hospital scheduling environment with dynamic disruptions.

    Novel mechanics on top of the base scheduling problem:
      - Doctor fatigue accumulates; scheduling an exhausted doctor incurs reward penalty
      - Walk-in patients can appear at any step (config: enable_dynamic_events=True)
      - Rooms can fail mid-episode forcing rescheduling or creative solutions
      - Patient priority can escalate mid-episode (stable → critical)
      - Hard and easy modes both available via constructor flags
    """

    metadata = {"version": "2.0.0", "name": "HospitalSchedulingEnv"}

    def __init__(
        self,
        patients: List[Patient],
        doctors: List[Doctor],
        rooms: List[Room],
        total_slots: int = 16,
        max_steps: Optional[int] = None,
        enable_dynamic_events: bool = False,
        dynamic_events: Optional[List[DynamicEvent]] = None,
        seed: int = 42,
    ):
        self._init_patients        = copy.deepcopy(patients)
        self._init_doctors         = copy.deepcopy(doctors)
        self._init_rooms           = copy.deepcopy(rooms)
        self.total_slots           = total_slots
        self.max_steps             = max_steps or (len(patients) * 3)
        self.enable_dynamic_events = enable_dynamic_events
        self._init_events          = copy.deepcopy(dynamic_events or [])
        self._seed                 = seed

        # mutable state
        self.patients: List[Patient]             = []
        self.doctors:  List[Doctor]              = []
        self.rooms:    List[Room]                = []
        self.schedule: List[ScheduleEntry]       = []
        self.pending_events: List[DynamicEvent]  = []
        self.fired_events:   List[DynamicEvent]  = []
        self.current_step    = 0
        self.reward_so_far   = 0.0
        self.disruption_count = 0
        self._done           = False
        self._rng            = random.Random(seed)

        self.reset()

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self.patients          = copy.deepcopy(self._init_patients)
        self.doctors           = copy.deepcopy(self._init_doctors)
        self.rooms             = copy.deepcopy(self._init_rooms)
        self.schedule          = []
        self.pending_events    = copy.deepcopy(self._init_events)
        self.fired_events      = []
        self.current_step      = 0
        self.reward_so_far     = 0.0
        self.disruption_count  = 0
        self._done             = False
        self._rng              = random.Random(self._seed)
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode finished. Call reset().")

        self.current_step += 1

        # 1. Apply any dynamic events that fire at this step
        just_fired = self._apply_events()

        # 2. Apply fatigue decay for resting doctors (not used this step yet)
        self._decay_fatigue()

        # 3. Execute the scheduling action
        reward, info = self._apply_action(action)
        if just_fired:
            info["events_fired"] = [e.event_type.value for e in just_fired]

        self.reward_so_far += reward

        # 4. Check episode end
        visible_patients = [p for p in self.patients if p.arrived_at_step <= self.current_step]
        all_scheduled    = all(p.assigned_doctor_id is not None for p in visible_patients)
        out_of_budget    = self.current_step >= self.max_steps
        self._done       = all_scheduled or out_of_budget

        if self._done:
            bonus, bonus_info = self._compute_final_bonus()
            reward            += bonus
            self.reward_so_far += bonus
            info.update(bonus_info)

        obs = self._make_observation(just_fired)
        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    def state(self) -> Observation:
        return self._make_observation()

    # ── Dynamic event engine ────────────────────────────────────────────────

    def _apply_events(self) -> List[DynamicEvent]:
        """Fire any events scheduled for this step. Returns list of fired events."""
        if not self.enable_dynamic_events:
            return []

        fired = []
        for event in self.pending_events:
            if event.was_applied or event.occurs_at_step != self.current_step:
                continue
            self._fire_event(event)
            event.was_applied = True
            self.fired_events.append(event)
            self.disruption_count += 1
            fired.append(event)

        self.pending_events = [e for e in self.pending_events if not e.was_applied]
        return fired

    def _fire_event(self, event: DynamicEvent):
        if event.event_type == EventType.PATIENT_ARRIVAL:
            # Surprise walk-in patient joins the queue
            p = event.payload.get("patient")
            if p:
                p.arrived_at_step = self.current_step
                p.is_walk_in      = True
                self.patients.append(copy.deepcopy(p))

        elif event.event_type == EventType.DOCTOR_SICK:
            # Doctor becomes unavailable mid-shift
            doc = self._get_doctor(event.affected_id)
            if doc:
                doc.is_available   = False
                doc.available_slots = []

        elif event.event_type == EventType.ROOM_EQUIPMENT_FAIL:
            # Room goes offline — any future bookings in it are blocked
            room = self._get_room(event.affected_id)
            if room:
                room.is_operational = False
                room.available_slots = []

        elif event.event_type == EventType.PATIENT_DETERIORATED:
            # Patient's condition worsens — priority escalates
            patient = self._get_patient(event.affected_id)
            if patient and patient.assigned_doctor_id is None:
                patient.priority     = Priority.CRITICAL
                patient.deadline_slot = event.payload.get("new_deadline")

        elif event.event_type == EventType.SURGE:
            # Mass casualty event — multiple critical patients arrive at once
            for p in event.payload.get("patients", []):
                p.arrived_at_step = self.current_step
                p.is_walk_in      = True
                self.patients.append(copy.deepcopy(p))

    # ── Fatigue mechanics ───────────────────────────────────────────────────

    def _decay_fatigue(self):
        """Doctors who haven't been used recently recover some fatigue."""
        for doc in self.doctors:
            if doc.fatigue_level > 0:
                doc.fatigue_level = max(0.0, doc.fatigue_level - FATIGUE_DECAY)

    def _apply_fatigue(self, doctor: Doctor):
        """Increase doctor fatigue after assigning an appointment."""
        doctor.fatigue_level       = min(1.0, doctor.fatigue_level + FATIGUE_PER_PATIENT)
        doctor.patients_seen_today += 1

    # ── Action application ──────────────────────────────────────────────────

    def _apply_action(self, action: Action) -> Tuple[float, Dict]:
        info: Dict[str, Any] = {}

        patient = self._get_patient(action.patient_id)
        doctor  = self._get_doctor(action.doctor_id)
        room    = self._get_room(action.room_id)

        if patient is None:
            return CONFLICT_PENALTY, {"error": f"Unknown patient {action.patient_id}"}
        if doctor is None:
            return CONFLICT_PENALTY, {"error": f"Unknown doctor {action.doctor_id}"}
        if room is None:
            return CONFLICT_PENALTY, {"error": f"Unknown room {action.room_id}"}
        if patient.assigned_doctor_id is not None:
            return CONFLICT_PENALTY, {"error": f"Patient {action.patient_id} already scheduled"}
        if not doctor.is_available:
            return CONFLICT_PENALTY, {"error": f"Doctor {action.doctor_id} is unavailable (sick/offline)"}
        if not room.is_operational:
            return CONFLICT_PENALTY, {"error": f"Room {action.room_id} is out of service (equipment failure)"}
        if patient.arrived_at_step > self.current_step:
            return CONFLICT_PENALTY, {"error": f"Patient {action.patient_id} not yet arrived"}

        end_slot = action.start_slot + patient.appointment_duration
        if action.start_slot < 0 or end_slot > self.total_slots:
            return CONFLICT_PENALTY, {"error": "Slot out of range"}
        if action.start_slot < patient.earliest_slot:
            return CONFLICT_PENALTY, {"error": "Before patient earliest slot"}

        needed = set(range(action.start_slot, end_slot))
        if not needed.issubset(set(doctor.available_slots)):
            return CONFLICT_PENALTY * 0.5, {"error": "Doctor not available for those slots"}
        if not needed.issubset(set(room.available_slots)):
            return CONFLICT_PENALTY * 0.5, {"error": "Room not available for those slots"}

        dept_ok = doctor.department == patient.required_department
        room_ok = (room.department == patient.required_department and
                   room.room_type  == patient.required_room_type)

        # Apply assignment
        patient.assigned_doctor_id  = doctor.id
        patient.assigned_room_id    = room.id
        patient.assigned_start_slot = action.start_slot
        for sl in needed:
            doctor.available_slots.remove(sl)
            room.available_slots.remove(sl)

        self.schedule.append(ScheduleEntry(
            patient_id=patient.id, doctor_id=doctor.id, room_id=room.id,
            start_slot=action.start_slot, end_slot=end_slot,
        ))

        # Apply fatigue to doctor
        self._apply_fatigue(doctor)

        # Compute step reward
        base  = PRIORITY_WEIGHTS[patient.priority]
        bonus = 0.0

        if dept_ok:
            bonus += 0.2
        else:
            bonus -= 0.2
            info["warning"] = "Department mismatch"

        if room_ok:
            bonus += 0.1
        else:
            bonus -= 0.1

        if patient.deadline_slot is not None:
            if end_slot <= patient.deadline_slot:
                bonus += 0.15
            else:
                bonus += DEADLINE_PENALTY
                info["deadline_missed"] = True

        # Fatigue penalty — agent should avoid overloading one doctor
        if doctor.fatigue_level >= FATIGUE_THRESHOLD:
            bonus += FATIGUE_PENALTY
            info["fatigue_penalty"] = True
            info["doctor_fatigue"]  = round(doctor.fatigue_level, 2)

        # Bonus for scheduling walk-in patients (harder to fit)
        if patient.is_walk_in:
            bonus += 0.1
            info["walk_in_bonus"] = True

        step_reward = round(base + bonus, 4)
        info["scheduled"]  = patient.id
        info["priority"]   = patient.priority.value
        return step_reward, info

    def _compute_final_bonus(self) -> Tuple[float, Dict]:
        visible   = [p for p in self.patients if p.arrived_at_step <= self.current_step]
        scheduled = [p for p in visible if p.assigned_doctor_id is not None]

        fraction      = len(scheduled) / max(len(visible), 1)
        weight_sum    = sum(PRIORITY_WEIGHTS[p.priority] for p in visible)
        weight_done   = sum(PRIORITY_WEIGHTS[p.priority] for p in scheduled)
        weighted_frac = weight_done / max(weight_sum, 1e-9)

        # Fatigue balance bonus — agent that spread load among doctors gets a bonus
        fatigues = [d.fatigue_level for d in self.doctors if d.is_available]
        fatigue_spread_bonus = 0.0
        if len(fatigues) > 1:
            avg_f = sum(fatigues) / len(fatigues)
            variance = sum((f - avg_f) ** 2 for f in fatigues) / len(fatigues)
            # Low variance = balanced load = bonus
            fatigue_spread_bonus = 0.1 * max(0.0, 1.0 - variance * 5)

        # Disruption response bonus — handled more events = harder = higher bonus
        disruption_bonus = min(0.1 * self.disruption_count, 0.3)

        bonus = 0.45 * fraction + 0.45 * weighted_frac + fatigue_spread_bonus + disruption_bonus
        return round(bonus, 4), {
            "final_fraction":     round(fraction, 3),
            "weighted_fraction":  round(weighted_frac, 3),
            "fatigue_spread_bonus": round(fatigue_spread_bonus, 3),
            "disruption_bonus":   round(disruption_bonus, 3),
            "total_scheduled":    len(scheduled),
            "total_visible":      len(visible),
            "disruptions_handled": self.disruption_count,
        }

    def _make_observation(self, just_fired: Optional[List] = None) -> Observation:
        visible = [p for p in self.patients if p.arrived_at_step <= self.current_step]
        scheduled = sum(1 for p in visible if p.assigned_doctor_id is not None)
        fatigues = [d.fatigue_level for d in self.doctors if d.is_available]
        avg_fatigue = sum(fatigues) / max(len(fatigues), 1)

        return Observation(
            patients=copy.deepcopy(visible),
            doctors=copy.deepcopy(self.doctors),
            rooms=copy.deepcopy(self.rooms),
            schedule=copy.deepcopy(self.schedule),
            current_step=self.current_step,
            max_steps=self.max_steps,
            scheduled_count=scheduled,
            total_patients=len(visible),
            reward_so_far=round(self.reward_so_far, 4),
            done=self._done,
            active_events=copy.deepcopy(just_fired or []),
            avg_doctor_fatigue=round(avg_fatigue, 3),
            disruption_count=self.disruption_count,
        )

    def _get_patient(self, pid):
        return next((p for p in self.patients if p.id == pid), None)

    def _get_doctor(self, did):
        return next((d for d in self.doctors if d.id == did), None)

    def _get_room(self, rid):
        return next((r for r in self.rooms if r.id == rid), None)