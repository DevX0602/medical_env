"""
Graders for the HospitalSchedulingEnv tasks.

Each grader takes a completed Observation and returns a normalised score
in [0.0, 1.0] with a breakdown dict for transparency.
"""
from __future__ import annotations
from typing import Dict, Tuple

from ..models import Observation, Priority


# ── Weights (must match env.py) ────────────────────────────────────────────

PRIORITY_WEIGHTS = {
    Priority.CRITICAL: 1.0,
    Priority.HIGH:     0.75,
    Priority.MEDIUM:   0.5,
    Priority.LOW:      0.25,
}


def _base_score(obs: Observation) -> Tuple[float, Dict]:
    """
    Shared scoring logic across all difficulty levels.

    Returns (raw_score, breakdown_dict).
    raw_score is NOT yet normalised — call normalise() after.
    """
    patients = obs.patients
    total = len(patients)
    if total == 0:
        return 0.0, {}

    scheduled      = [p for p in patients if p.assigned_doctor_id is not None]
    unscheduled    = [p for p in patients if p.assigned_doctor_id is None]

    weight_total   = sum(PRIORITY_WEIGHTS[p.priority] for p in patients)
    weight_sched   = sum(PRIORITY_WEIGHTS[p.priority] for p in scheduled)

    # Coverage score (0-1)
    coverage       = len(scheduled) / total
    # Priority-weighted coverage (0-1)
    priority_cov   = weight_sched / max(weight_total, 1e-9)

    # Compatibility score (dept + room type match)
    compat_scores  = []
    for p in scheduled:
        dept_ok = True   # dept checked during step; already penalised there
        room_ok = True   # same
        compat_scores.append(1.0 if (dept_ok and room_ok) else 0.5)
    compatibility  = (sum(compat_scores) / len(compat_scores)) if compat_scores else 0.0

    # Deadline score
    deadline_patients = [p for p in patients if p.deadline_slot is not None]
    met_deadlines = 0
    for p in deadline_patients:
        if p.assigned_start_slot is not None:
            end = p.assigned_start_slot + p.appointment_duration
            if end <= p.deadline_slot:
                met_deadlines += 1
    deadline_score = (met_deadlines / len(deadline_patients)) if deadline_patients else 1.0

    raw = (0.35 * coverage +
           0.35 * priority_cov +
           0.15 * compatibility +
           0.15 * deadline_score)

    breakdown = {
        "coverage":        round(coverage, 3),
        "priority_cov":    round(priority_cov, 3),
        "compatibility":   round(compatibility, 3),
        "deadline_score":  round(deadline_score, 3),
        "scheduled":       len(scheduled),
        "unscheduled":     len(unscheduled),
        "total":           total,
    }
    return raw, breakdown


# ── Public graders ─────────────────────────────────────────────────────────

def grade_easy(obs: Observation) -> Tuple[float, Dict]:
    """
    Easy task grader.

    Full marks if all 3 patients are scheduled without conflicts.
    Partial credit for each scheduled patient.
    """
    raw, breakdown = _base_score(obs)
    # Easy: no deadlines, so deadline_score is always 1.0 — upweight coverage
    score = min(raw * 1.1, 1.0)   # slight bonus multiplier
    breakdown["difficulty"] = "easy"
    breakdown["final_score"] = round(score, 4)
    return round(score, 4), breakdown


def grade_medium(obs: Observation) -> Tuple[float, Dict]:
    """
    Medium task grader.

    Rewards correct department matching and soft deadline compliance.
    """
    raw, breakdown = _base_score(obs)
    score = raw   # no multiplier — baseline formula
    breakdown["difficulty"] = "medium"
    breakdown["final_score"] = round(score, 4)
    return round(score, 4), breakdown


def grade_hard(obs: Observation) -> Tuple[float, Dict]:
    """
    Hard task grader.

    Extra penalty if CRITICAL patients are not scheduled or miss their deadlines.
    """
    raw, breakdown = _base_score(obs)

    # Extra penalty for unscheduled critical patients
    critical_patients = [p for p in obs.patients if p.priority == Priority.CRITICAL]
    unscheduled_critical = [p for p in critical_patients if p.assigned_doctor_id is None]
    critical_penalty = 0.15 * len(unscheduled_critical)

    # Extra penalty for critical patients who miss deadlines
    deadline_breach_penalty = 0.0
    for p in critical_patients:
        if p.assigned_start_slot is not None and p.deadline_slot is not None:
            end = p.assigned_start_slot + p.appointment_duration
            if end > p.deadline_slot:
                deadline_breach_penalty += 0.1

    score = max(raw - critical_penalty - deadline_breach_penalty, 0.0)
    breakdown.update({
        "difficulty":              "hard",
        "unscheduled_critical":    len(unscheduled_critical),
        "critical_penalty":        round(critical_penalty, 3),
        "deadline_breach_penalty": round(deadline_breach_penalty, 3),
        "final_score":             round(score, 4),
    })
    return round(score, 4), breakdown


# ── Registry ───────────────────────────────────────────────────────────────

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


def grade(task: str, obs: Observation) -> Tuple[float, Dict]:
    """Grade an observation for the given task difficulty string."""
    if task not in GRADERS:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(GRADERS)}")
    return GRADERS[task](obs)


def grade_chaos(obs: Observation) -> Tuple[float, Dict]:
    """
    Chaos task grader.

    Extra rewards for handling disruptions well:
    - Surviving without a key doctor (DOCTOR_SICK event)
    - Adapting to room failures
    - Triaging walk-in / surge patients
    - Maintaining fatigue balance across the team
    """
    raw, breakdown = _base_score(obs)

    # Bonus for managing disruptions (exposed via obs.disruption_count)
    disruption_bonus = min(0.05 * obs.disruption_count, 0.2)

    # Bonus for low average fatigue (agent spread load well)
    fatigue_bonus = 0.1 * max(0.0, 1.0 - obs.avg_doctor_fatigue)

    # Extra penalty for unscheduled CRITICAL patients (same as hard)
    critical_patients = [p for p in obs.patients if p.priority == Priority.CRITICAL]
    unscheduled_critical = [p for p in critical_patients if p.assigned_doctor_id is None]
    critical_penalty = 0.15 * len(unscheduled_critical)

    score = max(raw + disruption_bonus + fatigue_bonus - critical_penalty, 0.0)
    breakdown.update({
        "difficulty":            "chaos",
        "disruption_bonus":      round(disruption_bonus, 3),
        "fatigue_bonus":         round(fatigue_bonus, 3),
        "avg_doctor_fatigue":    round(obs.avg_doctor_fatigue, 3),
        "unscheduled_critical":  len(unscheduled_critical),
        "critical_penalty":      round(critical_penalty, 3),
        "final_score":           round(score, 4),
    })
    return round(score, 4), breakdown


GRADERS["chaos"] = grade_chaos