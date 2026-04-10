from __future__ import annotations
import json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hospital_env.env import HospitalSchedulingEnv
from hospital_env.models import Action, Priority
from hospital_env.tasks.scenerio import task_easy, task_medium, task_hard
from hospital_env.tasks.graders import grade

# ── Config ────────────────────────────────────────────────────────────────

MODEL_NAME = "greedy-agent"

TASK_MAP = {
    "easy": task_easy,
    "medium": task_medium,
    "hard": task_hard
}

# ── Logging helpers ───────────────────────────────────────────────────────

def log_start(task: str, total_patients: int, model: str):
    print(json.dumps({
        "event": "[START]",
        "task": task,
        "total_patients": total_patients,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), flush=True)


def log_step(step_num: int, action: dict, reward: float, done: bool, info: dict):
    print(json.dumps({
        "event": "[STEP]",
        "step": step_num,
        "action": action,
        "reward": round(reward, 4),
        "done": done,
        "info": info,
    }), flush=True)


def log_end(task: str, score: float, breakdown: dict, total_steps: int, elapsed: float):
    print(json.dumps({
        "event": "[END]",
        "task": task,
        "score": round(score, 4),
        "breakdown": breakdown,
        "total_steps": total_steps,
        "elapsed_sec": round(elapsed, 2),
    }), flush=True)


# ── Greedy Agent ──────────────────────────────────────────────────────────

def _greedy_fallback(obs) -> dict:
    priority_order = [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]
    unscheduled = [p for p in obs.patients if p.assigned_doctor_id is None]
    unscheduled.sort(key=lambda p: priority_order.index(p.priority))

    for patient in unscheduled:
        matching_doctors = [d for d in obs.doctors if d.department == patient.required_department]
        if not matching_doctors:
            matching_doctors = obs.doctors

        matching_rooms = [r for r in obs.rooms
                          if r.department == patient.required_department
                          and r.room_type == patient.required_room_type]
        if not matching_rooms:
            matching_rooms = obs.rooms

        for doctor in matching_doctors:
            for room in matching_rooms:
                both_free = sorted(set(doctor.available_slots) & set(room.available_slots))
                dur = patient.appointment_duration

                for i in range(len(both_free) - dur + 1):
                    block = both_free[i:i + dur]
                    if block[-1] - block[0] == dur - 1 and block[0] >= patient.earliest_slot:
                        return {
                            "patient_id": patient.id,
                            "doctor_id": doctor.id,
                            "room_id": room.id,
                            "start_slot": block[0],
                        }

    # fallback fallback
    p = unscheduled[0] if unscheduled else obs.patients[0]
    return {
        "patient_id": p.id,
        "doctor_id": obs.doctors[0].id,
        "room_id": obs.rooms[0].id,
        "start_slot": p.earliest_slot,
    }


# ── Run Task ──────────────────────────────────────────────────────────────

def run_task(task_name: str) -> float:
    patients, doctors, rooms = TASK_MAP[task_name]()
    env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)
    obs = env.reset()

    log_start(task_name, obs.total_patients, MODEL_NAME)
    t0 = time.time()

    while not obs.done:
        action_dict = _greedy_fallback(obs)

        try:
            action = Action(
                patient_id=str(action_dict.get("patient_id", "")),
                doctor_id=str(action_dict.get("doctor_id", "")),
                room_id=str(action_dict.get("room_id", "")),
                start_slot=int(action_dict.get("start_slot", 0)),
            )
        except:
            action = Action("NONE", "NONE", "NONE", 0)

        result = env.step(action)
        obs = result.observation

        log_step(
            step_num=obs.current_step,
            action=action_dict,
            reward=result.reward,
            done=result.done,
            info=result.info,
        )

    elapsed = time.time() - t0
    score, breakdown = grade(task_name, obs)

    log_end(task_name, score, breakdown, obs.current_step, elapsed)
    return score


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    scores = {}

    for task in ["easy", "medium", "hard"]:
        try:
            scores[task] = run_task(task)
        except Exception as e:
            print(json.dumps({
                "event": "[END]",
                "task": task,
                "score": 0.0,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }), flush=True)
            scores[task] = 0.0

    avg = sum(scores.values()) / len(scores)

    print(json.dumps({
        "event": "SUMMARY",
        "scores": scores,
        "average": round(avg, 4),
        "model": MODEL_NAME,
    }), flush=True)


if __name__ == "__main__":
    main()