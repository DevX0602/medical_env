# 🏥 HospitalSchedulingEnv

> **OpenEnv-compatible RL environment for real-world hospital patient scheduling.**
> An AI agent assigns patients to doctors and rooms across time slots, balancing
> priorities, hard deadlines, department constraints, and scarce resources.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green)](https://openenv.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)

---

## Why Hospital Scheduling?

Hospital scheduling is one of the most consequential real-world planning problems:
- Misscheduled CRITICAL patients can lead to preventable harm
- Departments, room types, and doctor specialisms must all align
- Resources are scarce and time-constrained
- Priorities must be respected under uncertainty

This makes it ideal for evaluating agents that must reason about **constraints,
priorities, and partial information** — exactly the challenges frontier LLMs face.

---

## Environment Overview

| Property | Value |
|---|---|
| Domain | Healthcare / Operations Research |
| Task type | Scheduling / Constraint Satisfaction |
| API | `step()` / `reset()` / `state()` over HTTP |
| Action space | Structured: (patient_id, doctor_id, room_id, start_slot) |
| Observation space | Structured: patients, doctors, rooms, schedule, step info |
| Reward range | `[−1.0, 2.0]` per step |
| Score range | `[0.0, 1.0]` (grader output) |
| Time slots | 16 slots per day (08:00–16:00, 30-min increments) |

---

## Action Space

Each step the agent submits **one scheduling action**:

```json
{
  "patient_id": "P01",
  "doctor_id":  "D1",
  "room_id":    "R1",
  "start_slot": 4
}
```

**Hard constraints the agent must respect:**
- Doctor must be from the patient's required department
- Room must match the patient's required type and department
- All required time slots must be free for both doctor and room
- `start_slot ≥ patient.earliest_slot`
- `start_slot + duration ≤ 16` (within the day)

---

## Observation Space

Every `reset()` / `step()` / `state()` call returns:

```json
{
  "patients": [
    {
      "id": "P01", "name": "Jack", "priority": "critical",
      "required_department": "emergency", "required_room_type": "icu",
      "appointment_duration": 4, "earliest_slot": 0, "deadline_slot": 4,
      "assigned_doctor_id": null, "assigned_room_id": null, "assigned_start_slot": null
    }
  ],
  "doctors":  [{ "id": "D1", "department": "emergency", "available_slots": [0,1,...] }],
  "rooms":    [{ "id": "R1", "room_type": "icu", "department": "emergency", "available_slots": [...] }],
  "schedule": [],
  "current_step": 0, "max_steps": 30,
  "scheduled_count": 0, "total_patients": 10,
  "reward_so_far": 0.0, "done": false
}
```

---

## Reward Function

### Per-step reward
```
reward = priority_weight              # CRITICAL=1.0, HIGH=0.75, MEDIUM=0.5, LOW=0.25
       + dept_match        (+0.20)    # doctor department matches patient
       + room_match        (+0.10)    # room type + dept matches patient
       + deadline_bonus    (+0.15)    # appointment fits within deadline_slot
       # Penalties applied instead if constraints violated:
       - dept_mismatch     (−0.20)
       - room_mismatch     (−0.10)
       - deadline_missed   (−0.30)
       - conflict          (−0.50)    # invalid action (double-book, unknown ID, etc.)
```

### Episode-end bonus
```
bonus = 0.5 × (scheduled / total) + 0.5 × (priority-weighted scheduled / total)
```

The reward is **dense** — every step produces a non-zero signal reflecting partial progress.

---

## Tasks

### 🟢 Easy
| Property | Value |
|---|---|
| Patients | 3 |
| Doctors | 1 (General) |
| Rooms | 2 (Consultation) |
| Departments | General only |
| Deadlines | None |
| Max steps | 9 |
| **Baseline score** | **1.0000** |
| Target score | ≥ 0.90 |

### 🟡 Medium
| Property | Value |
|---|---|
| Patients | 6 |
| Doctors | 3 (Cardiology, General, Orthopedics) |
| Rooms | 4 |
| Departments | 3 |
| Deadlines | Soft (2 patients) |
| Max steps | 18 |
| **Baseline score** | **1.0000** |
| Target score | ≥ 0.75 |

### 🔴 Hard
| Property | Value |
|---|---|
| Patients | 10 |
| Doctors | 5 (all departments) |
| Rooms | 9 (includes ICU, Surgery) |
| Departments | All 5 (incl. Emergency, Pediatrics) |
| Deadlines | Hard (2 CRITICAL with tight deadlines) |
| Max steps | 30 |
| **Baseline score** | **0.8750** |
| Target score | ≥ 0.60 |

### Baseline Scores Summary

| Agent | Easy | Medium | Hard | Average |
|---|---|---|---|---|
| Greedy (deterministic) | 1.0000 | 1.0000 | 0.8750 | **0.9583** |
| Random (seed=42, 200 eps avg) | 1.0000 | 0.9305 | 0.7200 | 0.8835 |

---

## HTTP API (OpenEnv spec)

The server exposes these endpoints on port **7860**:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Ping — returns `{"status": "ok"}` |
| `GET` | `/tasks` | List all tasks with metadata |
| `POST` | `/reset` | `{"task": "easy\|medium\|hard", "session_id": "..."}` |
| `POST` | `/step` | `{"patient_id", "doctor_id", "room_id", "start_slot", "session_id"}` |
| `GET` | `/state` | `?session_id=default` |
| `POST` | `/grade` | Grade current session |

---

## Quickstart

### 1. Install
```bash
git clone <your-repo>
cd hospital-env
pip install -r requirements.txt
```

### 2. Run the HTTP server
```bash
python server.py
# Server starts on http://localhost:7860
curl http://localhost:7860/health
```

### 3. Run the LLM inference script
```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

### 4. Run the greedy baseline
```bash
python baseline/run_baseline.py --task all --verbose
```

### 5. Run tests
```bash
python tests/test_env.py
# 26/26 tests pass
```

### 6. Use the web UI
```bash
pip install gradio
python app.py
# Open http://localhost:7860
```

---

## Project Structure

```
hospital-env/
├── inference.py             ← ⭐ MANDATORY: LLM agent, [START]/[STEP]/[END] logs
├── server.py                ← FastAPI HTTP server (/reset /step /state /health)
├── app.py                   ← Gradio interactive web UI
├── openenv.yaml             ← OpenEnv specification + HTTP endpoints
├── Dockerfile               ← HF Spaces ready (port 7860, /health check)
├── requirements.txt
├── setup.py
├── Makefile
├── README.md
├── hospital_env/
│   ├── __init__.py
│   ├── env.py               ← HospitalSchedulingEnv (step/reset/state)
│   ├── models.py            ← Typed dataclasses (Patient, Doctor, Room, Action…)
│   └── tasks/
│       ├── scenarios.py     ← task_easy / task_medium / task_hard
│       └── graders.py       ← grade() → deterministic score 0.0–1.0
├── baseline/
│   ├── run_baseline.py      ← Greedy agent (reproducible scores)
│   └── run_random.py        ← Random agent (lower bound comparison)
└── tests/
    └── test_env.py          ← 26 unit + integration tests
```

---

## Inference Log Format

The `inference.py` script emits structured JSON to stdout:

```json
{"event": "[START]", "task": "hard", "total_patients": 10, "model": "...", "timestamp": "..."}
{"event": "[STEP]",  "step": 1, "action": {"patient_id": "P01", ...}, "reward": 1.45, "done": false, "info": {...}}
{"event": "[STEP]",  "step": 2, ...}
{"event": "[END]",   "task": "hard", "score": 0.875, "breakdown": {...}, "total_steps": 10, "elapsed_sec": 12.3}
```

---

## Docker

```bash
# Build
docker build -t hospital-env .

# Run server (default)
docker run -p 7860:7860 hospital-env

# Run inference with LLM
docker run \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="hf_..." \
  hospital-env python inference.py

# Run baseline only
docker run hospital-env python baseline/run_baseline.py --task all
```

---

## Deploy to Hugging Face Spaces

1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces) — **Docker** SDK
2. Add tag `openenv` in the Space settings
3. Push this repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/<username>/<space-name>
   git push hf main
   ```
4. Set Space secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
5. HF auto-builds from the `Dockerfile` and serves on port 7860
6. Automated validator will ping `GET /health` → must return 200

---

## Grading Formula

```
score = 0.35 × coverage
      + 0.35 × priority_weighted_coverage
      + 0.15 × compatibility
      + 0.15 × deadline_compliance

# Hard task only:
score -= 0.15 × unscheduled_critical_count
score -= 0.10 × critical_deadline_breaches
score = max(score, 0.0)
```

Graders are **deterministic** — same schedule always produces the same score.

---

## License

MIT