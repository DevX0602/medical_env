"""
tests/test_env.py
=================
Unit + integration tests for HospitalSchedulingEnv.
Run with:  python -m pytest tests/ -v
       or: python tests/test_env.py
"""
from __future__ import annotations
import sys, os, copy, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hospital_env import HospitalSchedulingEnv, Action
from hospital_env.models import (
    Department, Doctor, Patient, Priority, Room, RoomType,
)
from hospital_env.tasks import task_easy, task_medium, task_hard, grade


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_simple_env():
    """Minimal 1-patient environment for unit tests."""
    patients = [Patient(
        id="P1", name="Test", priority=Priority.HIGH,
        required_department=Department.GENERAL,
        required_room_type=RoomType.CONSULTATION,
        appointment_duration=2, earliest_slot=0,
    )]
    doctors = [Doctor(
        id="D1", name="Dr. Test", department=Department.GENERAL,
        available_slots=list(range(16)),
    )]
    rooms = [Room(
        id="R1", room_type=RoomType.CONSULTATION,
        department=Department.GENERAL,
        available_slots=list(range(16)),
    )]
    return HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)


# ── Test classes ──────────────────────────────────────────────────────────

class TestReset(unittest.TestCase):
    def test_reset_returns_observation(self):
        env = _make_simple_env()
        obs = env.reset()
        self.assertEqual(obs.current_step, 0)
        self.assertEqual(obs.scheduled_count, 0)
        self.assertFalse(obs.done)
        self.assertEqual(len(obs.patients), 1)

    def test_reset_clears_previous_episode(self):
        env = _make_simple_env()
        env.step(Action(patient_id="P1", doctor_id="D1", room_id="R1", start_slot=0))
        obs = env.reset()
        self.assertIsNone(obs.patients[0].assigned_doctor_id)
        self.assertEqual(len(obs.schedule), 0)
        self.assertEqual(obs.reward_so_far, 0.0)


class TestStep(unittest.TestCase):
    def test_valid_action_schedules_patient(self):
        env = _make_simple_env()
        result = env.step(Action("P1", "D1", "R1", 0))
        self.assertGreater(result.reward, 0)
        self.assertTrue(result.done)
        self.assertEqual(result.observation.scheduled_count, 1)

    def test_step_removes_used_slots(self):
        env = _make_simple_env()
        env.step(Action("P1", "D1", "R1", 0))
        obs = env.state()
        # Slots 0 and 1 should be consumed
        self.assertNotIn(0, obs.doctors[0].available_slots)
        self.assertNotIn(1, obs.doctors[0].available_slots)
        self.assertNotIn(0, obs.rooms[0].available_slots)

    def test_invalid_patient_id_returns_penalty(self):
        env = _make_simple_env()
        result = env.step(Action("WRONG", "D1", "R1", 0))
        self.assertLess(result.reward, 0)

    def test_invalid_doctor_id_returns_penalty(self):
        env = _make_simple_env()
        result = env.step(Action("P1", "WRONG", "R1", 0))
        self.assertLess(result.reward, 0)

    def test_double_booking_returns_penalty(self):
        """Doctor slot conflict: same doctor booked twice at overlapping slots."""
        patients = [
            Patient("P1", "A", Priority.LOW, Department.GENERAL, RoomType.CONSULTATION, 2),
            Patient("P2", "B", Priority.LOW, Department.GENERAL, RoomType.CONSULTATION, 2),
        ]
        doctors = [Doctor("D1", "Dr", Department.GENERAL, list(range(16)))]
        rooms   = [
            Room("R1", RoomType.CONSULTATION, Department.GENERAL, list(range(16))),
            Room("R2", RoomType.CONSULTATION, Department.GENERAL, list(range(16))),
        ]
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)
        env.step(Action("P1", "D1", "R1", 0))       # books D1 slots 0-1
        result = env.step(Action("P2", "D1", "R2", 0))  # same doctor, overlapping
        self.assertLess(result.reward, 0)

    def test_slot_out_of_range_returns_penalty(self):
        env = _make_simple_env()
        result = env.step(Action("P1", "D1", "R1", 15))  # duration=2, end=17 > 16
        self.assertLess(result.reward, 0)

    def test_earliest_slot_respected(self):
        patients = [Patient(
            "P1", "A", Priority.LOW, Department.GENERAL,
            RoomType.CONSULTATION, 1, earliest_slot=5,
        )]
        doctors = [Doctor("D1", "Dr", Department.GENERAL, list(range(16)))]
        rooms   = [Room("R1", RoomType.CONSULTATION, Department.GENERAL, list(range(16)))]
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)
        result = env.step(Action("P1", "D1", "R1", 3))  # before earliest
        self.assertLess(result.reward, 0)

    def test_step_after_done_raises(self):
        env = _make_simple_env()
        env.step(Action("P1", "D1", "R1", 0))
        with self.assertRaises(RuntimeError):
            env.step(Action("P1", "D1", "R1", 0))

    def test_already_scheduled_patient_returns_penalty(self):
        """Scheduling an already-assigned patient must return a conflict penalty."""
        patients = [
            Patient("P1", "A", Priority.LOW, Department.GENERAL, RoomType.CONSULTATION, 1),
            Patient("P2", "B", Priority.LOW, Department.GENERAL, RoomType.CONSULTATION, 1),
        ]
        doctors = [Doctor("D1", "Dr", Department.GENERAL, list(range(16)))]
        rooms   = [Room("R1", RoomType.CONSULTATION, Department.GENERAL, list(range(16)))]
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms, max_steps=10)
        env.step(Action("P1", "D1", "R1", 0))           # valid — schedules P1
        result = env.step(Action("P1", "D1", "R1", 4))  # P1 already scheduled → penalty
        self.assertLess(result.reward, 0)


class TestState(unittest.TestCase):
    def test_state_does_not_advance_step(self):
        env = _make_simple_env()
        obs1 = env.state()
        obs2 = env.state()
        self.assertEqual(obs1.current_step, obs2.current_step)

    def test_state_reflects_latest_changes(self):
        env = _make_simple_env()
        env.step(Action("P1", "D1", "R1", 0))
        obs = env.state()
        self.assertEqual(obs.scheduled_count, 1)


class TestDeadlineReward(unittest.TestCase):
    def test_deadline_bonus_within(self):
        patients = [Patient(
            "P1", "A", Priority.MEDIUM, Department.GENERAL,
            RoomType.CONSULTATION, 2, earliest_slot=0, deadline_slot=4,
        )]
        doctors = [Doctor("D1", "Dr", Department.GENERAL, list(range(16)))]
        rooms   = [Room("R1", RoomType.CONSULTATION, Department.GENERAL, list(range(16)))]
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)
        result = env.step(Action("P1", "D1", "R1", 1))   # ends at slot 3 ≤ 4
        self.assertNotIn("deadline_missed", result.info)

    def test_deadline_penalty_breach(self):
        patients = [Patient(
            "P1", "A", Priority.MEDIUM, Department.GENERAL,
            RoomType.CONSULTATION, 2, earliest_slot=0, deadline_slot=2,
        )]
        doctors = [Doctor("D1", "Dr", Department.GENERAL, list(range(16)))]
        rooms   = [Room("R1", RoomType.CONSULTATION, Department.GENERAL, list(range(16)))]
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)
        result = env.step(Action("P1", "D1", "R1", 2))   # ends at slot 4 > 2
        self.assertTrue(result.info.get("deadline_missed"))


class TestDepartmentMismatch(unittest.TestCase):
    def test_dept_mismatch_lowers_reward(self):
        """Scheduling a cardiology patient with a general doctor should penalise."""
        patients = [Patient(
            "P1", "A", Priority.HIGH, Department.CARDIOLOGY,
            RoomType.CONSULTATION, 1,
        )]
        doctors = [
            Doctor("D1", "Dr General", Department.GENERAL, list(range(16))),
            Doctor("D2", "Dr Cardio",  Department.CARDIOLOGY, list(range(16))),
        ]
        rooms = [Room("R1", RoomType.CONSULTATION, Department.GENERAL, list(range(16)))]
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)
        # Use wrong department doctor
        result_wrong  = env.step(Action("P1", "D1", "R1", 0))
        env.reset()
        # Use correct department doctor
        result_correct = env.step(Action("P1", "D2", "R1", 0))
        self.assertLess(result_wrong.reward, result_correct.reward)


class TestMaxStepsBudget(unittest.TestCase):
    def test_episode_ends_at_budget(self):
        """Episode must terminate when max_steps is exhausted."""
        patients = [
            Patient("P1", "A", Priority.LOW, Department.GENERAL, RoomType.CONSULTATION, 1),
            Patient("P2", "B", Priority.LOW, Department.GENERAL, RoomType.CONSULTATION, 1),
        ]
        doctors = [Doctor("D1", "Dr", Department.GENERAL, list(range(16)))]
        rooms   = [Room("R1", RoomType.CONSULTATION, Department.GENERAL, list(range(16)))]
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms, max_steps=1)
        result = env.step(Action("P1", "D1", "R1", 0))
        self.assertTrue(result.done)  # budget hit after 1 step


class TestEasyTask(unittest.TestCase):
    def test_perfect_schedule_scores_high(self):
        patients, doctors, rooms = task_easy()
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)
        env.step(Action("P3", "D1", "R1", 2))
        env.step(Action("P2", "D1", "R1", 0))
        obs = env.state()
        if not obs.done:
            env.step(Action("P1", "D1", "R2", 3))
        obs = env.state()
        score, breakdown = grade("easy", obs)
        self.assertGreaterEqual(score, 0.8)

    def test_empty_schedule_scores_low(self):
        patients, doctors, rooms = task_easy()
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)
        obs = env.state()
        score, _ = grade("easy", obs)
        self.assertLess(score, 0.2)


class TestMediumTask(unittest.TestCase):
    def test_all_departments_present(self):
        patients, doctors, rooms = task_medium()
        depts = {p.required_department for p in patients}
        self.assertIn(Department.CARDIOLOGY, depts)
        self.assertIn(Department.GENERAL, depts)
        self.assertIn(Department.ORTHOPEDICS, depts)

    def test_grade_range(self):
        patients, doctors, rooms = task_medium()
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)
        obs = env.state()
        score, _ = grade("medium", obs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestHardTask(unittest.TestCase):
    def test_critical_patients_exist(self):
        patients, _, _ = task_hard()
        critical = [p for p in patients if p.priority == Priority.CRITICAL]
        self.assertGreaterEqual(len(critical), 2)

    def test_critical_penalty_applied(self):
        """Leaving critical patients unscheduled must push score below medium."""
        patients, doctors, rooms = task_hard()
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)
        obs = env.state()
        score_empty, breakdown = grade("hard", obs)
        self.assertGreater(breakdown.get("unscheduled_critical", 0), 0)
        self.assertEqual(score_empty, 0.0)


class TestGraderOutputRange(unittest.TestCase):
    def _run_greedy(self, task_name):
        from baseline.run_baseline import greedy_agent
        from hospital_env.tasks import task_easy, task_medium, task_hard
        task_map = {"easy": task_easy, "medium": task_medium, "hard": task_hard}
        patients, doctors, rooms = task_map[task_name]()
        env = HospitalSchedulingEnv(patients=patients, doctors=doctors, rooms=rooms)
        return greedy_agent(env, verbose=False)

    def test_easy_score_in_range(self):
        obs = self._run_greedy("easy")
        score, _ = grade("easy", obs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_medium_score_in_range(self):
        obs = self._run_greedy("medium")
        score, _ = grade("medium", obs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_hard_score_in_range(self):
        obs = self._run_greedy("hard")
        score, _ = grade("hard", obs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir=os.path.dirname(__file__), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)