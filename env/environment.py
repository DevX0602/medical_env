from env.models import Observation, Action, Reward
from env.tasks import get_easy_task, get_medium_task, get_hard_task
from env.graders import grade_response
import random

import random


class MedicalEnv:
    def __init__(self):
        self.task = None
        self.state = None
        self.done = False

    def reset(self, episode_num=0):
        tasks = [
            get_easy_task(),
            get_medium_task(),
            get_hard_task()
        ]

        self.task = tasks[episode_num % len(tasks)]

        self.state = {
            "patient_symptoms": self.task["symptoms"],
            "history": self.task["history"],
            "conversation": [],
            "step_count": 0
        }

        self.done = False
        print(f"[TASK] Difficulty: {self.task['difficulty']}")
        return Observation(**self.state)

    def simulate_patient_reply(self, question):
        reliability = self.task.get("reliability", 1.0)
        q = question.lower()

        # 🔥 unreliable behavior
        if random.random() > reliability:
            return random.choice([
                "I'm not sure",
                "Maybe it's nothing",
                "I feel fine actually",
                "Hard to say"
            ])

        # 🔥 normal responses
        if "pain" in q:
            return "Yes, sharp chest pain"
        if "duration" in q:
            return "Started 2 days ago"
        if "surgery" in q:
            return "Yes, had surgery last week"

        return "I feel weak and tired"

    def step(self, action: Action):
        if self.done:
            raise Exception("Episode finished")

        self.state["step_count"] += 1

        # 🔥 Handle ASK action
        if action.type == "ask":
            self.state["conversation"].append("Agent: " + action.content)

            reply = self.simulate_patient_reply(action.content)
            self.state["conversation"].append("Patient: " + reply)

            reward = Reward(score=0.2, feedback="Good follow-up question")

            return Observation(**self.state), reward, False, {}

        # 🔥 Handle RESPOND action
        self.state["conversation"].append("Agent: " + action.content)

        score, feedback = grade_response(
            action.content,
            self.task,
            self.state["step_count"]
        )

        reward = Reward(score=score, feedback=feedback)

        if score > 0.8 or self.state["step_count"] >= 4:
            self.done = True

        return Observation(**self.state), reward, self.done, {}