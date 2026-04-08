from env.environment import MedicalEnv
from env.models import Action
import random

# reproducibility
random.seed(42)
import os

API_BASE_URL = os.getenv("API_BASE_URL", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline")
HF_TOKEN = os.getenv("HF_TOKEN", "none")
def generate_response(obs):
    symptoms = obs.patient_symptoms.lower()
    history = obs.history.lower()

    # Step 0 → ask
    if obs.step_count == 0:
        return "ask", "Can you describe the severity and duration of your symptoms?"

    # Step 1 → main reasoning
    if "not sure" in str(obs.conversation).lower():
        return "respond", (
            "Since the information is unclear, further medical evaluation is necessary "
            "before confirming a diagnosis."
        )
    if obs.step_count == 1:
        if "surgery" in history:
            return "respond", (
                "This could indicate an infection because of your recent surgery. "
                "Further diagnostic tests are needed to confirm."
            )

        if "chest pain" in symptoms:
            return "respond", (
                "This may indicate a heart-related issue because of chest pain and breathlessness. "
                "Please seek immediate medical attention."
            )

        if "fever" in symptoms and "cough" in symptoms:
            return "respond", (
                "This appears to be a flu-like condition because of fever and cough. "
                "Please consult a doctor if symptoms persist."
            )

    # 🔥 Step 2 → refined reasoning (NEW)
    if obs.step_count == 2:
        return "respond", (
            "Based on the symptoms and additional context, this assessment is likely accurate, "
            "but confirmation through medical evaluation is important."
        )

    # 🔥 Step 3 → final conclusion (NEW)
    if obs.step_count >= 3:
        return "respond", (
            "It is strongly recommended to consult a healthcare professional for proper diagnosis "
            "and treatment."
        )

    return "respond", "More information is needed. Please consult a doctor."
print("[START] Running Medical Environment")

env = MedicalEnv()
total_score = 0
episodes = 3

for episode in range(episodes):
    print(f"\n[EPISODE {episode+1}]")

    obs = env.reset(episode)
    done = False
    episode_score = 0   # 🔥 NEW

    print("[STATE]", obs)

    while not done:
        action_type, content = generate_response(obs)

        action = Action(type=action_type, content=content)

        obs, reward, done, _ = env.step(action)

        print("[STEP]")
        print("Action:", action.type, "-", action.content)
        print("Reward:", reward)

        episode_score += reward.score

    print(f"[EPISODE SCORE]: {episode_score:.2f}")  
    total_score += episode_score

print("\n[END]")
print("Final Score:", total_score / episodes)
print(f"[FINAL NORMALIZED SCORE]: {(total_score / episodes)/2:.2f}")