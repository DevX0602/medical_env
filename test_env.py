print("RUNNING FILE")
from env.environment import MedicalEnv
from env.models import Action

env = MedicalEnv()

obs = env.reset()
print("Initial State:", obs)

done = False

while not done:
    action = Action(
        type="respond",
        content = "This may indicate infection due to recent surgery. I recommend follow-up tests and consulting a doctor."
    )

    obs, reward, done, _ = env.step(action)

    print("Reward:", reward)