import os
from fastapi import FastAPI, HTTPException
from env.environment import MedicalEnv
from env.models import Action, Observation

app = FastAPI(title="MedicalEnv", version="1.0.0")
env = MedicalEnv()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "medical-env",
        "description": "Multi-step medical reasoning environment for diagnosis under uncertainty"
    }

@app.get("/schema")
def schema():
    # Returns the JSON schema for your Pydantic models
    return {
        "action": Action.schema(),
        "observation": Observation.schema(),
        "state": {"type": "object", "description": "Internal environment state"}
    }

@app.post("/reset")
def reset(episode_num: int = 0):
    return env.reset(episode_num=episode_num)

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.post("/mcp")
def mcp():
    # Minimal JSON-RPC placeholder for the validator
    return {"jsonrpc": "2.0", "result": "ok", "id": 1}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

@app.get("/state")
def get_state():
    """Returns the current raw state of the environment."""
    if env.state is None:
        # If the environment hasn't been reset yet, state might be None
        return {"error": "Environment not initialized. Call /reset first."}
    return env.state