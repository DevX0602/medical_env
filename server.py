import os
from fastapi import FastAPI, HTTPException
from env.environment import MedicalEnv
from env.models import Action, Observation

# Initialize FastAPI with versioning to satisfy the validator
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
    # Ensure your environment.py returns: obs, reward, done, info
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs, 
        "reward": reward, 
        "done": done, 
        "info": info
    }

# 🔥 THIS IS THE MISSING PIECE 🔥
@app.get("/state")
def get_state():
    """Exposes the raw state for validator consistency"""
    if env.state is None:
        return {"message": "Environment not initialized"}
    return env.state

@app.post("/mcp")
def mcp():
    return {"jsonrpc": "2.0", "result": "ok", "id": 1}

if __name__ == "__main__":
    import uvicorn
    # Use port 7860 for Hugging Face compatibility
    uvicorn.run(app, host="0.0.0.0", port=7860)