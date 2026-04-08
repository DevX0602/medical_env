from pydantic import BaseModel
from typing import List, Optional


class Observation(BaseModel):
    patient_symptoms: str
    history: Optional[str]
    conversation: List[str]
    step_count: int


class Action(BaseModel):
    type: str  # "respond" OR "ask"
    content: str


class Reward(BaseModel):
    score: float
    feedback: str