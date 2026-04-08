def get_easy_task():
    return {
        "symptoms": "fever, cough, sore throat",
        "history": "none",
        "expected": "flu",
        "difficulty": "easy",
        "reliability": 0.9
    }


def get_medium_task():
    return {
        "symptoms": "chest pain, shortness of breath",
        "history": "smoker",
        "expected": "heart attack",
        "difficulty": "medium",
        "reliability": 0.7
    }


def get_hard_task():
    return {
        "symptoms": "mild fever, fatigue",
        "history": "recent surgery",
        "expected": "infection",
        "difficulty": "hard",
        "hidden_risk": True,
        "reliability": 0.5
    }