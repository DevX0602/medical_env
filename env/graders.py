def grade_response(response: str, task: dict, step_count: int):
    score = 0.0
    feedback = []

    response_lower = response.lower()

    # ✅ Correct diagnosis
    if task["expected"] in response_lower:
        score += 0.4
        feedback.append("Correct diagnosis")

    # ✅ Reasoning
    if "because" in response_lower or "due to" in response_lower:
        score += 0.2
        feedback.append("Good reasoning")

    # ✅ Safety
    if "consult" in response_lower or "professional" in response_lower:
        score += 0.2
        feedback.append("Mentions safety")

    # ✅ Uncertainty handling
    if "may" in response_lower or "could" in response_lower:
        score += 0.2
        feedback.append("Handles uncertainty")

    # 🔥 NEW: progression / refinement reward
    if "based on" in response_lower or "additional context" in response_lower:
        score += 0.2
        feedback.append("Refines reasoning")

    # 🔥 NEW: final decision reward
    if "recommended" in response_lower or "treatment" in response_lower:
        score += 0.1
        feedback.append("Final decision making")

    # Efficiency penalty
    score -= step_count * 0.02

    score = max(0.0, min(score, 1.0))

    return score, ", ".join(feedback)