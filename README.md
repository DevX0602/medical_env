# 🏥 Medical Diagnosis RL Environment (OpenEnv)

## 📌 Overview
This project implements a **real-world reinforcement learning environment** for medical diagnosis.

The agent interacts with a simulated patient and must:
- Ask relevant follow-up questions
- Provide a diagnosis
- Refine reasoning
- Recommend next steps

The goal is to evaluate **multi-step reasoning under uncertainty**, not just final answers.

---

## 🎯 Tasks

### 1. Easy Task
- Symptoms: fever, cough, sore throat  
- Goal: Identify flu-like condition

### 2. Medium Task
- Symptoms: mild fever, fatigue  
- History: recent surgery  
- Goal: Identify possible infection

### 3. Hard Task
- Requires:
  - Asking clarifying questions  
  - Handling uncertainty  
  - Providing safe medical advice  

---

## 🧠 Action Space

| Action Type | Description |
|------------|------------|
| ask        | Ask follow-up questions |
| respond    | Provide diagnosis or recommendation |

---

## 👀 Observation Space

```json
{
  "patient_symptoms": "string",
  "history": "string",
  "conversation": "list",
  "step_count": "int"
}