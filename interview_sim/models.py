from __future__ import annotations

QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question": {"type": "string"},
        "question_type": {
            "type": "string",
            "enum": ["background", "technical", "systems_design", "behavioral", "follow_up"],
        },
        "why_this_interviewer_asks": {"type": "string"},
        "expected_answer": {"type": "string"},
        "evaluation_rubric": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "question",
        "question_type",
        "why_this_interviewer_asks",
        "expected_answer",
        "evaluation_rubric",
    ],
}


EVALUATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number"},
        "verdict": {
            "type": "string",
            "enum": ["strong", "adequate", "weak", "incorrect"],
        },
        "feedback_to_candidate": {"type": "string"},
        "missing_points": {"type": "array", "items": {"type": "string"}},
        "corrected_or_expected_answer": {"type": "string"},
        "suggested_follow_up_question": {"type": "string"},
    },
    "required": [
        "score",
        "verdict",
        "feedback_to_candidate",
        "missing_points",
        "corrected_or_expected_answer",
        "suggested_follow_up_question",
    ],
}