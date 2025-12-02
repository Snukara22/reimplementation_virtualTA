import os
import json
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from wordsegment import load, segment

load()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Create a .env file or export it in your shell."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_LOG = LOG_DIR / "hitl_feedback.jsonl"


def _build_style_instructions(onboarding_pref: str) -> str:
    onboarding_pref = (onboarding_pref or "concise").lower()
    if onboarding_pref == "step_by_step":
        return (
            "Explain the idea step by step, keeping each step short and clear. "
            "Use simple numbered steps."
        )
    if onboarding_pref == "deep_dive":
        return (
            "Give a deeper explanation with more detail and intuition, but keep it organized "
            "and avoid unnecessary fluff."
        )
    return "Give a concise but clear explanation, in 2–6 short paragraphs."


def _chat_completion(messages, temperature: float = 0.4) -> str:
    """Helper to call OpenAI chat completions with the new client."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _build_hint(question: str) -> str:
    prompt = (
        "You are a helpful TA for CSC 517 (Software Engineering).\n"
        "Given the student's exam or homework question, write a short hint (1–3 sentences) "
        "that nudges them in the right direction but does not fully solve it.\n\n"
        f"Question: {question}\n\n"
        "Hint:"
    )

    return _chat_completion(
        [
            {"role": "system", "content": "You provide short, targeted hints."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )


def answer_question(question: str, first_guess: str = "", onboarding_pref: str = "concise"):
    """
    Main entry point used by app.py.

    Returns:
    {
        "answer": "<clean explanation string>",
        "hint": "<short hint>",
        "reflection_questions": [...],
        "next_step": "<one suggestion>"
    }
    """
    question = (question or "").strip()
    first_guess = (first_guess or "").strip()

    if not question:
        return {
            "answer": "Please enter a question about CSC 517 so I can help.",
            "hint": "",
            "reflection_questions": [],
            "next_step": "",
        }

    key_terms = ", ".join(segment(question))[:200]

    style_instructions = _build_style_instructions(onboarding_pref)

    user_prompt_parts = [
        "You are a teaching assistant for CSC 517 (Software Engineering) at NC State.",
        "Explain the answer in a way that helps the student understand and learn, not just memorize.",
        "Do NOT paste large raw textbook chunks. Paraphrase in your own words.",
        "Keep the explanation focused on the question and avoid unrelated details.",
        "",
        f"Question: {question}",
    ]

    if first_guess:
        user_prompt_parts.append(
            f"\nStudent's first attempt / idea:\n{first_guess}\n\n"
            "Start by validating or gently correcting this idea, then give the full explanation."
        )

    user_prompt_parts.append(f"\nStyle: {style_instructions}")

    answer_text = _chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "You are a clear, friendly TA for CSC 517. "
                    "You answer questions accurately and concisely, using examples when helpful."
                ),
            },
            {"role": "user", "content": "\n".join(user_prompt_parts)},
        ],
        temperature=0.4,
    )

    # Generate hint, reflection questions
    try:
        hint_text = _build_hint(question)
    except Exception:
        hint_text = ""

    reflection_questions = [
        "What is the main idea or principle behind this answer?",
        "Which part of the explanation was most confusing, and why?",
        "How would you apply this concept to a slightly different example?",
    ]

    next_step = "Try explaining this concept aloud as if teaching it to a friend, using your own words."

    return {
        "answer": answer_text,
        "hint": hint_text,
        "reflection_questions": reflection_questions,
        "next_step": next_step,
        "keywords": key_terms,
    }


def log_feedback(data: dict):
    """
    Append feedback events to a JSONL log file.
    """
    if not isinstance(data, dict):
        return

    enriched = dict(data)
    enriched["timestamp"] = datetime.utcnow().isoformat()

    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(enriched) + "\n")
