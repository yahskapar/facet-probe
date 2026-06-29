"""Prompt rendering helpers shared by examples and downstream adapters."""

from __future__ import annotations

SYSTEM_INSTRUCTION = (
    "You are a careful, helpful assistant. Read all evidence before answering. "
    "Follow the output-format instruction exactly."
)


def choice_letters(n_choices: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n_choices)]


def build_mcq_question_block(question: str, choices: list[str] | tuple[str, ...]) -> str:
    lines = ["Question: " + question.strip(), "Choices:"]
    for letter, text in zip(choice_letters(len(choices)), choices, strict=True):
        lines.append(f"  {letter}) {str(text).strip()}")
    lines.append("Respond with exactly one line: 'Answer: <LETTER>'.")
    return "\n".join(lines)


def render_text_component(label: str | None, text: str) -> str:
    if label:
        return f"{label}: {text.strip()}"
    return text.strip()
