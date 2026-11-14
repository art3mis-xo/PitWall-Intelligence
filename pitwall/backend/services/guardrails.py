# pitwall/backend/services/guardrails.py
"""Guardrails for the chat agent, shared with evals/agent_eval.py so the app and its eval apply
the same rules.

Input:  screen_input() refuses prompt-injection attempts before any LLM call (and quota) is used.
Output: ungrounded() finds numbers in a reply that no tool returned; the agent attaches a
        visible warning instead of silently showing them.
"""
import re

NUMBER = re.compile(r"\d+(?:\.\d+)?")
# hyphen-minus plus the Unicode dashes LLMs write as minus signs (U+2010, U+2011 non-breaking
# hyphen, U+2012, U+2013, U+2014, U+2212)
MINUS = "-‐‑‒–—−"
SERIES_NAME = re.compile(r"\bF1\b|\bFormula\W*1\b|\bFormula One\b", re.IGNORECASE)


def numbers(text: str) -> list[float]:
    """Every number in the text, including inside "P4" or "R14". A minus sign counts only when it
    starts a token, so a range like "2018-2024" stays two positive years."""
    text = SERIES_NAME.sub(" ", text.replace(",", ""))   # the "1" in "F1" is not a statistic
    out = []
    for m in NUMBER.finditer(text):
        before = text[max(0, m.start() - 2):m.start()]
        negative = len(before) >= 1 and before[-1] in MINUS and (len(before) < 2 or not before[0].isalnum())
        out.append(-float(m.group()) if negative else float(m.group()))
    return out


def matches(n: float, source: float) -> bool:
    # equal up to rounding (62.4 vs 62.42, 0.62 vs 0.624), or a stated percentage written as a
    # fraction (62.4% -> 0.624). Not the reverse: tools print percentages explicitly, so turning
    # 0.105% into 10.5% is an error to catch.
    def close(a, b):
        return abs(a - b) <= (0.051 if abs(b) >= 1 else 0.0051)
    return close(n, source) or close(n, source / 100)


def ungrounded(reply: str, sources: str) -> list[float]:
    """Numbers in the reply that appear nowhere in the sources (tool outputs and user messages)."""
    pool = numbers(sources)
    return [n for n in numbers(reply) if not any(matches(n, s) for s in pool)]


def grounding_note(found: list[float]) -> str:
    unique = list(dict.fromkeys(found))
    shown = ", ".join(f"{n:g}" for n in unique)
    verb, pronoun = ("is", "it") if len(unique) == 1 else ("are", "them")
    return f"\n\n_Note: {shown} {verb} not from PitWall's data or model, so treat {pronoun} as unverified._"


# ── input screen ──────────────────────────────────────────────────────────────
INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|the)\b.{0,20}"
    r"\b(instructions?|rules|prompt|guidelines|directions)\b",
    r"\b(system|developer|hidden|initial|original)\s+(prompt|message|instructions?)\b",
    r"\b(reveal|show|print|repeat|output|display|leak|dump)\b.{0,40}\b(prompt|instructions?|rules)\b",
    r"\byou are now\b|\bact as (an?|the) (?!f1|formula)\w+|\bpretend (to be|you are)\b|\bDAN\b|\bjailbreak",
    r"\b(repeat|print|copy)\b.{0,30}\b(text|everything|words)\b.{0,20}\b(above|before)\b",
]]
REFUSAL = ("I can only help with F1 predictions, explanations and driver statistics from PitWall's data, "
           "and I can't change or reveal my instructions.")


def screen_input(message: str) -> str | None:
    """Return a refusal if the message looks like a prompt-injection attempt, else None."""
    return REFUSAL if any(p.search(message) for p in INJECTION_PATTERNS) else None
