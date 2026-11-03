# evals/eval_llm.py
"""
LLM eval framework for PitWall's fine-tuned strategy model.
Three eval types:
  1. Factual accuracy   — did it get the facts right?
  2. Depth calibration  — does fan mode vs engineer mode actually differ?
  3. Hallucination rate — did it invent things that didn't happen?
"""

import json
import os
from urllib.request import Request, urlopen
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_ID = os.getenv(
    "PITWALL_EVAL_MODEL",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
)
ADAPTER_PATH = os.getenv("PITWALL_EVAL_ADAPTER")
JUDGE_PROVIDER = os.getenv("EVAL_JUDGE_PROVIDER", "auto").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

model = None
tokenizer = None
judge = None


def _load_model():
    global model, tokenizer

    if model is not None:
        return model, tokenizer

    load_kwargs = {}
    if ADAPTER_PATH:
        adapter_path = Path(ADAPTER_PATH)
        if not adapter_path.is_absolute():
            adapter_path = ROOT / adapter_path
        if not adapter_path.is_dir():
            raise FileNotFoundError(
                f"PITWALL_EVAL_ADAPTER points to a missing directory: {adapter_path}"
            )
        load_kwargs["adapter_path"] = str(adapter_path)

    print(f"Loading evaluation model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID, **load_kwargs)
    return model, tokenizer


def _get_judge():
    global judge
    if judge is None:
        provider = JUDGE_PROVIDER
        if provider == "auto":
            provider = "groq" if os.getenv("GROQ_API_KEY") else "ollama"

        if provider == "groq":
            if not os.getenv("GROQ_API_KEY"):
                raise RuntimeError(
                    "EVAL_JUDGE_PROVIDER=groq requires GROQ_API_KEY in the environment"
                )
            judge = ChatGroq(model=GROQ_MODEL, temperature=0)
        elif provider == "ollama":
            judge = "ollama"
        else:
            raise ValueError(
                "EVAL_JUDGE_PROVIDER must be 'auto', 'groq', or 'ollama'"
            )
    return judge


def _judge_text(judge_client, prompt: str) -> str:
    if judge_client == "ollama":
        request = Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read())['response']
        except OSError as error:
            raise RuntimeError(
                f"Could not reach Ollama at {OLLAMA_URL}. Start it with 'ollama serve' "
                f"and pull {OLLAMA_MODEL!r}."
            ) from error

    return judge_client.invoke(prompt).content


def _generate(prompt: str, max_tokens: int, temperature: float) -> str:
    model, tokenizer = _load_model()
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temperature),
    )

# ── Test cases — ground truth from FastF1 ────────────────────────────────────
TEST_CASES = [
    {
        "id":       "monaco_2024_red_flag",
        "prompt":   "What happened at lap 1 of the 2024 Monaco GP and why did it matter strategically?",
        "facts":    [
            "red flag",
            "collision",
            "Perez",
            "restart",
            "free stop",
        ],
        "must_not_contain": ["safety car deployed", "virtual safety car"],
    },
    {
        "id":       "monaco_2024_winner",
        "prompt":   "Who won the 2024 Monaco Grand Prix and what was their strategy?",
        "facts":    [
            "Leclerc",
            "Ferrari",
            "one stop",
            "hard compound",
        ],
        "must_not_contain": ["Verstappen won", "Red Bull won"],
    },
]

DEPTH_PROMPTS = [
    {"question": "Why did the team pit during the safety car?"},
    {"question": "How should a team choose between an undercut and an overcut?"},
]


# ── Eval 1: Factual accuracy ──────────────────────────────────────────────────
def eval_factual_accuracy(test_cases: list) -> dict:
    model, tokenizer = _load_model()
    results = []

    for tc in test_cases:
        response = _generate(tc["prompt"], max_tokens=300, temperature=0.1)

        # Check must-contain facts
        facts_hit = sum(
            1 for f in tc["facts"]
            if f.lower() in response.lower()
        )
        facts_score = facts_hit / len(tc["facts"])

        # Check must-not-contain (hallucinations)
        hallucinations = [
            f for f in tc["must_not_contain"]
            if f.lower() in response.lower()
        ]

        results.append({
            "id":             tc["id"],
            "facts_score":    round(facts_score, 2),   # target: > 0.8
            "hallucinations": hallucinations,           # target: []
            "response":       response,
        })

    avg_facts = sum(r["facts_score"] for r in results) / len(results)
    total_hallucinations = sum(len(r["hallucinations"]) for r in results)

    return {
        "avg_factual_score":  round(avg_facts, 3),
        "total_hallucinations": total_hallucinations,
        "results":            results,
    }


# ── Eval 2: Depth calibration ─────────────────────────────────────────────────
def eval_depth_calibration(prompts: list) -> dict:
    """
    Fan and engineer prompts about same event — responses should differ
    measurably in technical vocabulary
    """
    TECHNICAL_TERMS = [
        "deg rate", "delta", "undercut", "overcut",
        "tyre life", "stint", "compound", "pit window",
        "net positive", "lap time loss", "s/lap"
    ]

    model, tokenizer = _load_model()
    results = []
    for prompt_pair in prompts:
        fan_response = _generate(
            f"[FAN MODE] {prompt_pair['question']}",
            max_tokens=200,
            temperature=0.3,
        )
        eng_response = _generate(
            f"[ENGINEER MODE] {prompt_pair['question']}",
            max_tokens=200,
            temperature=0.3,
        )

        fan_tech = sum(1 for t in TECHNICAL_TERMS if t in fan_response.lower())
        eng_tech = sum(1 for t in TECHNICAL_TERMS if t in eng_response.lower())

        results.append({
            "question":        prompt_pair["question"],
            "fan_tech_terms":  fan_tech,
            "eng_tech_terms":  eng_tech,
            "gap":             eng_tech - fan_tech,   # target: > 2
        })

    avg_gap = sum(r["gap"] for r in results) / len(results)
    return {
        "avg_depth_gap":   round(avg_gap, 2),   # target: > 2.0
        "calibrated":      avg_gap > 2.0,
        "results":         results,
    }


# ── Eval 3: LLM-as-judge ─────────────────────────────────────────────────────
def eval_llm_as_judge(test_cases: list) -> dict:
    """
    Use a separate free/open model to grade the local model's outputs.
    """
    model, tokenizer = _load_model()
    judge = _get_judge()
    scores = []

    for tc in test_cases:
        response = _generate(tc["prompt"], max_tokens=300, temperature=0.3)

        # The configured judge grades the output.
        judgement = _judge_text(judge, f"""You are evaluating an F1 strategy AI assistant.

Question: {tc["prompt"]}

AI Response: {response}

Ground truth facts that should be present: {tc["facts"]}

Score this response 1-5 on each dimension:
- Accuracy (are the facts correct?)
- Clarity (is it easy to understand?)
- F1 domain knowledge (does it sound like an expert?)

Respond in JSON only:
{{"accuracy": X, "clarity": X, "domain_knowledge": X, "reasoning": "one line"}}""")

        score = json.loads(judgement)
        score["id"] = tc["id"]
        scores.append(score)

    avg = lambda k: round(sum(s[k] for s in scores) / len(scores), 2)
    return {
        "avg_accuracy":        avg("accuracy"),        # target: > 3.5/5
        "avg_clarity":         avg("clarity"),
        "avg_domain_knowledge": avg("domain_knowledge"),
        "scores":              scores,
    }