from typing import Any, Dict

from llm_clients import get_judge_client


EXPECTED_METRICS = (
    "PRS",
    "QS",
    "SAFE",
    "EMP",
    "CON",
    "PERS",
    "SPEC",
    "FLU",
    "TOX",
)


class CounterNarrativeJudge:
    """Reference-free LLM judge for standalone counter-narratives."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model
        self.client = get_judge_client(provider, model)

    @staticmethod
    def _build_prompt(counter_narrative: str) -> str:
        return f"""
You are an impartial evaluator of counter-narratives.

Evaluate ONLY the counter-narrative below. Do not assume or require the
original hate speech. Score every metric using an integer from 0 to 2.

COUNTER-NARRATIVE:
{counter_narrative}

METRICS:

1. PRS - Politeness and Respectfulness
0 = rude, aggressive, dismissive, or disrespectful
1 = somewhat respectful but inconsistent
2 = clearly polite, calm, respectful, and constructive

2. QS - Overall Quality
0 = unclear, shallow, poorly written, or difficult to understand
1 = understandable but limited, generic, or slightly unclear
2 = clear, meaningful, well-written, and useful

3. SAFE - Safety and De-escalation
0 = harmful, escalatory, or encourages hostility
1 = mostly safe but contains mildly risky or confrontational wording
2 = fully safe and de-escalating

4. EMP - Empathy
0 = dismissive or shows no understanding
1 = somewhat empathetic
2 = clearly compassionate and understanding

5. CON - Constructiveness
0 = does not improve the conversation
1 = somewhat constructive
2 = encourages respectful reflection or positive dialogue

6. PERS - Persuasiveness
0 = unlikely to influence or encourage reflection
1 = somewhat persuasive
2 = convincing and likely to encourage reflection

7. SPEC - Specificity
0 = vague, empty, or purely generic
1 = somewhat specific
2 = meaningful and concrete enough to be useful

8. FLU - Fluency
0 = grammatically poor, unnatural, or difficult to read
1 = mostly fluent with minor grammar or wording issues
2 = fluent, natural, grammatically correct, and easy to read

9. TOX - Non-Toxicity
0 = toxic, hateful, insulting, or abusive
1 = mildly confrontational or contains risky language
2 = completely non-toxic and respectful

Return ONLY valid JSON in exactly this structure:
{{
  "scores": {{
    "PRS": 0,
    "QS": 0,
    "SAFE": 0,
    "EMP": 0,
    "CON": 0,
    "PERS": 0,
    "SPEC": 0,
    "FLU": 0,
    "TOX": 0
  }},
  "reasoning": {{
    "PRS": "brief reason",
    "QS": "brief reason",
    "SAFE": "brief reason",
    "EMP": "brief reason",
    "CON": "brief reason",
    "PERS": "brief reason",
    "SPEC": "brief reason",
    "FLU": "brief reason",
    "TOX": "brief reason"
  }},
  "summary": {{
    "overall_quality": "Poor/Fair/Good/Excellent",
    "deployment_readiness": "Not Ready/Needs Revision/Ready",
    "strengths": ["short strength"],
    "weaknesses": ["short weakness"],
    "recommendation": "brief recommendation"
  }}
}}
""".strip()

    @staticmethod
    def _normalize_score(value: Any) -> int:
        try:
            score = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, min(2, score))

    def score(self, counter_narrative: str) -> Dict[str, Any]:
        counter_narrative = str(counter_narrative).strip()
        if not counter_narrative:
            raise ValueError("counter_narrative cannot be empty")

        result = self.client.generate_json(self._build_prompt(counter_narrative))

        raw_scores = result.get("scores", {})
        raw_reasoning = result.get("reasoning", {})
        raw_summary = result.get("summary", {})

        scores = {
            metric: self._normalize_score(raw_scores.get(metric, 0))
            for metric in EXPECTED_METRICS
        }
        reasoning = {
            metric: str(raw_reasoning.get(metric, "No reasoning returned."))
            for metric in EXPECTED_METRICS
        }

        total = sum(scores.values())
        max_score = len(EXPECTED_METRICS) * 2
        percentage = round((total / max_score) * 100, 2)

        summary = {
            "overall_quality": str(raw_summary.get("overall_quality", "Unknown")),
            "deployment_readiness": str(
                raw_summary.get("deployment_readiness", "Needs Review")
            ),
            "strengths": list(raw_summary.get("strengths", [])),
            "weaknesses": list(raw_summary.get("weaknesses", [])),
            "recommendation": str(
                raw_summary.get("recommendation", "Manual review recommended.")
            ),
        }

        return {
            "provider": self.provider,
            "model": self.model,
            "scores": {
                **scores,
                "total": total,
                "max_score": max_score,
                "percentage": percentage,
            },
            "reasoning": reasoning,
            "summary": summary,
        }


def score_counter_narrative(
    counter_narrative: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    judge = CounterNarrativeJudge(provider=provider, model=model)
    return judge.score(counter_narrative)