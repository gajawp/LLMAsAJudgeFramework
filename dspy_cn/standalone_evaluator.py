from typing import Any, Dict

from llm_judge import CounterNarrativeJudge
from rewards.base import RewardInput
from rewards.distinct2 import Distinct2Score
from rewards.r1_safety import SafetyNonToxicity
from rewards.r2_empathy import EmpathyReward
from rewards.r4_non_confrontational import NonConfrontationalTone


class StandaloneCounterNarrativeEvaluator:
    """Runs the LLM judge and existing standalone reward functions."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        llm_weight: float = 0.5,
        reward_weight: float = 0.5,
    ):
        if llm_weight < 0 or reward_weight < 0:
            raise ValueError("Score weights must be non-negative")
        if llm_weight + reward_weight == 0:
            raise ValueError("At least one score weight must be positive")

        weight_total = llm_weight + reward_weight
        self.llm_weight = llm_weight / weight_total
        self.reward_weight = reward_weight / weight_total

        # Created once and reused for every CSV row.
        self.judge = CounterNarrativeJudge(provider=provider, model=model)
        self.r1 = SafetyNonToxicity()
        self.r2 = EmpathyReward()
        self.r4 = NonConfrontationalTone()
        self.distinct2 = Distinct2Score()

    def score(self, counter_narrative: str) -> Dict[str, Any]:
        counter_narrative = str(counter_narrative).strip()
        if not counter_narrative:
            raise ValueError("counter_narrative cannot be empty")

        llm_result = self.judge.score(counter_narrative)

        reward_input = RewardInput(
            hate_speech="",
            counter_narrative=counter_narrative,
            ground_truth=None,
        )

        r1 = float(self.r1.score(reward_input))
        r2 = float(self.r2.score(reward_input))
        r4 = float(self.r4.score(reward_input))
        distinct2 = float(self.distinct2.score(reward_input))

        # Distinct-2 is reported separately and is not part of the reward average.
        combined_reward_score = round((r1 + r2 + r4) / 3.0, 6)
        llm_percentage = float(llm_result["scores"]["percentage"])
        reward_percentage = round(combined_reward_score * 100.0, 2)

        # Equal weighting by default; configurable in the constructor.
        final_score_percentage = round(
            self.llm_weight * llm_percentage
            + self.reward_weight * reward_percentage,
            2,
        )

        summary = llm_result["summary"]

        return {
            "counter_narrative": counter_narrative,
            "llm_judge": llm_result,
            "reward_functions": {
                "R1_safety_non_toxicity": r1,
                "R2_empathy": r2,
                "R4_non_confrontational": r4,
                "Distinct2": distinct2,
                "combined_reward_score": combined_reward_score,
            },
            "final_summary": {
                "llm_percentage": llm_percentage,
                "reward_percentage": reward_percentage,
                "final_score_percentage": final_score_percentage,
                "llm_weight": self.llm_weight,
                "reward_weight": self.reward_weight,
                "overall_quality": summary["overall_quality"],
                "deployment_readiness": summary["deployment_readiness"],
                "strengths": summary["strengths"],
                "weaknesses": summary["weaknesses"],
                "recommendation": summary["recommendation"],
            },
        }


def evaluate_counter_narrative(
    counter_narrative: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    evaluator = StandaloneCounterNarrativeEvaluator(
        provider=provider,
        model=model,
    )
    return evaluator.score(counter_narrative)