from pathlib import Path
import json
import time
import pandas as pd

from standalone_evaluator import StandaloneCounterNarrativeEvaluator


# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FILE = Path("Multitarget-CONAN.csv")
OUTPUT_FILE = Path("Multitarget-CONAN_scored.csv")

CN_COLUMN = "COUNTER_NARRATIVE"

CONFIG_FILE = Path("config.yaml")

MAX_ROWS = 10        # None -> evaluate entire dataset

CHECKPOINT_EVERY = 10

REQUEST_DELAY_SECONDS = 0.2


# ==========================================================
# LOAD CONFIG
# ==========================================================

with open(CONFIG_FILE) as f:
    cfg = json.load(f) if CONFIG_FILE.suffix == ".json" else __import__("yaml").safe_load(f)

provider = cfg["judge_llm"]["provider"]
model = cfg["judge_llm"]["model"]


# ==========================================================
# INITIALIZE EVALUATOR
# ==========================================================

evaluator = StandaloneCounterNarrativeEvaluator(
    provider=provider,
    model=model
)


# ==========================================================
# READ DATA
# ==========================================================

df = pd.read_csv(INPUT_FILE)

if MAX_ROWS is not None:
    df = df.head(MAX_ROWS)

print(f"Loaded {len(df)} rows")


# ==========================================================
# EVALUATION
# ==========================================================

results = []

for idx, row in df.iterrows():

    cn = str(row[CN_COLUMN]).strip()

    print(f"Evaluating row {idx+1}/{len(df)}")

    try:

        evaluation = evaluator.score(cn)

        llm = evaluation["llm_judge"]

        scores = llm["scores"]
        reasoning = llm["reasoning"]
        summary = llm["summary"]

        reward = evaluation["reward_functions"]

        final_summary = evaluation["final_summary"]

        output_row = row.to_dict()

        output_row.update({

            # --------------------------
            # LLM Judge
            # --------------------------

            "PRS": scores.get("PRS", 0),
            "QS": scores.get("QS", 0),
            "SAFE": scores.get("SAFE", 0),
            "EMP": scores.get("EMP", 0),
            "CON": scores.get("CON", 0),
            "PERS": scores.get("PERS", 0),
            "SPEC": scores.get("SPEC", 0),
            "FLU": scores.get("FLU", 0),
            "TOX": scores.get("TOX", 0),

            "LLM_TOTAL": scores["total"],
            "LLM_MAX_SCORE": scores["max_score"],
            "LLM_PERCENTAGE": scores["percentage"],

            # --------------------------
            # Reasoning
            # --------------------------

            "PRS_REASON": reasoning.get("PRS", ""),
            "QS_REASON": reasoning.get("QS", ""),
            "SAFE_REASON": reasoning.get("SAFE", ""),
            "EMP_REASON": reasoning.get("EMP", ""),
            "CON_REASON": reasoning.get("CON", ""),
            "PERS_REASON": reasoning.get("PERS", ""),
            "SPEC_REASON": reasoning.get("SPEC", ""),
            "FLU_REASON": reasoning.get("FLU", ""),
            "TOX_REASON": reasoning.get("TOX", ""),

            # --------------------------
            # Reward Functions
            # --------------------------

            "R1_SAFETY":
                reward["R1_safety_non_toxicity"],

            "R2_EMPATHY":
                reward["R2_empathy"],

            "R4_NON_CONFRONTATIONAL":
                reward["R4_non_confrontational"],

            "DISTINCT2":
                reward["Distinct2"],

            "COMBINED_REWARD_SCORE":
                reward["combined_reward_score"],

            # --------------------------
            # Final Summary
            # --------------------------

            "OVERALL_QUALITY":
                final_summary["overall_quality"],

            "DEPLOYMENT_READINESS":
                final_summary["deployment_readiness"],

            "STRENGTHS":
                " | ".join(final_summary["strengths"]),

            "WEAKNESSES":
                " | ".join(final_summary["weaknesses"]),

            "RECOMMENDATION":
                final_summary["recommendation"]

        })

    except Exception as e:

        output_row = row.to_dict()

        output_row["ERROR"] = str(e)

    results.append(output_row)

    # --------------------------
    # Checkpoint Save
    # --------------------------

    if (idx + 1) % CHECKPOINT_EVERY == 0:

        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(f"Checkpoint saved after {idx+1} rows")

    time.sleep(REQUEST_DELAY_SECONDS)


# ==========================================================
# FINAL SAVE
# ==========================================================

pd.DataFrame(results).to_csv(
    OUTPUT_FILE,
    index=False
)

print()

print("=" * 60)

print(f"Finished evaluation.")

print(f"Output saved to {OUTPUT_FILE}")

print("=" * 60)