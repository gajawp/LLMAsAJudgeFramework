from llm_judge import score_counter_narrative

cn = "Everyone deserves dignity and respect. Differences in identity do not make anyone less valuable."

score = score_counter_narrative(cn)

print(score)