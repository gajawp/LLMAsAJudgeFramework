from standalone_evaluator import evaluate_counter_narrative
import json

cn = "Everyone deserves dignity and respect. Differences in identity do not make anyone less valuable."

result = evaluate_counter_narrative(cn)

print(json.dumps(result, indent=2))