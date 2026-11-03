# evals/run_all.py
import json
import subprocess
import sys
from pathlib import Path

from eval_llm import (
    DEPTH_PROMPTS,
    TEST_CASES,
    eval_depth_calibration,
    eval_factual_accuracy,
    eval_llm_as_judge,
)
from eval_models import eval_tyre_deg_model, eval_race_winner_model

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evals" / "reports"


def run_all_evals():
    report = {}
    REPORTS.mkdir(exist_ok=True)

    print("Running data pipeline evals...")
    # Run pytest programmatically
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "evals/test_data_pipeline.py", "-v",
         "--json-report", f"--json-report-file={REPORTS / 'data_pipeline.json'}"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    print(result.stdout)

    print("\nRunning LLM evals...")
    report["factual_accuracy"]  = eval_factual_accuracy(TEST_CASES)
    report["depth_calibration"] = eval_depth_calibration(DEPTH_PROMPTS)
    report["llm_as_judge"]      = eval_llm_as_judge(TEST_CASES)

    # Save report
    with open(REPORTS / "llm_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("EVAL SUMMARY")
    print("="*50)
    print(f"Factual accuracy:    {report['factual_accuracy']['avg_factual_score']}")
    print(f"Hallucinations:      {report['factual_accuracy']['total_hallucinations']}")
    print(f"Depth gap (fan→eng): {report['depth_calibration']['avg_depth_gap']}")
    print(f"LLM judge accuracy:  {report['llm_as_judge']['avg_accuracy']}/5")

if __name__ == "__main__":
    run_all_evals()