"""
Paired comparison of the 2026 predictions against the grid-position baseline.

Re-analyses the per-race results saved by evals/final_test.py (no retraining,
no new choices): for each race, difference = model Spearman - baseline Spearman.
  - 95% CI of the mean difference: percentile bootstrap over races (10,000 resamples, seed 0)
  - Wilcoxon signed-rank test, two-sided

    python evals/paired_test.py
Writes evals/results/paired_test_2026.json and .md
"""
import json
import sys

import numpy as np
from scipy.stats import wilcoxon

from common import RESULTS, TEST_SEASON, ROOT, grid_baseline_rank, header, meta, per_race, write

sys.path.insert(0, str(ROOT))
import train  # noqa: E402

N_BOOT, SEED = 10_000, 0


def main():
    saved = json.loads((RESULTS / f"final_test_{TEST_SEASON}.json").read_text())
    df = train.load_dataset()
    te = df[df.season == TEST_SEASON]
    base = per_race(te, grid_baseline_rank(te)).set_index(["season", "round"])["spearman"]
    assert round(float(base.mean()), 3) == saved["results"][0]["spearman_mean"], "baseline differs from final_test"

    rng = np.random.default_rng(SEED)
    rows = []
    for r in saved["results"][1:]:
        model = {(p["season"], p["round"]): p["spearman"] for p in r["per_race"]}
        diff = np.array([model[k] - base[k] for k in base.index])
        boot = rng.choice(diff, size=(N_BOOT, len(diff)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        w = wilcoxon(diff, alternative="two-sided")
        rows.append({"model": r["model"], "config": r["config"], "n_races": len(diff),
                     "mean_diff": round(float(diff.mean()), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                     "races_model_better": int((diff > 0).sum()), "races_baseline_better": int((diff < 0).sum()),
                     "wilcoxon_statistic": float(w.statistic), "wilcoxon_p": round(float(w.pvalue), 3),
                     "per_race_diff": [round(float(d), 4) for d in diff]})

    m = meta("python evals/paired_test.py", df, source=f"evals/results/final_test_{TEST_SEASON}.json",
             bootstrap={"resamples": N_BOOT, "seed": SEED, "method": "percentile, resampling races"})
    lines = header(f"Paired test vs grid baseline ({TEST_SEASON})", m) + [
        f"Per-race difference in Spearman (model − grid baseline) over {rows[0]['n_races']} races of {TEST_SEASON}, "
        f"from `{m['source']}`. 95% CI: percentile bootstrap over races ({N_BOOT:,} resamples, seed {SEED}). "
        "p-value: Wilcoxon signed-rank, two-sided.",
        "",
        "| Model | Mean difference | 95% CI | Races model better / baseline better | Wilcoxon p |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | "
                     f"{r['races_model_better']} / {r['races_baseline_better']} | {r['wilcoxon_p']:.3f} |")
    write(f"paired_test_{TEST_SEASON}", {"meta": m, "results": rows}, lines)


if __name__ == "__main__":
    main()
