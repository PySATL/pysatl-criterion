"""Regenerate independent fixtures: python tests/multiple_testing/generate_reference.py."""

import json
from pathlib import Path

import statsmodels
from statsmodels.stats.multitest import multipletests


METHODS = [
    "bonferroni",
    "sidak",
    "holm",
    "holm-sidak",
    "simes-hochberg",
    "hommel",
    "fdr_bh",
    "fdr_by",
    "fdr_tsbh",
    "fdr_tsbky",
]
CASES = [
    [0.04, 0.001, 0.7, 0.02, 0.02],
    [0.01, 0.04, 0.03],
    [0.0, 1.0, 0.0, 0.5],
    [0.01, 0.01, 0.01],
    [0.9, 0.9, 0.9],
    [0.001],
    [1.0],
    [0.001, 0.02, 0.06, 0.2, 0.8],
]


def main():
    records = []
    for method in METHODS:
        for p in CASES:
            for alpha in (0.01, 0.05, 0.1):
                rejected, adjusted, _, _ = multipletests(p, alpha, method, maxiter=1)
                records.append(
                    {
                        "method": method,
                        "p": p,
                        "alpha": alpha,
                        "rejected": rejected.tolist(),
                        "adjusted": adjusted.tolist(),
                    }
                )
    output = {
        "source": f"statsmodels {statsmodels.__version__} multipletests; maxiter=1",
        "records": records,
    }
    path = Path(__file__).parent / "fixtures" / "statsmodels.json"
    path.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
