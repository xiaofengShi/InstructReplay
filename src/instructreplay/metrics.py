"""Five-benchmark acquisition/forgetting metrics in percentage points."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .io import write_new_json

BENCHMARKS = ("IFEval", "MMLU", "MATH-500", "BBH", "HumanEval+")


def acquisition_forgetting(*, initial, adapted, confirmed_initial, confirmed_adapted, questions=512):
    if type(questions) is not int or questions <= 0:
        raise ValueError("questions must be a positive integer")
    for value in (confirmed_initial, confirmed_adapted):
        if type(value) is not int or not 0 <= value <= questions:
            raise ValueError("confirmation counts must be integers within the fixed denominator")
    for scores in (initial, adapted):
        if set(scores) != set(BENCHMARKS):
            raise ValueError("all five named benchmark scores are required")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 100
               for x in scores.values()):
            raise ValueError("benchmark scores must be finite percentages in [0, 100]")
    drops = [max(0.0, initial[b] - adapted[b]) for b in BENCHMARKS]
    acquisition = 100 * (confirmed_adapted - confirmed_initial) / questions
    forgetting = sum(drops) / len(BENCHMARKS)
    return {"A": acquisition, "F": forgetting, "W": max(drops), "S": acquisition - forgetting}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with initial/adapted score maps and confirmation counts")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    write_new_json(args.output, acquisition_forgetting(**json.loads(args.input.read_text())))


if __name__ == "__main__":
    main()
