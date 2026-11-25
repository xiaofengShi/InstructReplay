# Evaluation

The paper compares each adapted checkpoint with its exact initial model on the same questions, tokenizer, prompt templates, decoding rules and scorers. Domain answers are closed-book. Source evidence is available only to the judges, never to the student or its deployment prompt.

## Scores

The final paper uses five General benchmarks: IFEval (120 prompts, prompt-level strict accuracy), MMLU (285 questions, option likelihood), MATH-500 (100 questions), BBH (270 questions) and HumanEval+ (64 tasks). Scores entering aggregation are percentages, not fractions. The main paper's subsets and protocol choices were developed during the study; they are not an independent preregistered final test.

Use the paper's sampling, decoding and scoring specifications for comparability. Generation-based MMLU is a different evaluation. HumanEval+ requires both base and extended EvalPlus tests to pass; model-generated code must execute in an isolated evaluator. IFEval character-frequency constraints must use literal character counts, and its fallback randomness must be seeded. BBH answers require task-aware extraction. Fully indented Python function responses must be dedented before execution. These scorer choices can materially affect the scores.

This package aggregates already scored results; it does not ship the benchmark harnesses, judge services or evaluation question text. The [arXiv preprint and its ancillary files](https://arxiv.org/abs/2611.32918) provide the detailed scoring protocol and final tables.

## Metrics

Let `C` be the number of domain questions whose core answer receives a confirming majority of three judges. The denominator is fixed at 512, including ineligible and unresolved questions. Model-judged confirmation is an uncalibrated proxy and should not be described as human-verified accuracy.

- `A = 100 * (C_adapted - C_initial) / 512`, a signed acquisition change.
- `F = sum(max(0, initial[b] - adapted[b])) / 5` across all five General benchmarks.
- `W` is the largest single-benchmark drop.
- `S = A - F`.

These metrics are in percentage points. Improved General scores do not offset losses in `F`. A positive `S` does not establish noninferiority or no forgetting. Report the five raw scores alongside A/F/W/S.

```json
{
  "initial": {"IFEval": 70, "MMLU": 60, "MATH-500": 50, "BBH": 40, "HumanEval+": 30},
  "adapted": {"IFEval": 65, "MMLU": 61, "MATH-500": 50, "BBH": 35, "HumanEval+": 25},
  "confirmed_initial": 100,
  "confirmed_adapted": 164,
  "questions": 512
}
```

Save this object as `scores.json`, then run `instructreplay-metrics scores.json --output metrics.json`. This example gives `A=12.5`, `F=3`, `W=5`, `S=9.5`. Missing dimensions are rejected rather than silently switching to a four-benchmark result.
