# InstructReplay results

These are aggregate results from [*InstructReplay: Replaying Assistant Turns to Preserve General Behavior During Factual Fine-Tuning*](https://arxiv.org/abs/2611.32918) by Xiaofeng Shi, Qian Kou, and Hua Zhou (2026). Use the [CSV](results.csv) for the six-setting results and the [evaluation guide](evaluation.md) for metric inputs and scoring details.

## Main comparison

A is the signed change in domain confirmation out of 512 questions. F averages the positive part of each decline across five General benchmarks and includes zeros for benchmarks with no decline. S is A minus F. All values are percentage points relative to each model’s initial checkpoint. The table reproduces rounded paper values. Differences computed from displayed numbers may vary by 0.01 pp because the paper calculates the metrics before rounding.

| Domain | Initial model | SFT A | SFT F | SFT S | Replay A | Replay F | Replay S | Best baseline by S | ΔS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| CommonCorpus | SmolLM3-3B | 7.62 | 25.10 | −17.49 | 5.66 | 4.66 | **1.01** | Self-Rewrite (−5.83) | +6.84 |
| CommonCorpus | Qwen3-4B | 8.79 | 20.33 | −11.54 | 9.96 | 1.40 | **8.56** | Self-Rewrite (6.82) | +1.74 |
| CommonCorpus | Phi-4-mini-instruct | 9.57 | 23.51 | −13.94 | 9.96 | 2.74 | **7.22** | Self-Rewrite (2.06) | +5.16 |
| PMC | SmolLM3-3B | 20.90 | 42.97 | −22.07 | 22.46 | 4.26 | **18.20** | RAFT (−5.91) | +24.11 |
| PMC | Qwen3-4B | 18.55 | 19.73 | −1.17 | 18.75 | 0.17 | **18.58** | RAFT (9.79) | +8.80 |
| PMC | Phi-4-mini-instruct | 25.98 | 36.81 | −10.83 | 26.17 | 1.39 | **24.78** | RAFT (11.86) | +12.93 |
| Mean of six settings | | **15.23** | **28.07** | | **15.49** | **2.44** | | | |

The seven baselines were Naive SFT, LoRA-SFT, Self-Rewrite, SDFT, DFT, RAFT, and Domain-KL. Replay settings were selected on CommonCorpus with SmolLM3 and Qwen3. The four transfer settings are CommonCorpus with Phi-4-mini and all three PMC runs. Their margins over the strongest baseline by S ranged from +5.16 to +24.11 pp.

### Five General benchmarks

Mean change from the initial checkpoint across the six settings, in pp. Positive values are gains.

| Method | IFEval | MMLU | MATH-500 | BBH | HumanEval+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive SFT | −44.9 | −0.4 | −28.0 | −36.0 | −30.5 |
| InstructReplay | −2.6 | −1.3 | −0.2 | −0.7 | −1.0 |

Per-benchmark signed means and the forgetting metric F summarize different quantities. F averages per-setting losses after treating improvements as zero. InstructReplay’s largest observed MMLU drop was 4.56 pp on CommonCorpus with SmolLM3.

## Replay controls

Means over the same six settings. IFEval length is the median response length relative to the initial checkpoint.

| Replay input | Target | IFEval change (pp) | IFEval length | A (pp) | F (pp) |
| --- | --- | ---: | ---: | ---: | ---: |
| Domain answers (Domain-KL) | Reference distributions | −43.3 | 0.19× | 14.0 | 20.6 |
| Unprompted samples (BOS replay) | Reference distributions | −14.2 | 0.62× | 15.6 | 9.2 |
| Dataset’s assistant answers (Gold replay) | Dataset tokens | −7.4 | 0.74× | 15.7 | 7.5 |
| Initial model’s assistant answers (Hard) | Sampled tokens | −4.0 | 0.82× | 15.0 | 3.1 |
| Initial model’s assistant answers (InstructReplay) | Reference distributions | −2.6 | 0.85× | 15.5 | 2.4 |

BOS replay and Hard matched InstructReplay’s response-token budget. Domain-KL constrained roughly one tenth as many positions. Gold used the same prompts and token loss as Hard, with the dataset’s own replies. Hard versus soft targets were compared at the shared replay weight of 0.25.

## Scope and reproducibility

The study covers three 3–4B-class model families, two domains, and 300 optimizer steps. The settings share General benchmark prompts. Domain scoring uses model judges without human calibration. The five General benchmarks cover instruction following, likelihood-scored knowledge, mathematics, reasoning, and code. Safety behavior, long-context use, and larger models remain untested. The biomedical task measures closed-book answers about articles.

Source: the manuscript’s Main comparison and replay-control tables and the CommonCorpus and PMC endpoint tables in the appendix. Training uses 9,600 domain pairs and one replay example per pair. Scripts for public reproduction, input requirements, and scorer boundaries are in [training](training.md) and [evaluation](evaluation.md).
