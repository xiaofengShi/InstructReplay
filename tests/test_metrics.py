import pytest

from instructreplay.metrics import BENCHMARKS, acquisition_forgetting


def test_five_dimensional_metric_keeps_signed_acquisition_and_positive_drops():
    initial = dict(zip(BENCHMARKS, [70, 60, 50, 40, 30]))
    adapted = dict(zip(BENCHMARKS, [65, 61, 50, 35, 25]))
    result = acquisition_forgetting(initial=initial, adapted=adapted,
                                   confirmed_initial=100, confirmed_adapted=164)
    assert result == {"A": 12.5, "F": 3.0, "W": 5.0, "S": 9.5}
    result = acquisition_forgetting(initial=initial, adapted=initial,
                                   confirmed_initial=164, confirmed_adapted=100)
    assert result["A"] == result["S"] == -12.5


def test_missing_humaneval_and_invalid_scores_fail_closed():
    scores = dict.fromkeys(BENCHMARKS, 50)
    for bad in ({k: v for k, v in scores.items() if k != "HumanEval+"},
                {**scores, "BBH": float("nan")}, {**scores, "MMLU": 101}):
        with pytest.raises(ValueError):
            acquisition_forgetting(initial=scores, adapted=bad, confirmed_initial=1, confirmed_adapted=2)


def test_ineligible_and_unknown_questions_remain_in_fixed_denominator():
    scores = dict.fromkeys(BENCHMARKS, 50)
    result = acquisition_forgetting(initial=scores, adapted=scores,
                                   confirmed_initial=0, confirmed_adapted=374)
    assert result["A"] == 100 * 374 / 512
