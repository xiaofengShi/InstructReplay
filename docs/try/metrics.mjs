export const benchmarks = ["IFEval", "MMLU", "MATH-500", "BBH", "HumanEval+"];

export function calculateMetrics(scores) {
  const { questions, confirmed_initial, confirmed_adapted, initial, adapted } = scores;
  if (!Number.isSafeInteger(questions) || questions < 1) {
    throw new RangeError("The domain question count must be a positive integer.");
  }
  for (const count of [confirmed_initial, confirmed_adapted]) {
    if (!Number.isSafeInteger(count) || count < 0 || count > questions) {
      throw new RangeError("Confirmed counts must be whole numbers between zero and the question count.");
    }
  }
  for (const values of [initial, adapted]) {
    if (Object.keys(values).length !== benchmarks.length
        || benchmarks.some((name) => !Number.isFinite(values[name]) || values[name] < 0 || values[name] > 100)) {
      throw new RangeError("Enter a percentage between 0 and 100 for all five benchmarks.");
    }
  }
  const drops = benchmarks.map((name) => Math.max(0, initial[name] - adapted[name]));
  const A = 100 * (confirmed_adapted - confirmed_initial) / questions;
  const F = drops.reduce((total, drop) => total + drop, 0) / benchmarks.length;
  return { A, F, W: Math.max(...drops), S: A - F };
}
