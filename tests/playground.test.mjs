import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { calculateMetrics } from "../docs/try/metrics.mjs";

const fixture = JSON.parse(readFileSync("docs/try/example.json", "utf8"));

test("the synthetic example matches the Python CLI", () => {
  const dir = mkdtempSync(join(tmpdir(), "instructreplay-metrics-"));
  try {
    const output = join(dir, "metrics.json");
    execFileSync(process.env.PYTHON ?? "python3", [
      "-m", "instructreplay.metrics", "docs/try/example.json", "--output", output,
    ], { env: { ...process.env, PYTHONPATH: "src" } });
    const expected = { A: 12.5, F: 3, W: 5, S: 9.5 };
    assert.deepEqual(calculateMetrics(fixture), expected);
    assert.deepEqual(JSON.parse(readFileSync(output, "utf8")), expected);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("keeps acquisition signed and clips each benchmark decline", () => {
  const scores = structuredClone(fixture);
  scores.confirmed_adapted = 36;
  scores.adapted.IFEval = 100;
  assert.deepEqual(calculateMetrics(scores), { A: -12.5, F: 2, W: 5, S: -14.5 });
});

test("rejects missing, out-of-range, and non-integer inputs", () => {
  const invalid = [
    { ...fixture, questions: 0 },
    { ...fixture, confirmed_initial: 1.5 },
    { ...fixture, confirmed_adapted: 513 },
    { ...fixture, initial: { ...fixture.initial, MMLU: -1 } },
    { ...fixture, adapted: { ...fixture.adapted, BBH: NaN } },
    { ...fixture, adapted: {} },
  ];
  for (const scores of invalid) {
    assert.throws(() => calculateMetrics(scores), RangeError);
  }
});
