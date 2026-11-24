import { benchmarks, calculateMetrics } from "./metrics.mjs";

const form = document.querySelector("#scores");
const status = document.querySelector("#status");
const download = document.querySelector("#export-scores");
const reset = document.querySelector("#reset-scores");
const output = document.querySelector("#metrics-output");
const placeholder = document.querySelector("#playground-intro");
let current;

function inputValue(name) {
  return form.elements.namedItem(name).valueAsNumber;
}

function readScores() {
  const scoreMap = (stage) =>
    Object.fromEntries(benchmarks.map((name) => [name, inputValue(`${stage}-${name}`)]));
  return {
    initial: scoreMap("initial"),
    adapted: scoreMap("adapted"),
    confirmed_initial: inputValue("confirmed-initial"),
    confirmed_adapted: inputValue("confirmed-adapted"),
    questions: inputValue("questions"),
  };
}

function showScores(message) {
  download.disabled = true;
  output.dataset.current = "false";
  current = readScores();
  try {
    const metrics = calculateMetrics(current);
    for (const [name, value] of Object.entries(metrics)) {
      document.querySelector(`#metric-${name}`).textContent = new Intl.NumberFormat("en", {
        maximumFractionDigits: 4,
      }).format(value);
    }
    status.textContent = message;
    download.disabled = false;
    output.dataset.current = "true";
  } catch (error) {
    status.textContent = error.message;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  showScores("Metrics updated.");
});

form.addEventListener("input", () => {
  download.disabled = true;
  output.dataset.current = "false";
  status.textContent = "Inputs changed. Select Calculate to update the metrics.";
});

reset.addEventListener("click", () => {
  form.reset();
  showScores("Synthetic example restored.");
});

download.addEventListener("click", () => {
  const file = new Blob([JSON.stringify(current, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(file);
  const link = document.createElement("a");
  link.href = url;
  link.download = "instructreplay-scores.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

showScores("Synthetic example loaded.");
form.hidden = false;
placeholder.hidden = true;
