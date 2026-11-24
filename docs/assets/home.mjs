const filter = document.querySelector("#domain-filter");
const picker = document.querySelector("#results-domain");
const count = document.querySelector("#filter-count");
const rows = Array.from(document.querySelectorAll("#results table tr[data-domain]"));

filter.hidden = false;
picker.addEventListener("change", () => {
  const domain = picker.value;
  for (const row of rows) {
    row.hidden = domain !== "all" && row.dataset.domain !== domain;
  }
  const shown = rows.filter((row) => !row.hidden).length;
  count.textContent = `${shown} of ${rows.length} settings`;
});
