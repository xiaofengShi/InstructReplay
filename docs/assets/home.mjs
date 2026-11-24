const filter = document.querySelector("#domain-filter");
const buttons = Array.from(filter.querySelectorAll("[data-domain-filter]"));
const count = document.querySelector("#filter-count");
const rows = Array.from(document.querySelectorAll("#results table tr[data-domain]"));

for (const button of buttons) {
  button.addEventListener("click", () => {
    const domain = button.dataset.domainFilter;
    for (const option of buttons) {
      option.setAttribute("aria-pressed", String(option === button));
    }
    for (const row of rows) {
      row.hidden = domain !== "all" && row.dataset.domain !== domain;
    }
    const shown = rows.filter((row) => !row.hidden).length;
    count.textContent = `${shown} of ${rows.length} settings`;
  });
}
filter.hidden = false;
