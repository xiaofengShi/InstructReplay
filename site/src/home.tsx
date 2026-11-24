import { renderToStaticMarkup } from "react-dom/server";
import { RiArrowDownLine, RiArrowRightLine, RiExternalLinkLine, RiGithubFill, RiLoopRightLine } from "@remixicon/react";
import { Button, ButtonLink } from "@/components/base/buttons/button";
import { LinkButton } from "@/components/base/buttons/link-button";
import { Chip } from "@/components/base/badges/chip";

const github = "https://github.com/xiaofengShi/InstructReplay";
const project = "https://xiaofengshi.github.io/InstructReplay/";
const title = "InstructReplay · Factual fine-tuning with assistant-turn replay";
const description = "Keep general assistant behavior during factual fine-tuning. Compare six InstructReplay experiments, try the evaluation math, and get a versioned Python implementation.";
const socialDescription = "Across six settings, mean general forgetting fell from 28.07 to 2.44 percentage points while factual acquisition stayed comparable.";

const domains = [
  ["all", "All settings"],
  ["commoncorpus", "CommonCorpus"],
  ["pmc", "PMC"],
] as const;

// The published CSV contains plain scalar fields with no embedded commas.
function readResults(csv: string) {
  const [header, ...rows] = csv.trim().split(/\r?\n/);
  const fields = header.split(",");
  return rows.map((row) => Object.fromEntries(row.split(",").map((value, index) => [fields[index], value])));
}

function Bar({ label, value, max, replay = false }: { label: string; value: string; max: number; replay?: boolean }) {
  return (
    <div className={replay ? "series replay" : "series"}>
      <span>{label}</span>
      <span className="track" aria-hidden="true"><i style={{ width: `${Number(value) / max * 100}%` }} /></span>
      <strong>{value}</strong>
    </div>
  );
}

function SiteHeader() {
  return (
    <header className="site-header">
      <div className="wrap header-row">
        <a className="brand" href="#top"><span className="brand-mark"><RiLoopRightLine aria-hidden="true" /></span>InstructReplay</a>
        <nav aria-label="Project navigation">
          <a href="#results">Results</a>
          <a href="#method">Method</a>
          <a href="#reproduce">Reproduce</a>
        </nav>
        <ButtonLink className="github-button" href={github} variant="secondary" size="small" leadingIcon={RiGithubFill}>GitHub</ButtonLink>
      </div>
    </header>
  );
}

function Hero() {
  return (
    <>
      <div className="hero">
        <div className="wrap hero-grid">
          <div className="hero-copy">
            <p className="eyebrow">Factual fine-tuning · 2026 study</p>
            <h1>Learn new facts.<br /><span>Keep the assistant.</span></h1>
            <p className="intro">InstructReplay reuses the initial model’s own assistant turns during factual fine-tuning to preserve how it responds.</p>
            <div className="actions">
              <ButtonLink href="#results" trailingIcon={RiArrowDownLine}>Explore the results</ButtonLink>
              <ButtonLink href="try/" variant="secondary" trailingIcon={RiArrowRightLine}>Try the metrics</ButtonLink>
            </div>
            <p className="byline">Xiaofeng Shi, Qian Kou, and Hua Zhou<span>Beijing Academy of Artificial Intelligence (BAAI)</span></p>
          </div>
          <figure className="scoreboard dark" aria-labelledby="scoreboard-title">
            <div className="panel-header">
              <h2 id="scoreboard-title" className="text-title-2-medium">Acquisition and forgetting</h2>
              <p>Mean across six settings · percentage points</p>
            </div>
            <div className="comparison">
              <h3>General forgetting <span>↓ lower is better</span></h3>
              <Bar label="Naive SFT" value="28.07" max={30} />
              <Bar label="InstructReplay" value="2.44" max={30} replay />
            </div>
            <div className="comparison">
              <h3>Domain acquisition <span>↑ higher is better</span></h3>
              <Bar label="Naive SFT" value="15.23" max={20} />
              <Bar label="InstructReplay" value="15.49" max={20} replay />
            </div>
            <figcaption>Bars share a scale within each metric.</figcaption>
          </figure>
        </div>
      </div>
      <div className="scope">
        <dl className="wrap">
          <div><dt>Model families</dt><dd>3</dd></div>
          <div><dt>Domains</dt><dd>2</dd></div>
          <div><dt>General benchmarks</dt><dd>5</dd></div>
          <div><dt>Adaptation baselines</dt><dd>7</dd></div>
        </dl>
      </div>
    </>
  );
}

function Results({ csv }: { csv: string }) {
  const rows = readResults(csv);
  return (
    <section id="results" className="wrap section" aria-labelledby="results-title">
      <div className="section-head">
        <div><p className="eyebrow">Main comparison</p><h2 id="results-title">Highest trade-off score in all six settings.</h2></div>
        <p>InstructReplay outscored all seven adaptation baselines on <strong>S = A − F</strong> in each setting. Across the four transfer settings, its lead over the strongest baseline ranged from <strong>5.16 to 24.11 points</strong>.</p>
      </div>
      <div className="table-card">
        <div className="table-toolbar">
          <h3>Per-setting results</h3>
          <LinkButton href="results.csv" download trailingIcon={RiArrowDownLine}>Download CSV</LinkButton>
          <div id="domain-filter" className="filter" hidden>
            <div className="filter-buttons" role="group" aria-label="Filter by domain">
              {domains.map(([value, label]) => (
                <Button key={value} variant="secondary" size="small" data-domain-filter={value} aria-pressed={value === "all"}>{label}</Button>
              ))}
            </div>
            <output id="filter-count" aria-live="polite">{rows.length} of {rows.length} settings</output>
          </div>
        </div>
        <div className="table-scroll" tabIndex={0} role="region" aria-label="Per-setting results, horizontally scrollable on narrow screens">
          <table aria-label="Per-setting results in percentage points">
            <thead><tr><th scope="col">Domain and model</th><th scope="col" className="num">SFT A</th><th scope="col" className="num">SFT F</th><th scope="col" className="num">Replay A</th><th scope="col" className="num">Replay F</th><th scope="col" className="num emphasis">Replay S</th><th scope="col">Best baseline S</th></tr></thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${row.domain}/${row.model}`} data-domain={row.domain.toLowerCase()}>
                  <th scope="row"><span className="setting">{row.domain} · {row.setting}</span>{row.model}</th>
                  <td className="num">{row.naive_sft_A_pp}</td>
                  <td className="num">{row.naive_sft_F_pp}</td>
                  <td className="num">{row.instructreplay_A_pp}</td>
                  <td className="num">{row.instructreplay_F_pp}</td>
                  <td className="num emphasis">{row.instructreplay_S_pp}</td>
                  <td>{row.best_baseline_S_pp.replace("-", "−")} · {row.best_baseline}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="table-note"><strong>A</strong> is the signed gain on 512 closed-book domain questions. <strong>F</strong> averages the nonnegative drops on five General benchmarks. Scores are in percentage points relative to the initial model.</p>
      </div>
      <details className="detail-card">
        <summary>See the change on each General benchmark</summary>
        <div className="table-scroll" tabIndex={0} role="region" aria-label="Mean changes by General benchmark">
          <table aria-label="Mean benchmark changes in percentage points">
            <thead><tr><th scope="col">Method</th><th scope="col" className="num">IFEval</th><th scope="col" className="num">MMLU</th><th scope="col" className="num">MATH-500</th><th scope="col" className="num">BBH</th><th scope="col" className="num">HumanEval+</th></tr></thead>
            <tbody>
              <tr><th scope="row">Naive SFT</th><td className="num">−44.9</td><td className="num">−0.4</td><td className="num">−28.0</td><td className="num">−36.0</td><td className="num">−30.5</td></tr>
              <tr><th scope="row">InstructReplay</th><td className="num">−2.6</td><td className="num">−1.3</td><td className="num">−0.2</td><td className="num">−0.7</td><td className="num">−1.0</td></tr>
            </tbody>
          </table>
        </div>
        <p className="table-note">These signed means differ from F, which clips gains before averaging the drops. MMLU is scored by option likelihood. <a href={`${github}/blob/main/docs/results.md`}>Detailed results and protocol notes</a></p>
      </details>
    </section>
  );
}

function Method() {
  return (
    <>
      <div className="section-band">
        <section id="method" className="wrap section" aria-labelledby="method-title">
          <div className="section-head">
            <div><p className="eyebrow">Method</p><h2 id="method-title">Replay the assistant turn.</h2></div>
            <p>Factual QA teaches short domain answers. Replay keeps the adapted model close to its original behavior on the general prompts and response prefixes where that behavior was expressed.</p>
          </div>
          <ol className="card-grid steps">
            <li className="card"><Chip color="blue" aria-hidden="true">01</Chip><h3>Generate once</h3><p>Run 9,600 general instructions through the initial model’s native chat template and save the sampled replies as exact tokens.</p></li>
            <li className="card"><Chip color="blue" aria-hidden="true">02</Chip><h3>Pair the inputs</h3><p>Train on one domain QA sequence and one saved assistant turn per pair. Reuse the replay buffer for another domain on the same checkpoint.</p></li>
            <li className="card"><Chip color="blue" aria-hidden="true">03</Chip><h3>Match the reference</h3><p>Learn domain answers by cross-entropy. Apply forward KL to match the frozen initial model’s full next-token distributions on response positions in replay.</p></li>
          </ol>
          <div className="equation">
            <code>Loss = domain cross-entropy + 0.25 × replay forward KL</code>
            <p>Each term averages over response tokens in a sequence before averaging over sequences.</p>
          </div>
        </section>
      </div>
      <section id="controls" className="wrap section" aria-labelledby="controls-title">
        <div className="section-head">
          <div><p className="eyebrow">Controlled comparisons</p><h2 id="controls-title">Where replay happens matters.</h2></div>
          <p>Applying the reference loss to the initial model’s assistant turns preserved instruction following better than applying it to domain answers or unprompted samples.</p>
        </div>
        <div className="control-grid">
          <figure className="card" aria-labelledby="control-chart-title">
            <h3 id="control-chart-title">Mean IFEval drop by KL context</h3>
            <Bar label="Domain answers" value="43.3" max={45} />
            <Bar label="BOS samples" value="14.2" max={45} />
            <Bar label="Assistant turns" value="2.6" max={45} replay />
            <figcaption>Percentage points across six settings. BOS replay and assistant-turn replay matched response-token budgets. Domain-KL saw fewer positions.</figcaption>
          </figure>
          <article className="card">
            <h3>The model’s own answers retain more</h3>
            <p>With the same prompts and token loss, replaying answers from the initial model resulted in <strong>3.1 pp</strong> forgetting. Replaying the tested dataset’s answers gave <strong>7.5 pp</strong>.</p>
            <p>On the initial model’s answers, soft targets gave <strong>2.4 pp</strong> forgetting at the same replay weight.</p>
            <LinkButton href={`${github}/blob/main/docs/results.md#replay-controls`} trailingIcon={RiExternalLinkLine}>See all controls</LinkButton>
          </article>
        </div>
      </section>
    </>
  );
}

function Reproduce() {
  return (
    <div className="section-band">
      <section id="reproduce" className="wrap section" aria-labelledby="reproduce-title">
        <div className="section-head">
          <div><p className="eyebrow">Reproduce</p><h2 id="reproduce-title">Run it yourself.</h2></div>
          <p>Try the evaluation math on your CPU or browser. To fine-tune, bring a local checkpoint, licensed data, and the benchmark scorers.</p>
        </div>
        <div className="card-grid entry-list">
          <article className="card">
            <h3>Try the calculation</h3>
            <p>Change five benchmark scores and domain counts, then export the same inputs for the Python CLI.</p>
            <ButtonLink href="try/" trailingIcon={RiArrowRightLine}>Open the playground</ButtonLink>
          </article>
          <article className="card">
            <h3>Get a fixed version</h3>
            <p>Download the tagged source archive and its SHA-256 checksum. The release includes code, results, and examples. <a className="inline-link" href={`${github}/blob/main/README.md#verify-a-release`}>Verify your download</a>.</p>
            <ButtonLink href={`${github}/releases/latest`} variant="secondary" trailingIcon={RiExternalLinkLine}>Latest GitHub release</ButtonLink>
          </article>
          <article className="card">
            <h3>Fine-tune on your data</h3>
            <p>Generate replay once, train on paired inputs, and compare the initial and adapted checkpoints using the same protocol.</p>
            <ButtonLink href={`${github}/blob/main/docs/training.md`} variant="secondary" trailingIcon={RiExternalLinkLine}>Training guide</ButtonLink>
          </article>
        </div>
        <p className="scope-note"><strong>Study scope.</strong> Three 3–4B-class families, two domains, 300 optimization steps, and shared General benchmark questions. Domain acquisition uses model judges without human calibration. The study leaves safety behavior, long-context use, and larger models open.</p>
      </section>
    </div>
  );
}

function Community() {
  return (
    <section id="community" className="wrap section" aria-labelledby="community-title">
      <div className="community-card">
        <div>
          <h2 id="community-title">Share what happened in your run.</h2>
          <p>Successful, partial, and failed reproductions help show where the recipe transfers. The issue form captures the checkpoint, training setup, and evaluation protocol so others can compare results.</p>
        </div>
        <div className="community-links">
          <ButtonLink href={`${github}/issues/new?template=reproduction.yml`} trailingIcon={RiExternalLinkLine}>Share a reproduction</ButtonLink>
          <LinkButton href={`${github}/issues/new?template=bug.yml`}>Report a bug</LinkButton>
          <LinkButton href={`${github}/blob/main/CONTRIBUTING.md`}>Contribute</LinkButton>
        </div>
      </div>
    </section>
  );
}

export function render(styles: string, csv: string) {
  return "<!doctype html>\n<!-- Generated by npm run build --prefix site. -->\n" + renderToStaticMarkup(
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width,initial-scale=1" />
        <meta name="description" content={description} />
        <link rel="canonical" href={project} />
        <meta property="og:type" content="website" />
        <meta property="og:title" content="InstructReplay — Learn new facts. Keep the assistant." />
        <meta property="og:description" content={socialDescription} />
        <meta property="og:url" content={project} />
        <meta property="og:image" content={`${project}assets/result-card.png`} />
        <meta property="og:image:alt" content="Mean forgetting 28.07 versus 2.44 points. Mean acquisition 15.23 versus 15.49 points." />
        <meta name="twitter:card" content="summary_large_image" />
        <title>{title}</title>
        <style>{styles}</style>
        <script type="module" src="assets/home.mjs"></script>
      </head>
      <body id="top">
        <a className="skip" href="#main">Skip to content</a>
        <SiteHeader />
        <main id="main">
          <Hero />
          <Results csv={csv} />
          <Method />
          <Reproduce />
          <Community />
        </main>
        <footer id="citation">
          <div className="wrap">
            <h2>Citation</h2>
            <p>Xiaofeng Shi, Qian Kou, and Hua Zhou. <cite>InstructReplay: Replaying Assistant Turns to Preserve General Behavior During Factual Fine-Tuning.</cite> 2026.</p>
            <div className="footer-links">
              <LinkButton href={`${github}/blob/main/CITATION.cff`}>Citation metadata</LinkButton>
              <LinkButton href={`${github}/blob/main/LICENSE`}>Apache-2.0 software license</LinkButton>
            </div>
          </div>
        </footer>
      </body>
    </html>,
  ) + "\n";
}
