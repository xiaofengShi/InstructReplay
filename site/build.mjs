import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const directory = fileURLToPath(new URL(".", import.meta.url));
const scratch = new URL(".build/", import.meta.url);
await mkdir(scratch, { recursive: true });
execFileSync("./node_modules/.bin/tailwindcss", [
  "-i", "styles/home.css", "-o", ".build/home.css", "--minify",
], { cwd: directory, stdio: "inherit" });
await build({
  absWorkingDir: directory,
  entryPoints: ["src/home.tsx"],
  outfile: ".build/render.mjs",
  platform: "node",
  format: "esm",
  jsx: "automatic",
  bundle: true,
  packages: "external",
});

const { render } = await import(new URL("render.mjs", scratch));
const [styles, results] = await Promise.all([
  readFile(new URL("home.css", scratch), "utf8"),
  readFile(new URL("../docs/results.csv", import.meta.url), "utf8"),
]);
await writeFile(new URL("../docs/index.html", import.meta.url), render(styles, results));
