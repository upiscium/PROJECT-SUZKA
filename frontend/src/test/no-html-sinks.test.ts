// @vitest-environment node
//
// IP-17c / D1 §9 G1 precondition: the frontend never executes model output as
// HTML or script. REVISIT TRIGGER: a frontend change that renders HTML or
// model-derived content requires revisiting the approval design (D1 section 9).
// Do not weaken this test to make such a change pass; revisit the design first.
import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const SRC = join(__dirname, "..");
const ROOT = join(SRC, "..");

const SINKS: Array<[string, RegExp]> = [
  ["dangerouslySetInnerHTML", /dangerouslySetInnerHTML/],
  ["innerHTML", /innerHTML/],
  ["outerHTML", /outerHTML/],
  ["insertAdjacentHTML", /insertAdjacentHTML/],
  ["document.write", /document\s*\.\s*write(ln)?\b/],
  ["eval", /(^|[^\w$.])eval\s*\(|\.eval\s*\(|\[\s*["'`]eval["'`]\s*\]/],
  ["new Function", /\bnew\s+Function\b|(^|[^\w$.])Function\s*\(/],
  ["setTimeout/setInterval string", /\bset(Timeout|Interval)\s*\(\s*["'`]/],
  ["rehype-raw", /rehype-raw/],
  ["markdown/html renderer", /\b(react-markdown|marked|markdown-it|remark|rehype|mdx|showdown|snarkdown|micromark|dompurify|sanitize-html|html-react-parser)\b/],
];

const BANNED_DEPS = /^(react-markdown|marked|markdown-it|remark.*|rehype.*|@mdx-js\/.*|mdx.*|showdown|snarkdown|micromark.*|html-react-parser|dompurify|sanitize-html|@next\/mdx)$/;

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(ts|tsx|js|jsx|mjs|cjs|html|md|mdx)$/.test(name)) out.push(p);
  }
  return out;
}

function isTest(path: string): boolean {
  return /\.test\.(ts|tsx)$/.test(path) || path.includes(`${join("src", "test")}`);
}

function stripComments(code: string): string {
  // Strip block and line comments (naive but sufficient; string URLs with // are kept by the lookbehind).
  return code.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:"'`\\])\/\/.*$/gm, "$1");
}

export function findSinks(code: string): string[] {
  const stripped = stripComments(code);
  return SINKS.filter(([, re]) => re.test(stripped)).map(([name]) => name);
}

describe("frontend sink ban (D1 §9 G1)", () => {
  it("detector catches each banned sink", () => {
    for (const snippet of [
      "<div dangerouslySetInnerHTML={{__html: x}} />",
      "el.innerHTML = x",
      "el.outerHTML = x",
      "el.insertAdjacentHTML('beforeend', x)",
      "document.write(x)",
      "eval(x)",
      "window.eval(x)",
      "new Function('return 1')",
      "import raw from 'rehype-raw'",
      "import Markdown from 'react-markdown'",
      "import { marked } from 'marked'",
      "import md from 'markdown-it'",
    ]) {
      expect(findSinks(snippet), snippet).not.toEqual([]);
    }
    expect(findSinks("// innerHTML is banned\n/* eval( */ const a = 1;")).toEqual([]);
  });

  it("no banned sink in frontend/src non-test sources", () => {
    const offenders: string[] = [];
    for (const file of walk(SRC)) {
      if (isTest(relative(ROOT, file))) continue;
      const hits = findSinks(readFileSync(file, "utf8"));
      if (hits.length) offenders.push(`${relative(ROOT, file)}: ${hits.join(", ")}`);
    }
    expect(offenders).toEqual([]);
  });

  it("no markdown/HTML renderer in package.json dependencies", () => {
    const pkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
    const names = Object.keys({ ...pkg.dependencies, ...pkg.devDependencies, ...pkg.optionalDependencies, ...pkg.peerDependencies });
    expect(names.filter((n) => BANNED_DEPS.test(n))).toEqual([]);
  });
});
