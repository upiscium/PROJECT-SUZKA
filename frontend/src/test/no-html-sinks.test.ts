// @vitest-environment node
//
// IP-17c / D1 §9 G1 precondition: the frontend never executes model output as
// HTML or script. REVISIT TRIGGER: a frontend change that renders HTML or
// model-derived content requires revisiting the approval design (D1 section 9).
// Do not weaken this test to make such a change pass; revisit the design first.
//
// This is a TRIPWIRE, not a proof: it parses sources with the TypeScript
// parser (so comments and ordinary strings never hide or fake a sink) and
// flags known sink identifiers and markdown/HTML-renderer module specifiers.
// Indirect access (aliasing, computed names built at runtime) is not detected.
import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import ts from "typescript";

const SRC = join(__dirname, "..");
const ROOT = join(SRC, "..");

const SINK_NAMES = new Set(["dangerouslySetInnerHTML", "innerHTML", "outerHTML", "insertAdjacentHTML"]);
const BANNED_MODULE_RE =
  /^(react-markdown|marked|marked-react|markdown-it|markdown-to-jsx|commonmark|unified|showdown|snarkdown|micromark.*|remark.*|rehype.*|mdx.*|@mdx-js\/.*|@next\/mdx|@uiw\/react-md-editor|html-react-parser|dompurify|isomorphic-dompurify|sanitize-html|xss|highlight\.js)(\/.*)?$/;
const BANNED_DEPS = new RegExp(BANNED_MODULE_RE.source.replace(/\(\\\/\.\*\)\?\$$/, "$"));

export function findSinks(code: string): string[] {
  const sf = ts.createSourceFile("x.tsx", code, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const hits = new Set<string>();
  const checkModule = (text: string) => {
    if (BANNED_MODULE_RE.test(text)) hits.add(`module:${text}`);
  };
  const strArg = (n: ts.Node | undefined) => (n && ts.isStringLiteralLike(n) ? n.text : undefined);
  const visit = (node: ts.Node) => {
    if (ts.isIdentifier(node) || ts.isPrivateIdentifier(node)) {
      if (SINK_NAMES.has(node.text)) hits.add(node.text);
    }
    if (ts.isJsxAttribute(node) && SINK_NAMES.has(node.name.getText())) hits.add(node.name.getText());
    if (ts.isElementAccessExpression(node)) {
      const k = strArg(node.argumentExpression);
      if (k && SINK_NAMES.has(k)) hits.add(k);
      if (k === "eval" || k === "Function") hits.add(k);
      if (k === "write" || k === "writeln") hits.add("document.write");
    }
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier) {
      const m = strArg(node.moduleSpecifier);
      if (m) checkModule(m);
    }
    if (ts.isImportEqualsDeclaration(node) && ts.isExternalModuleReference(node.moduleReference)) {
      const m = strArg(node.moduleReference.expression);
      if (m) checkModule(m);
    }
    if (ts.isCallExpression(node) || ts.isNewExpression(node)) {
      const callee = node.expression;
      const name = ts.isIdentifier(callee) ? callee.text : ts.isPropertyAccessExpression(callee) ? callee.name.text : undefined;
      if (name === "eval" || name === "Function") hits.add(name);
      if ((name === "setTimeout" || name === "setInterval") && strArg(node.arguments?.[0]) !== undefined) hits.add(`${name}(string)`);
      if (
        ts.isPropertyAccessExpression(callee) &&
        (callee.name.text === "write" || callee.name.text === "writeln") &&
        ts.isIdentifier(callee.expression) &&
        callee.expression.text === "document"
      )
        hits.add("document.write");
      if (ts.isCallExpression(node) && (callee.kind === ts.SyntaxKind.ImportKeyword || (ts.isIdentifier(callee) && callee.text === "require"))) {
        const m = strArg(node.arguments[0]);
        if (m) checkModule(m);
      }
    }
    // bare reference such as `setTimeout(eval)` or `const f = eval`
    if (ts.isIdentifier(node) && (node.text === "eval" || node.text === "Function") && !ts.isCallExpression(node.parent) && !ts.isNewExpression(node.parent)) {
      if (!(ts.isPropertyAccessExpression(node.parent) && node.parent.name === node) && !ts.isPropertyAssignment(node.parent)) hits.add(node.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return [...hits];
}

export function isTestPath(relPath: string): boolean {
  const p = relPath.split(sep).join("/");
  return p.startsWith("src/test/") || /\.test\.tsx?$/.test(p);
}

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(ts|tsx|js|jsx|mjs|cjs|mts|cts)$/.test(name)) out.push(p);
  }
  return out;
}

describe("frontend sink ban (D1 §9 G1)", () => {
  it.each([
    "<div dangerouslySetInnerHTML={{__html: x}} />",
    "el.innerHTML = x",
    "el.outerHTML = x",
    "el.insertAdjacentHTML('beforeend', x)",
    "el['innerHTML'] = x",
    "document.write(x)",
    "eval(x)",
    "window.eval(x)",
    "(0, eval)(x)",
    "const f = eval; f(x)",
    "window['eval'](x)",
    "new Function('return 1')",
    "Function('return 1')()",
    "globalThis.Function('return 1')",
    "setTimeout('alert(1)', 1)",
    "import raw from 'rehype-raw'",
    "import Markdown from 'react-markdown'",
    "import { marked } from 'marked'",
    "import md from 'markdown-it'",
    "import('react-markdown')",
    "require('marked')",
    "export * from 'remark-gfm'",
    "import hljs from 'highlight.js/lib/core'",
    // evasion probes: comment/string markers must not hide a real sink
    'const a = "x//y"; el.innerHTML = z;',
    'const b = "/*"; el.innerHTML = z; const c = "*/";',
    "const t = `//`; el.innerHTML = z;",
  ])("detects %s", (snippet) => {
    expect(findSinks(snippet), snippet).not.toEqual([]);
  });

  it.each([
    "// innerHTML is banned\n/* eval( */ const a = 1;",
    'const label = "Add a remark"; const m = "marked as read";',
    "const eval_count = 1; const marked = true;",
    "setTimeout(() => 1, 5)",
    "import { clsx } from 'clsx'",
  ])("does not flag %s", (snippet) => {
    expect(findSinks(snippet), snippet).toEqual([]);
  });

  it("test-path exemption is narrow", () => {
    expect(isTestPath("src/test/setup.ts")).toBe(true);
    expect(isTestPath("src/lib/api.test.ts")).toBe(true);
    expect(isTestPath("src/testing/x.ts")).toBe(false);
    expect(isTestPath("src/testbed/x.ts")).toBe(false);
    expect(isTestPath("src/test-utils/x.ts")).toBe(false);
    expect(isTestPath("src/lib/foo.test.ts.bak.ts")).toBe(false);
    expect(isTestPath("src/lib/api.testx.ts")).toBe(false);
  });

  it("no banned sink in frontend/src non-test sources", () => {
    const offenders: string[] = [];
    for (const file of walk(SRC)) {
      const rel = relative(ROOT, file);
      if (isTestPath(rel)) continue;
      const hits = findSinks(readFileSync(file, "utf8"));
      if (hits.length) offenders.push(`${rel}: ${hits.join(", ")}`);
    }
    expect(offenders).toEqual([]);
  });

  it("no markdown/HTML renderer in package.json dependencies", () => {
    const pkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
    const bundled = Array.isArray(pkg.bundledDependencies) ? pkg.bundledDependencies : Array.isArray(pkg.bundleDependencies) ? pkg.bundleDependencies : [];
    const names = [
      ...Object.keys({ ...pkg.dependencies, ...pkg.devDependencies, ...pkg.optionalDependencies, ...pkg.peerDependencies }),
      ...bundled,
    ];
    expect(names.filter((n: string) => BANNED_DEPS.test(n))).toEqual([]);
    expect(BANNED_DEPS.test("markdown-to-jsx")).toBe(true);
  });
});
