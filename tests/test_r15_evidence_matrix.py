"""Static R15 index checks, not execution evidence or Human acceptance."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "r15-evidence.md"
EXPECTED_IDS = {f"A{n}" for n in range(1, 28)} | {f"F{n}" for n in range(1, 23)}
CLASSES = {
    "MVI_WITNESSED",
    "MVI_NEGATIVE_PROVEN",
    "UNAVAILABLE_FAIL_CLOSED",
    "HUMAN_DEFERRED",
    "PARTIAL_OR_UNVERIFIED",
}
HEADINGS = (
    "D9",
    "Capacity",
    "Privacy",
    "Source",
    "U2",
    "U3",
    "U4",
    "U5",
    "U6",
    "U7",
    "Authority",
    "CI",
)
TEST_REF = re.compile(r"tests/test_[A-Za-z0-9_]+\.py::test_[A-Za-z0-9_]+")
INLINE = re.compile(r"`([^`\n]+)`")
ROW_START = re.compile(r"^\|\s*([AF][0-9]+)\b")
PLACEHOLDERS = {"none", "n/a", "todo", "tbd", "pending", "-", "—"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def meaningful(value: str) -> bool:
    return value.strip().lower() not in PLACEHOLDERS and len(value.split()) >= 4


def check_callable(source: str, name: str, reference: str) -> None:
    """Only unique top-level synchronous test definitions are valid nodes.

    Names are module-qualified: identical names in different modules are legal
    pytest nodes. Nested helpers/methods and duplicate definitions in one module
    must not satisfy a reference. Parse only; never import a referenced module.
    """
    tree = ast.parse(source, filename=reference)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
    ]
    require(
        len(definitions) == 1,
        f"missing or nonunique test callable: {reference}",
    )
    definition = definitions[0]
    if not isinstance(definition, ast.FunctionDef):
        raise ValueError(f"missing or nonunique test callable: {reference}")
    require(not definition.args.posonlyargs, f"invalid pytest node: {reference}")


def check_reference(reference: str, root: Path) -> None:
    path_text, separator, name = reference.partition("::")
    parts = Path(path_text).parts
    require(
        not Path(path_text).is_absolute()
        and len(parts) >= 2
        and parts[0] in {"tests", "docs"}
        and ".." not in parts,
        f"invalid local reference: {reference}",
    )
    path = root / path_text
    require(not path.is_symlink(), f"symlink reference: {reference}")
    require(path.is_file(), f"missing local reference: {reference}")
    require(path.resolve().is_relative_to(root.resolve()), "reference escapes root")
    if parts[0] == "tests":
        require(
            len(parts) == 2
            and bool(TEST_REF.fullmatch(reference))
            and separator == "::",
            f"invalid test reference: {reference}",
        )
        check_callable(path.read_text(encoding="utf-8"), name, reference)
    else:
        require(not separator and path.suffix == ".md", "invalid document reference")


def check_index(text: str, root: Path = ROOT) -> dict[str, tuple[str, ...]]:
    """Refuse structural/reference errors without asserting tests passed.

    This is a consistency check, not a semantic proof engine. In particular it
    cannot authenticate quoted Human decisions, live CI, or real-model logs.
    """
    rows: dict[str, tuple[str, ...]] = {}
    for line in text.splitlines():
        if not ROW_START.match(line):
            continue
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        require(len(cells) == 6, "malformed obligation row")
        clause, intent, classification, evidence, boundary, unresolved = cells
        require(clause in EXPECTED_IDS, f"unknown clause: {clause}")
        require(clause not in rows, f"duplicate clause: {clause}")
        require(len(intent.split()) >= 3, f"empty intent: {clause}")
        require(classification in CLASSES, f"invalid classification: {clause}")
        require(meaningful(boundary), f"empty evidence boundary: {clause}")
        require(meaningful(unresolved), f"empty unresolved condition: {clause}")
        refs = INLINE.findall(evidence)
        require(bool(refs), f"missing proof: {clause}")
        require(
            all(TEST_REF.fullmatch(ref) for ref in refs),
            f"invalid evidence reference: {clause}",
        )
        # Reject bare paths or trailing prose masquerading as callable evidence.
        remainder = INLINE.sub("", evidence).replace(";", "").strip()
        require(not remainder, f"unparsed evidence: {clause}")
        require(len(refs) == len(set(refs)), f"duplicate proof: {clause}")
        rows[clause] = cells
    require(set(rows) == EXPECTED_IDS, "missing A/F obligation coverage")

    # Check EVERY in-repository reference, not just the 49 evidence cells.
    # Local Markdown links are deliberately unsupported so no link can evade
    # the canonical inline docs/path.md or tests/path.py::test_node grammar.
    for target in re.findall(r"\]\(([^)]+)\)", text):
        require(target.startswith("https://github.com/"), "unsupported link target")
    for inline in INLINE.findall(text):
        if inline.startswith(("tests/", "docs/")):
            check_reference(inline, root)
    # A malformed unquoted test path must not silently escape inline checks.
    prose = re.sub(r"^```[^\n]*\n.*?^```\s*$", "", text, flags=re.MULTILINE | re.DOTALL)
    quoted_refs = Counter(INLINE.findall(prose))
    for candidate in re.findall(r"tests/[^\s`;|)]+", prose):
        require(bool(TEST_REF.fullmatch(candidate)), f"unparsed test node: {candidate}")
        require(quoted_refs[candidate] > 0, "unquoted test reference")
        quoted_refs[candidate] -= 1

    sections: dict[str, str] = {}
    matches = list(re.finditer(r"^### ([A-Za-z0-9]+) — .+$", text, re.MULTILINE))
    for index, match in enumerate(matches):
        key = match.group(1)
        require(key not in sections, f"duplicate crosscutting section: {key}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[key] = text[match.end() : end]
        require(len(sections[key].split()) >= 25, f"empty crosscutting section: {key}")
    require(set(sections) == set(HEADINGS), "missing crosscutting section")

    # Narrow semantic guardrails: refuse known false-positive classifications.
    for clause in ("A3", "F16"):
        row = rows[clause]
        require(
            row[2] == "UNAVAILABLE_FAIL_CLOSED", f"identity false positive: {clause}"
        )
        require(
            "producer unavailable" in row[4].lower(),
            f"missing identity caveat: {clause}",
        )
        require("not proven" in row[5], f"missing original-scope caveat: {clause}")
    for clause in ("A8", "F6"):
        row = rows[clause]
        require(row[2] == "HUMAN_DEFERRED", f"deferral misclassified: {clause}")
        require("#324" in row[5] and "not proven" in row[5], "missing Human deferral")
    require(
        {key for key, row in rows.items() if row[2] == "HUMAN_DEFERRED"}
        == {"A8", "F6"},
        "uncontrolled Human deferral",
    )
    row = rows["A19"]
    require(row[2] == "PARTIAL_OR_UNVERIFIED", "A19 recovery misclassified")
    require(
        "#324" in row[5] and "not proven" in row[5],
        "missing A19 advanced-only deferral",
    )
    for clause in ("A12", "F9", "F14"):
        require("UNKNOWN" in " ".join(rows[clause][4:]), "competence not UNKNOWN")
    for clause in ("A26", "A27", "F22"):
        require(rows[clause][2] == "PARTIAL_OR_UNVERIFIED", "stage gate false positive")
    require(
        all("test_r14" not in ref for ref in INLINE.findall(rows["F22"][3])),
        "another stage's F22 is not R15 evidence",
    )
    caveats = {
        "Capacity": ("133,729,894", "16,777,216", "134,217,728", "487,834"),
        "U2": ("#323/#324", "positive lasting Relationship", "not proven"),
        "U3": ("no second revision", "UNKNOWN", "actual human intent"),
        "U4": ("verified competence remains UNKNOWN",),
        "U7": (
            "SCIENTIFIC EVIDENCE ONLY",
            "unsupported monetary claims",
            "case B confidence",
            "increased with no E2",
            "external_directory DENY",
            "#343",
        ),
        "Authority": ("no whole-R15", "Human acceptance or deployment", "independent"),
        "CI": (
            "40176534738",
            "4b1dc8486a55a2fd2a8a459912f533e7c27b30f7",
            "not for this uncommitted two-file change",
        ),
    }
    for section, phrases in caveats.items():
        normalized = " ".join(sections[section].split())
        for phrase in phrases:
            require(phrase in normalized, f"missing {section} caveat: {phrase}")
    require("6058471359" in text and "6058579682" in text, "missing frozen sources")
    return rows


def document() -> str:
    return DOCUMENT.read_text(encoding="utf-8")


def replace_row(text: str, clause: str, replacement: str) -> str:
    lines = text.splitlines()
    matches = [
        index for index, line in enumerate(lines) if line.startswith(f"| {clause} |")
    ]
    require(len(matches) == 1, "fixture must select exactly one row")
    lines[matches[0]] = replacement
    return "\n".join(lines)


def row_text(text: str, clause: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"| {clause} |"))


def test_r15_evidence_index_is_complete_and_source_checked() -> None:
    rows = check_index(document())
    assert len(rows) == 49
    assert sum(key.startswith("A") for key in rows) == 27
    assert sum(key.startswith("F") for key in rows) == 22


@pytest.mark.parametrize("clause", ["A1", "F1"])
@pytest.mark.parametrize("tamper", ["missing", "duplicate", "unknown", "empty_proof"])
def test_obligation_row_tampering_is_refused(clause: str, tamper: str) -> None:
    text = document()
    row = row_text(text, clause)
    if tamper == "missing":
        replacement = ""
    elif tamper == "duplicate":
        replacement = f"{row}\n{row}"
    elif tamper == "unknown":
        replacement = row.replace(f"| {clause} |", "| A28 |", 1)
    else:
        cells = row.split("|")
        cells[4] = " "
        replacement = "|".join(cells)
    with pytest.raises(ValueError):
        check_index(replace_row(text, clause, replacement))


@pytest.mark.parametrize(
    "reference",
    [
        "tests/test_r15_missing_evidence.py::test_missing",
        "tests/test_r15_u1_scope.py::test_nonexistent_callable",
        "tests/test_r15_u1_scope.py",
        "tests/test_r15_u1_scope.py::test_r15_u1_pure_modules_are_not_runtime_producers::extra",
        "tests/../tests/test_r15_u1_scope.py::test_r15_u1_pure_modules_are_not_runtime_producers",
        "docs/no_such_document.md",
    ],
)
def test_invalid_local_reference_is_refused(reference: str) -> None:
    with pytest.raises(ValueError):
        check_reference(reference, ROOT)


def test_invalid_reference_in_matrix_is_refused() -> None:
    text = document()
    row = row_text(text, "A1")
    reference = TEST_REF.search(row)
    assert reference is not None
    tampered = row.replace(
        reference.group(), "tests/test_r15_u1_scope.py::test_missing"
    )
    with pytest.raises(ValueError, match="missing or nonunique"):
        check_index(replace_row(text, "A1", tampered))


def test_dangling_reference_outside_matrix_is_refused() -> None:
    with pytest.raises(ValueError, match="missing local reference"):
        check_index(document() + "\n`docs/no_such_document.md`\n")


@pytest.mark.parametrize("clause", ["A3", "F16", "A8", "F6", "A19", "A26", "A27"])
def test_false_positive_classification_is_refused(clause: str) -> None:
    text = document()
    row = row_text(text, clause)
    cells = row.split("|")
    cells[3] = " MVI_WITNESSED "
    with pytest.raises(ValueError):
        check_index(replace_row(text, clause, "|".join(cells)))


def test_uncontrolled_pass_classification_is_refused() -> None:
    text = document()
    row = row_text(text, "A1").replace("MVI_WITNESSED", "PASS")
    with pytest.raises(ValueError, match="invalid classification"):
        check_index(replace_row(text, "A1", row))


def test_a19_whole_row_human_deferral_is_refused() -> None:
    text = document()
    row = row_text(text, "A19").replace("PARTIAL_OR_UNVERIFIED", "HUMAN_DEFERRED")
    with pytest.raises(ValueError, match="uncontrolled Human deferral"):
        check_index(replace_row(text, "A19", row))


@pytest.mark.parametrize("section", HEADINGS)
def test_missing_crosscutting_section_is_refused(section: str) -> None:
    text = document().replace(f"### {section} —", f"## {section} —", 1)
    with pytest.raises(ValueError, match="missing crosscutting"):
        check_index(text)


@pytest.mark.parametrize(
    "phrase",
    ["unsupported monetary claims", "increased with no E2", "external_directory DENY"],
)
def test_u7_semantic_caveat_removal_is_refused(phrase: str) -> None:
    with pytest.raises(ValueError, match="missing U7 caveat"):
        check_index(document().replace(phrase, "omitted evidence", 1))


@pytest.mark.parametrize(
    "source",
    [
        "def test_example(): pass\ndef test_example(): pass\n",
        "def helper():\n    def test_example(): pass\n",
        "class Container:\n    def test_example(self): pass\n",
        "async def test_example(): pass\n",
    ],
)
def test_ambiguous_or_noncollectable_ast_callable_is_refused(source: str) -> None:
    with pytest.raises(ValueError, match="missing or nonunique"):
        check_callable(source, "test_example", "in-memory negative fixture")
