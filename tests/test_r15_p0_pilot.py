"""P0 harness fixtures: these do not demonstrate Suzka NarrativeSelf updates."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.r15_p0.pilot import (
    Case,
    Choice,
    Relation,
    atomic_private_create,
    blinded_e_packet,
    expected_relation,
    pilot_cases,
    read_sealed,
    run_case,
    score_capture,
    seal_case,
)


class FixtureTransport:
    """Deterministic visible-response fixture, NOT an R15 semantic producer."""

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.calls: list[str] = []

    def __call__(self, message: str) -> dict[str, str]:
        index = len(self.calls)
        self.calls.append(message)
        return {"episode_id": f"episode-fixture-{index}", "response": self.replies[index]}


def annotation(case_id: str, choice: str, prior: str, target: str,
               relation: str, retrieval: str = "UNKNOWN") -> dict[str, str]:
    return {
        "case_id": case_id,
        "choice_at_E": choice,
        "probe_prior": prior,
        "probe_target": target,
        "probe_relation": relation,
        "retrieval": retrieval,
        "annotator": "fixture-blinded-observer",
    }


def fixture(case: Case, choice: Choice, *, probe_prior: Choice | None = None,
            probe_target: Choice | None = None,
            probe_relation: Relation | None = None):
    temp = TemporaryDirectory(prefix="suzka-p0-fixture-")
    root = Path(temp.name)
    manifest_path = root / "manifest.json"
    out_path = root / "capture.json"
    manifest = seal_case(case, manifest_path)
    expected = expected_relation(
        choice, case.target, revealed=case.control != "no_reveal"
    )
    correct_prior = choice if choice in (Choice.A, Choice.B, Choice.CLARIFY) else Choice.UNKNOWN
    correct_target = Choice(case.target) if case.control != "no_reveal" else Choice.UNKNOWN
    ann = annotation(
        case.case_id, choice.value,
        (probe_prior or correct_prior).value,
        (probe_target or correct_target).value,
        (probe_relation or expected).value,
    )
    transport = FixtureTransport(["設問を受領", "元の応答", "明確化", "検討結果"])
    if case.control == "no_reveal":
        transport = FixtureTransport(["設問を受領", "元の応答", "検討結果"])
    capture = run_case(manifest_path, out_path, transport)
    return temp, manifest, capture, ann, transport


class P0FixtureTests(unittest.TestCase):
    def test_four_base_observable_classes(self) -> None:
        for choice, relation in [
            (Choice.A, Relation.MISMATCH), (Choice.B, Relation.MATCH),
            (Choice.CLARIFY, Relation.NO_CHOICE), (Choice.UNKNOWN, Relation.UNKNOWN),
        ]:
            with self.subTest(choice=choice):
                temp, manifest, capture, ann, _ = fixture(
                    Case("case-4-class", "B", "natural", 0), choice
                )
                with temp:
                    result = score_capture(manifest, capture, ann)
                    self.assertEqual(result["expected_relation"], relation.value)
                    self.assertTrue(result["probe_correct"])
                    self.assertEqual(result["authority_scope"], "P0_BENCHMARK_ONLY_NOT_R15")

    def test_all_twelve_pilot_cases_are_distinct_and_balanced(self) -> None:
        cases = pilot_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case.case_id for case in cases}), 12)
        self.assertEqual(sum(case.control == "normal" for case in cases), 8)
        standard = cases[:8]
        self.assertEqual(sum(case.target == "A" for case in standard), 4)
        self.assertEqual(sum(case.lane == "natural" for case in standard), 4)
        self.assertEqual(sum(case.lane == "forced" for case in standard), 4)
        self.assertEqual(sum(case.control != "normal" for case in cases), 4)
        for variant in range(4):
            self.assertNotEqual(standard[variant].target, standard[variant + 4].target)

    def test_target_is_not_indicated_before_reveal(self) -> None:
        case = Case("no-leak", "B", "natural", 1)
        temp, manifest, capture, ann, transport = fixture(case, Choice.CLARIFY)
        with temp:
            self.assertEqual(len(transport.calls), 4)
            self.assertNotIn("事前に指定", transport.calls[0])
            self.assertNotIn("事前に指定", transport.calls[1])
            self.assertIn("案Bです", transport.calls[2])
            self.assertTrue(score_capture(manifest, capture, ann)["probe_correct"])

    def test_blinded_packet_excludes_reveal_and_case_target(self) -> None:
        case = Case("blind-out", "B", "natural", 0)
        temp, manifest, capture, _ann, _transport = fixture(case, Choice.A)
        with temp:
            packet = blinded_e_packet(manifest, capture)
            payload = json.dumps(packet, ensure_ascii=False)
            self.assertEqual(packet["visible_answer_E"], "元の応答")
            self.assertNotIn("case_id", packet)
            self.assertNotIn("sealed_target", packet)
            self.assertNotIn("今回の比較実験で事前に", payload)
            self.assertNotIn("元の回答との一致", payload)
            self.assertNotIn("blind-out", payload)

    def test_explicit_control_not_blind_to_target(self) -> None:
        case = Case("explicit-cont", "B", "natural", 1, "explicit")
        temp, manifest, capture, _ann, _transport = fixture(case, Choice.B)
        with temp:
            with self.assertRaisesRegex(ValueError, "expose the target"):
                blinded_e_packet(manifest, capture)

    def test_seal_is_immutable_and_changed_target_fails(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            seal_case(Case("seal-test", "B", "natural", 0), path)
            with self.assertRaises(FileExistsError):
                seal_case(Case("seal-test", "A", "natural", 0), path)
            data = json.loads(path.read_text())
            data["case"]["target"] = "A"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "seal mismatch"):
                read_sealed(path)

    def test_cross_case_fails_closed(self) -> None:
        temp, manifest, capture, ann, _ = fixture(Case("case-one", "B", "natural", 0), Choice.A)
        with temp:
            bad = {**capture, "case_id": "case-two"}
            with self.assertRaisesRegex(ValueError, "cross-case"):
                score_capture(manifest, bad, ann)

    def test_sealed_hash_tamper_fails_closed(self) -> None:
        temp, manifest, capture, ann, _ = fixture(Case("wrong-seal", "B", "natural", 0), Choice.A)
        with temp:
            bad = {**capture, "manifest_seal": "a" * 64}
            with self.assertRaisesRegex(ValueError, "changed-manifest"):
                score_capture(manifest, bad, ann)

    def test_missing_and_duplicated_reveal_fail(self) -> None:
        temp, manifest, capture, ann, _ = fixture(Case("missing-reveal", "B", "natural", 0), Choice.A)
        with temp:
            missing = {**capture, "turns": [x for x in capture["turns"] if x["step"] != "reveal"]}
            with self.assertRaisesRegex(ValueError, "missing, duplicated"):
                score_capture(manifest, missing, ann)
            duplicated = {**capture, "turns": [
                *capture["turns"][:3], capture["turns"][2], capture["turns"][3],
            ]}
            with self.assertRaisesRegex(ValueError, "missing, duplicated"):
                score_capture(manifest, duplicated, ann)

    def test_wrong_reveal_and_modified_original_response_metadata_fail(self) -> None:
        temp, manifest, capture, ann, _ = fixture(Case("bad-reveal", "B", "natural", 0), Choice.A)
        with temp:
            turns = [dict(x) for x in capture["turns"]]
            turns[2]["request"] = "今回の参照先は案Aでした"
            with self.assertRaisesRegex(ValueError, "does not match"):
                score_capture(manifest, {**capture, "turns": turns}, ann)
            turns = [dict(x) for x in capture["turns"]]
            turns[1]["episode_id"] = turns[0]["episode_id"]
            with self.assertRaisesRegex(ValueError, "duplicate episode"):
                score_capture(manifest, {**capture, "turns": turns}, ann)

    def test_reordered_events_fail(self) -> None:
        temp, manifest, capture, ann, _ = fixture(Case("reorder", "B", "natural", 0), Choice.A)
        with temp:
            reordered = {**capture, "turns": list(reversed(capture["turns"]))}
            with self.assertRaisesRegex(ValueError, "reordered"):
                score_capture(manifest, reordered, ann)

    def test_wrong_probe_claim_is_not_silent_success(self) -> None:
        temp, manifest, capture, ann, _ = fixture(
            Case("false-mismatch", "B", "natural", 0), Choice.CLARIFY,
            probe_relation=Relation.MISMATCH,
        )
        with temp:
            result = score_capture(manifest, capture, ann)
            self.assertEqual(result["expected_relation"], "NO_CHOICE")
            self.assertFalse(result["relation_correct"])
            self.assertFalse(result["probe_correct"])

    def test_no_reveal_requires_unknown_target_and_relation(self) -> None:
        temp, manifest, capture, ann, transport = fixture(
            Case("no-reveal", "B", "forced", 0, "no_reveal"), Choice.A
        )
        with temp:
            self.assertEqual(len(transport.calls), 3)
            result = score_capture(manifest, capture, ann)
            self.assertTrue(result["probe_correct"])
            self.assertEqual(result["expected_relation"], "UNKNOWN")
            bad = {**ann, "probe_target": "B", "probe_relation": "MISMATCH"}
            self.assertFalse(score_capture(manifest, capture, bad)["probe_correct"])

    def test_unqualified_choice_remains_unknown_even_if_target_revealed(self) -> None:
        temp, manifest, capture, ann, _ = fixture(Case("other", "B", "natural", 0), Choice.OTHER)
        with temp:
            result = score_capture(manifest, capture, ann)
            self.assertEqual(result["expected_relation"], "UNKNOWN")
            self.assertTrue(result["probe_correct"])

    def test_invalid_annotations_fail_closed(self) -> None:
        temp, manifest, capture, ann, _ = fixture(Case("bad-annotation", "B", "natural", 0), Choice.A)
        with temp:
            with self.assertRaisesRegex(ValueError, "exactly"):
                score_capture(manifest, capture, {**ann, "claimed_subject_truth": True})
            with self.assertRaisesRegex(ValueError, "invalid"):
                score_capture(manifest, capture, {**ann, "choice_at_E": "BOTH"})
            with self.assertRaisesRegex(ValueError, "identity"):
                score_capture(manifest, capture, {**ann, "case_id": "different"})

    def test_different_trial_cannot_overwrite_or_replay_capture(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            case = Case("immutable-capture", "B", "forced", 0)
            manifest_path, capture_path = root / "m.json", root / "run.json"
            seal_case(case, manifest_path)
            transport = FixtureTransport(["init", "A", "B", "done"])
            run_case(manifest_path, capture_path, transport)
            with self.assertRaisesRegex(FileExistsError, "trial output exists or was started"):
                run_case(manifest_path, capture_path, transport)
            self.assertEqual(len(transport.calls), 4)

    def test_snapshot_no_production_authority_and_no_contaminating_imports(self) -> None:
        import experiments.r15_p0.pilot as pilot
        self.assertFalse(hasattr(pilot, "NarrativeSelfSystem"))
        self.assertFalse(hasattr(pilot, "BeliefSystem"))
        self.assertFalse(hasattr(pilot, "SourceWitness"))

    def test_private_create_is_exclusive(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "private.json"
            atomic_private_create(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                atomic_private_create(path, {"x": 2})
            self.assertEqual(json.loads(path.read_text()), {"x": 1})

    def test_incomplete_trial_reservation_blocks_unsafe_retry(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path, capture_path = root / "m.json", root / "run.json"
            seal_case(Case("partial", "B", "natural", 0), manifest_path)
            calls: list[str] = []

            def fail_second(message: str) -> dict[str, str]:
                calls.append(message)
                if len(calls) == 2:
                    raise RuntimeError("test network interruption")
                return {"episode_id": "fixture-first", "response": "setup"}

            with self.assertRaisesRegex(RuntimeError, "network interruption"):
                run_case(manifest_path, capture_path, fail_second)
            self.assertTrue((root / "run.json.started.json").exists())
            self.assertFalse(capture_path.exists())
            with self.assertRaises(FileExistsError):
                run_case(manifest_path, capture_path, fail_second)
            self.assertEqual(len(calls), 2)

    def test_http_client_rejects_nonlocal_url_without_transport(self) -> None:
        from experiments.r15_p0.pilot import http_chat

        for base_url in (
            "https://127.0.0.1:8000",
            "http://example.com:8000",
            "http://localhost:8000@evil.example",
            "http://localhost:8000/path",
            "http://localhost:8000?redirect=elsewhere",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaisesRegex(ValueError, "localhost origin"):
                    http_chat(base_url, "synthetic input")

    def test_malformed_capture_fail_closed(self) -> None:
        temp, manifest, _capture, ann, _ = fixture(Case("bad-capture", "A", "natural", 1), Choice.A)
        with temp:
            with self.assertRaisesRegex(ValueError, "JSON object"):
                score_capture(manifest, [], ann)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
