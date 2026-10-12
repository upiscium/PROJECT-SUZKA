"""G2-R0 MOCK-ONLY four-owner epoch invariant tests.

This is not a real R07/B/DB1/R12 source graph, approved producer or G2 issuer.
Its versions and lock are simulated; positive results cannot be treated as
source attestation or R15 subjective admission. See Issue #294.
"""

from __future__ import annotations
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock, Thread
import json
import unittest


def digest(v: object) -> str:
    """Synthetic stable digest, NOT a real R07/DB1/R12 operation digest."""
    return sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Version:
    incarnation: str
    generation: int


class FixtureOwner:
    def __init__(self, label: str, value: dict) -> None:
        self.label = label
        self.value = deepcopy(value)
        self._incarnation_counter = 1
        self.version = Version(f"synthetic-incarnation-{self._incarnation_counter}", 1)

    def replace(self, value: dict) -> None:
        self.value = deepcopy(value)
        self.version = Version(self.version.incarnation, self.version.generation + 1)

    def restart_as_new_incarnation(self) -> None:
        # Each simulated restart/rollback rekeys the owner identity; a token
        # from incarnation 2 MUST NEVER recur at incarnation 3.
        self._incarnation_counter += 1
        self.version = Version(f"synthetic-incarnation-{self._incarnation_counter}", 1)


@dataclass(frozen=True)
class Candidate:
    versions: dict[str, Version]
    fingerprint: str


class SyntheticLedger:
    NAMES = ("r07", "b", "db1", "r12")
    PLAN = ("context.capture", "memory.episodic", "memory.experience", "session.turn")

    def __init__(self) -> None:
        self.gate = RLock()
        self.admitted = 0
        episode = {
            "episode_id": "ep-synthetic-1",
            "context_id": "context-synthetic-1",
            "user_input": "fixture input",
            "response": "fixture output",
            "source_event_id": "event-synthetic-1",
        }
        ep_digest = digest(episode)
        exp = {
            "experience_id": "exp-synthetic-1",
            "source_event_id": "event-synthetic-1",
            "source_event_sequence": 17,
            "episode_id": episode["episode_id"],
            "context_id": episode["context_id"],
        }
        ex_digest = digest(exp)
        capture = {
            "event_id": "event-synthetic-1",
            "sequence": 17,
            "context_id": episode["context_id"],
            "episode_id": episode["episode_id"],
            "ep_digest": ep_digest,
            "experience_id": exp["experience_id"],
            "experience_digest": ex_digest,
            "source_time": "2026-10-12T05:00:00Z",
        }
        b_digest = digest(capture)
        requirements = {
            "context.capture": b_digest,
            "memory.episodic": ep_digest,
            "memory.experience": ex_digest,
            "session.turn": digest("ephemeral-session-synthetic"),
        }
        data = {
            "r07": {
                "event_id": "event-synthetic-1",
                "sequence": 17,
                "kind": "chat",
                "source": "api.chat",
                "transaction": "completed",
                "event_terminal": True,
                "requirements": requirements,
                "outcomes": dict(requirements),
            },
            "b": {"preimage": capture, "operation_digest": b_digest},
            "db1": {"record": episode, "archived": False},
            "r12": {
                "original": exp,
                "lifecycle": "active",
                "current_revision": 0,
                "required_revisions_have_own_terminal": True,
                "source_episode_operation_digest": ep_digest,
            },
        }
        self.owners = {key: FixtureOwner(key, data[key]) for key in self.NAMES}

    def change(self, who: str, mutate) -> None:
        # Cooperative source-owning writer, behind the synthetic epoch gate.
        with self.gate:
            current = deepcopy(self.owners[who].value)
            mutate(current)
            self.owners[who].replace(current)

    def versions(self) -> dict[str, Version]:
        with self.gate:
            return {key: self.owners[key].version for key in self.NAMES}

    def inspect(self, after_read=None) -> tuple[str, Candidate | None]:
        # Optimistic cross-owner read. Gate briefly guards each read; it does
        # NOT hold a single writer lock across all reads. Version verification
        # is therefore necessary to reject a "Frankenstein" tuple.
        begin = self.versions()
        data = {}
        for owner in self.NAMES:
            with self.gate:
                data[owner] = deepcopy(self.owners[owner].value)
            if after_read is not None:
                after_read(owner, self)
        end = self.versions()
        if begin != end:
            return "STALE_EPOCH", None
        reason = verify_synthetic(data)
        if reason != "SYNTHETIC_CONSISTENT":
            return reason, None
        return "SYNTHETIC_INSPECTED_ONLY", Candidate(begin, digest(data))

    def admit(
        self,
        candidate: Candidate | None,
        *,
        before_publication: Callable[[], None] | None = None,
    ) -> str:
        # NOT an R15 admission; this increments only a toy fixture counter.
        # The key requirement is a held owner gate spanning source version
        # revalidation and toy publication; there is no gap for a writer.
        if candidate is None:
            return "NO_CANDIDATE"
        with self.gate:
            current = {key: self.owners[key].version for key in self.NAMES}
            if current != candidate.versions:
                return "STALE_AT_FIXTURE_ADMISSION"
            source = {key: deepcopy(self.owners[key].value) for key in self.NAMES}
            if verify_synthetic(source) != "SYNTHETIC_CONSISTENT":
                return "SOURCE_REVOKED"
            if digest(source) != candidate.fingerprint:
                return "DIFFERENT_SOURCE_PREIMAGE"
            if before_publication is not None:
                # Test-only deterministic injection point. The owner gate must
                # remain held across this last check and toy publication.
                before_publication()
            self.admitted += 1
            return "TOY_FIXTURE_ACCEPTED_NOT_G2"


def verify_synthetic(data: dict[str, dict]) -> str:
    """Negative-only schema/relationship checks over fake source values.

    It intentionally uses synthetic dict digests, not Suzka canonical encoders,
    verified R07 Journal records, B operation or R12 runtime implementations.
    """
    try:
        j, b, m, x = (data[k] for k in SyntheticLedger.NAMES)
        if j["kind"] != "chat" or j["source"] != "api.chat":
            return "UNTRUSTED_PRODUCER"
        if j["transaction"] != "completed" or j["event_terminal"] is not True:
            return "NOT_EXTERNALLY_TERMINAL"
        rq, oc = j["requirements"], j["outcomes"]
        if set(rq) != set(SyntheticLedger.PLAN) or oc != rq:
            return "INCOMPLETE_FOUR_PARTICIPANT_PLAN"
        cap, episode, exp = b["preimage"], m["record"], x["original"]
        if (
            digest(cap) != b["operation_digest"]
            or rq["context.capture"] != b["operation_digest"]
        ):
            return "BAD_B_PREIMAGE"
        if digest(episode) != rq["memory.episodic"]:
            return "BAD_DB1_OPERATION"
        if digest(exp) != rq["memory.experience"]:
            return "BAD_R12_OPERATION"
        if x["source_episode_operation_digest"] != rq["memory.episodic"]:
            return "BAD_R12_EPISODIC_SOURCE_LINK"
        if (j["event_id"], j["sequence"]) != (cap["event_id"], cap["sequence"]):
            return "WRONG_EVENT_BINDING"
        if (exp["source_event_id"], exp["source_event_sequence"]) != (
            j["event_id"],
            j["sequence"],
        ):
            return "WRONG_R12_EVENT_BINDING"
        if (
            cap["episode_id"],
            cap["ep_digest"],
            cap["experience_id"],
            cap["experience_digest"],
            cap["context_id"],
        ) != (
            episode["episode_id"],
            rq["memory.episodic"],
            exp["experience_id"],
            rq["memory.experience"],
            episode["context_id"],
        ):
            return "CROSS_OWNER_MISMATCH"
        if (exp["episode_id"], exp["context_id"]) != (
            episode["episode_id"],
            episode["context_id"],
        ):
            return "CROSS_OWNER_MISMATCH"
        if m["archived"]:
            return "SOURCE_ARCHIVED"
        if x["lifecycle"] != "active":
            return "SOURCE_RETRACTED"
        if not x["required_revisions_have_own_terminal"]:
            return "REVISION_TERMINALITY_MISSING"
        return "SYNTHETIC_CONSISTENT"
    except (KeyError, TypeError, ValueError):
        return "MALFORMED_SYNTHETIC_SOURCE"


class TestSyntheticG2EpochModel(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = SyntheticLedger()

    def test_01_consistent_four_owner_fixture_only(self) -> None:
        result, candidate = self.ledger.inspect()
        self.assertEqual(result, "SYNTHETIC_INSPECTED_ONLY")
        self.assertEqual(self.ledger.admit(candidate), "TOY_FIXTURE_ACCEPTED_NOT_G2")

    def test_02_missing_b_capture_fails(self) -> None:
        self.ledger.change("b", lambda x: x.pop("preimage"))
        self.assertEqual(self.ledger.inspect()[0], "MALFORMED_SYNTHETIC_SOURCE")

    def test_03_missing_session_turn_terminality_fails(self) -> None:
        self.ledger.change("r07", lambda x: x["outcomes"].pop("session.turn"))
        self.assertEqual(self.ledger.inspect()[0], "INCOMPLETE_FOUR_PARTICIPANT_PLAN")

    def test_04_unfinalized_journal_fails(self) -> None:
        self.ledger.change("r07", lambda x: x.update(transaction="prepared"))
        self.assertEqual(self.ledger.inspect()[0], "NOT_EXTERNALLY_TERMINAL")

    def test_05_wrong_context_binding_fails(self) -> None:
        self.ledger.change("db1", lambda x: x["record"].update(context_id="other-ctx"))
        self.assertEqual(self.ledger.inspect()[0], "BAD_DB1_OPERATION")

    def test_06_bad_original_db1_op_digest_fails(self) -> None:
        # Preserve outcome/requirement equality so the fixture reaches the
        # actual original DB1 operation-digest comparison branch.
        self.ledger.change(
            "r07",
            lambda x: (
                x["requirements"].update({"memory.episodic": "0" * 64}),
                x["outcomes"].update({"memory.episodic": "0" * 64}),
            ),
        )
        self.assertEqual(self.ledger.inspect()[0], "BAD_DB1_OPERATION")

    def test_07_bad_b_context_cross_owner_binding_fails(self) -> None:
        def change_b(b):
            b["preimage"]["context_id"] = "different-context"
            b["operation_digest"] = digest(b["preimage"])

        self.ledger.change("b", change_b)
        new_b_digest = self.ledger.owners["b"].value["operation_digest"]
        self.ledger.change(
            "r07",
            lambda x: (
                x["requirements"].update({"context.capture": new_b_digest}),
                x["outcomes"].update({"context.capture": new_b_digest}),
            ),
        )
        self.assertEqual(self.ledger.inspect()[0], "CROSS_OWNER_MISMATCH")

    def test_07b_r12_original_episode_digest_link_mismatch_fails(self) -> None:
        self.ledger.change(
            "r12", lambda x: x.update(source_episode_operation_digest="0" * 64)
        )
        self.assertEqual(self.ledger.inspect()[0], "BAD_R12_EPISODIC_SOURCE_LINK")

    def test_08_retraction_fails(self) -> None:
        self.ledger.change("r12", lambda x: x.update(lifecycle="retracted"))
        self.assertEqual(self.ledger.inspect()[0], "SOURCE_RETRACTED")

    def test_09_revision_own_terminal_missing_fails(self) -> None:
        self.ledger.change(
            "r12", lambda x: x.update(required_revisions_have_own_terminal=False)
        )
        self.assertEqual(self.ledger.inspect()[0], "REVISION_TERMINALITY_MISSING")

    def test_10_cross_owner_change_during_read_fails(self) -> None:
        def mutate_after_db1(owner, ledger):
            if owner == "db1":
                ledger.change("r12", lambda x: x.update(lifecycle="retracted"))

        self.assertEqual(self.ledger.inspect(mutate_after_db1)[0], "STALE_EPOCH")

    def test_10_aba_same_bytes_during_read_fails_on_generation(self) -> None:
        def revert_after_b(owner, ledger):
            if owner == "b":
                old = deepcopy(ledger.owners["b"].value)
                ledger.change("b", lambda x: x.update(operation_digest="0" * 64))
                ledger.change("b", lambda x: x.update(old))

        self.assertEqual(self.ledger.inspect(revert_after_b)[0], "STALE_EPOCH")

    def test_11_revocation_between_inspect_and_admit_fails(self) -> None:
        result, candidate = self.ledger.inspect()
        self.assertEqual(result, "SYNTHETIC_INSPECTED_ONLY")
        self.ledger.change("r12", lambda x: x.update(lifecycle="retracted"))
        self.assertEqual(self.ledger.admit(candidate), "STALE_AT_FIXTURE_ADMISSION")

    def test_11b_competing_writer_cannot_enter_toy_admission_publication(self) -> None:
        """A distinct thread must fail a nonblocking owner-gate attempt.

        This tests the MOCK gate around version revalidation and publication;
        it does not attest any real R07/B/DB1/R12 source gate or scheduler.
        """
        status, candidate = self.ledger.inspect()
        self.assertEqual(status, "SYNTHETIC_INSPECTED_ONLY")
        probe_results: list[bool] = []

        def concurrent_writer_probe() -> None:
            acquired = self.ledger.gate.acquire(blocking=False)
            probe_results.append(acquired)
            if acquired:
                try:
                    current = deepcopy(self.ledger.owners["r12"].value)
                    current["lifecycle"] = "retracted"
                    self.ledger.owners["r12"].replace(current)
                finally:
                    self.ledger.gate.release()

        def injection_inside_admission() -> None:
            competitor = Thread(target=concurrent_writer_probe)
            competitor.start()
            competitor.join(timeout=3)
            self.assertFalse(competitor.is_alive())
            self.assertEqual(probe_results, [False])

        self.assertEqual(
            self.ledger.admit(candidate, before_publication=injection_inside_admission),
            "TOY_FIXTURE_ACCEPTED_NOT_G2",
        )
        self.assertEqual(self.ledger.admitted, 1)
        self.assertEqual(self.ledger.owners["r12"].value["lifecycle"], "active")

    def test_12_owner_incarnation_restart_before_admit_fails(self) -> None:
        _, candidate = self.ledger.inspect()
        with self.ledger.gate:
            self.ledger.owners["db1"].restart_as_new_incarnation()
        self.assertEqual(self.ledger.admit(candidate), "STALE_AT_FIXTURE_ADMISSION")

    def test_13_source_history_pruned_before_admit_fails(self) -> None:
        _, candidate = self.ledger.inspect()
        self.ledger.change(
            "r12", lambda x: x.update(required_revisions_have_own_terminal=False)
        )
        self.assertEqual(self.ledger.admit(candidate), "STALE_AT_FIXTURE_ADMISSION")

    def test_14_unchanged_content_but_new_owner_generation_fails(self) -> None:
        _, candidate = self.ledger.inspect()
        self.ledger.change("r07", lambda x: None)
        self.assertEqual(self.ledger.admit(candidate), "STALE_AT_FIXTURE_ADMISSION")

    def test_15_invalid_source_label_rejected_in_toy_only(self) -> None:
        # This is ONLY a toy label allowlist check. A caller can also copy
        # "api.chat"; an actual registered-handler issuer does not exist.
        self.ledger.change("r07", lambda x: x.update(source="operator.claims"))
        self.assertEqual(self.ledger.inspect()[0], "UNTRUSTED_PRODUCER")

    def test_16_second_restart_has_distinct_nonreusable_incarnation(self) -> None:
        _, candidate0 = self.ledger.inspect()
        with self.ledger.gate:
            owner = self.ledger.owners["db1"]
            owner.restart_as_new_incarnation()
            first = owner.version
        _, candidate1 = self.ledger.inspect()
        with self.ledger.gate:
            owner.restart_as_new_incarnation()
            second = owner.version
        self.assertNotEqual(first.incarnation, second.incarnation)
        self.assertEqual(self.ledger.admit(candidate0), "STALE_AT_FIXTURE_ADMISSION")
        self.assertEqual(self.ledger.admit(candidate1), "STALE_AT_FIXTURE_ADMISSION")

    def test_17_unsupported_fifth_participant_fails(self) -> None:
        self.ledger.change(
            "r07",
            lambda x: (
                x["requirements"].update(extra="f" * 64),
                x["outcomes"].update(extra="f" * 64),
            ),
        )
        self.assertEqual(self.ledger.inspect()[0], "INCOMPLETE_FOUR_PARTICIPANT_PLAN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
