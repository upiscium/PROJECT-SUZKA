"""R17 IP-12: pure forecast, outcome-assessment and feedback values (#367, D-R17 D1).

Owner decision: #367 comment 6888292077. Nothing here issues, persists or
admits a record, reads a store, or authorizes an Action. Validation rejects
internally inconsistent values only; whether the inputs were current or the
caller was the trusted R17 owner is the issuer's job (IP-16 / IP-19).

V1 limits (D-R17 §1, §6):

- The forecast is a code-derived nominal prediction about one fixed forward
  operation, not a model prediction. Its only kind is
  ``EXPECT_ACTIVE_WITH_PAYLOAD`` and its uncertainty is ``NOT_ASSESSED``.
- For an authorized effect the comparison is MATCH by definition (the
  verifier's authorization already requires the recomputed digest to equal
  the approved payload digest), so a MATCH_SELF Feedback is not independent
  evidence that the prediction was right.
- Values built in tests are fixtures, never evidence of real-world learning.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Final, Self

from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE, MAX_PERSISTED_REVISION
from suzka.r16.contracts import (
    canonical_json,
    checksum,
    closed,
    digest,
    identifier,
    integer,
    parse_enum,
    parse_json,
    sequence,
)
from suzka.r16.history_codec import (
    PolicyKind,
    PolicyRow,
    canonical_uuid,
    decision_id,
    event_sequence,
)
from suzka.r16.successor_policy import FeedbackState, LedgerState, PolicyRef

SCHEMA_VERSION: Final = 1
# D-R17 §2 / §3 / §4: every canonical record fits in 4 KiB.
MAX_RECORD_BYTES: Final = 4096
FORECAST_ID_PREFIX: Final = "forecast-"
ASSESSMENT_ID_PREFIX: Final = "assessment-"
FEEDBACK_ID_PREFIX: Final = "feedback-"
# SuccessorPolicyInput reference kinds (D-R17 §5).
FEEDBACK_REF_KIND: Final = "r17-feedback"
# V1 issues only the first Feedback revision; supersession is future work.
V1_FEEDBACK_REVISION: Final = 1

_FORECAST_ID_DOMAIN: Final = b"PROJECT-SUZKA:R17:FORECAST-ID:V1\0"
_FORECAST_DOMAIN: Final = b"PROJECT-SUZKA:R17:FORECAST:V1\0"
_ASSESSMENT_ID_DOMAIN: Final = b"PROJECT-SUZKA:R17:ASSESSMENT-ID:V1\0"
_ASSESSMENT_DOMAIN: Final = b"PROJECT-SUZKA:R17:ASSESSMENT:V1\0"
_FEEDBACK_ID_DOMAIN: Final = b"PROJECT-SUZKA:R17:FEEDBACK-ID:V1\0"
_FEEDBACK_DOMAIN: Final = b"PROJECT-SUZKA:R17:FEEDBACK:V1\0"


class PredictionKind(str, Enum):
    EXPECT_ACTIVE_WITH_PAYLOAD = "EXPECT_ACTIVE_WITH_PAYLOAD"


class Uncertainty(str, Enum):
    # The code does not assess confidence; no value is a probability.
    NOT_ASSESSED = "NOT_ASSESSED"


class TargetState(str, Enum):
    ABSENT = "ABSENT"
    ACTIVE = "ACTIVE"


class ObservedState(str, Enum):
    ABSENT = "ABSENT"
    ACTIVE = "ACTIVE"
    UNKNOWN = "UNKNOWN"


class OutcomeState(str, Enum):
    """The three terminal outcomes of the independent R18 verifier."""

    ACTIVE_VERIFIED_AUTHORIZED = "ACTIVE_VERIFIED_AUTHORIZED"
    VERIFIED_ABSENT = "VERIFIED_ABSENT"
    UNAUTHORIZED_EFFECT = "UNAUTHORIZED_EFFECT"


class Comparison(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"


class Contributor(str, Enum):
    SELF = "SELF"
    # Kept in the enum; V1 never emits it (no qualified other-agent witness).
    OTHER = "OTHER"
    ENVIRONMENT = "ENVIRONMENT"
    UNKNOWN = "UNKNOWN"


# Ledger states in which a positive Feedback can legitimately exist: the
# authorized effect itself, or that effect followed by an inverse (D-R17 §5
# as-of rule). Any other ledger with a Feedback is a contradiction.
_FEEDBACK_LEDGER: Final = frozenset(
    {
        LedgerState.ACTIVE_VERIFIED_AUTHORIZED,
        LedgerState.INVERSE_INTENT_ISSUED,
        LedgerState.INVERSE_CLAIMED,
        LedgerState.INVERSE_IN_DOUBT,
        LedgerState.INVERSE_POLICY_DENIED,
        LedgerState.INVERSE_APPROVAL_EXPIRED_UNCLAIMED,
        LedgerState.INVERSE_VERIFIED_ABSENT,
        LedgerState.REVERTED,
    }
)
# Ledger states before any claim: a missing forecast is not yet a defect.
_PRE_CLAIM_LEDGER: Final = frozenset(
    {
        LedgerState.NONE,
        LedgerState.INTENT_ISSUED,
        LedgerState.POLICY_DENIED,
        LedgerState.APPROVAL_EXPIRED_UNCLAIMED,
    }
)


def _event(event_id: object, sequence_value: object, name: str) -> None:
    canonical_uuid(event_id, f"{name}_event_id")
    event_sequence(sequence_value, f"{name}_sequence")


def _revision(value: object, name: str, minimum: int = 0) -> int:
    return integer(value, name, minimum=minimum, maximum=MAX_PERSISTED_REVISION)


def _prefixed(value: object, prefix: str, name: str) -> str:
    if (
        type(value) is not str
        or not value.startswith(prefix)
        or len(value) != len(prefix) + 64
    ):
        raise ValueError(f"{name} must be a {prefix}<sha256> identity")
    digest(value[len(prefix) :], name)
    return value


def _bounded(payload: bytes, name: str) -> bytes:
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError(f"{name} exceeds its {MAX_RECORD_BYTES}-byte bound")
    return payload


def _str(value: object) -> str:
    if type(value) is not str:
        raise TypeError("expected a string")
    return value


def _int(value: object) -> int:
    if type(value) is not int:
        raise TypeError("expected an integer")
    return value


# ------------------------------------------------------------------ forecast

_FORECAST_BODY: Final = (
    "action_intent_digest",
    "action_intent_id",
    "admitting_event_id",
    "admitting_sequence",
    "as_of_r07_sequence",
    "decision_committed_digest",
    "decision_id",
    "forward_key",
    "incarnation_id",
    "lineage_scope",
    "obligation_group_key",
    "policy_kind",
    "policy_row",
    "predicted_payload_digest",
    "predicted_state",
    "prediction_kind",
    "prestate",
    "schema_version",
    "selected_candidate_digest",
    "selected_candidate_id",
    "target_expected_revision",
    "target_instance_id",
    "target_namespace",
    "uncertainty",
)


@dataclass(frozen=True, slots=True)
class ExpectedObservationV1:
    """One frozen pre-effect forecast for one terminal forward ActionIntent.

    Same-event WAL, terminal and R07 transaction identity are not part of the
    body; they belong to the postcommit receipt (#367 D1).
    """

    lineage_scope: str
    incarnation_id: str
    admitting_event_id: str
    admitting_sequence: int
    as_of_r07_sequence: int
    decision_id: str
    decision_committed_digest: str
    policy_kind: PolicyKind
    selected_candidate_id: str
    selected_candidate_digest: str
    action_intent_id: str
    action_intent_digest: str
    obligation_group_key: str
    forward_key: str
    target_instance_id: str
    target_namespace: str
    target_expected_revision: int
    predicted_payload_digest: str

    def __post_init__(self) -> None:
        identifier(self.lineage_scope, "lineage_scope")
        digest(self.incarnation_id, "incarnation_id")
        _event(self.admitting_event_id, self.admitting_sequence, "admitting")
        integer(
            self.as_of_r07_sequence,
            "as_of_r07_sequence",
            maximum=MAX_PERSISTED_EVENT_SEQUENCE,
        )
        if self.as_of_r07_sequence >= self.admitting_sequence:
            raise ValueError("the forecast's source horizon must precede its event")
        decision_id(self.decision_id, "decision_id")
        digest(self.decision_committed_digest, "decision_committed_digest")
        if type(self.policy_kind) is not PolicyKind:
            raise TypeError("policy_kind must be a PolicyKind")
        identifier(self.selected_candidate_id, "selected_candidate_id")
        digest(self.selected_candidate_digest, "selected_candidate_digest")
        identifier(self.action_intent_id, "action_intent_id")
        digest(self.action_intent_digest, "action_intent_digest")
        digest(self.obligation_group_key, "obligation_group_key")
        digest(self.forward_key, "forward_key")
        canonical_uuid(self.target_instance_id, "target_instance_id")
        identifier(self.target_namespace, "target_namespace")
        _revision(self.target_expected_revision, "target_expected_revision")
        digest(self.predicted_payload_digest, "predicted_payload_digest")
        _bounded(self.canonical_bytes(), "forecast")

    def check_binding(
        self,
        *,
        decision_id: str,
        decision_committed_digest: str,
        policy_kind: PolicyKind,
        action_intent_id: str,
        action_intent_digest: str,
        forward_key: str,
        target_instance_id: str,
        target_namespace: str,
        target_expected_revision: int,
    ) -> None:
        """D-R17 §2: refuse a forecast not bound to this Decision, I and target.

        The issuer (IP-16) passes values it read from the terminal Decision and
        ActionIntent; this check only compares them.
        """

        expected = {
            "decision_id": decision_id,
            "decision_committed_digest": decision_committed_digest,
            "policy_kind": policy_kind,
            "action_intent_id": action_intent_id,
            "action_intent_digest": action_intent_digest,
            "forward_key": forward_key,
            "target_instance_id": target_instance_id,
            "target_namespace": target_namespace,
            "target_expected_revision": target_expected_revision,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"forecast {name} does not match its source")

    def _body(self) -> dict[str, object]:
        return {
            "action_intent_digest": self.action_intent_digest,
            "action_intent_id": self.action_intent_id,
            "admitting_event_id": self.admitting_event_id,
            "admitting_sequence": self.admitting_sequence,
            "as_of_r07_sequence": self.as_of_r07_sequence,
            "decision_committed_digest": self.decision_committed_digest,
            "decision_id": self.decision_id,
            "forward_key": self.forward_key,
            "incarnation_id": self.incarnation_id,
            "lineage_scope": self.lineage_scope,
            "obligation_group_key": self.obligation_group_key,
            "policy_kind": self.policy_kind.value,
            # Only priority row 4 (Plan B) may lead to a forward effect.
            "policy_row": PolicyRow.ROW_4.value,
            "predicted_payload_digest": self.predicted_payload_digest,
            "predicted_state": TargetState.ACTIVE.value,
            "prediction_kind": PredictionKind.EXPECT_ACTIVE_WITH_PAYLOAD.value,
            "prestate": TargetState.ABSENT.value,
            "schema_version": SCHEMA_VERSION,
            "selected_candidate_digest": self.selected_candidate_digest,
            "selected_candidate_id": self.selected_candidate_id,
            "target_expected_revision": self.target_expected_revision,
            "target_instance_id": self.target_instance_id,
            "target_namespace": self.target_namespace,
            "uncertainty": Uncertainty.NOT_ASSESSED.value,
        }

    @property
    def forecast_id(self) -> str:
        return FORECAST_ID_PREFIX + checksum(_FORECAST_ID_DOMAIN, self._body())

    @property
    def forecast_digest(self) -> str:
        """The ``expectation_digest`` the R18 approval challenge binds."""
        return checksum(_FORECAST_DOMAIN, self._body())

    def canonical_value(self) -> dict[str, object]:
        return {
            **self._body(),
            "forecast_digest": self.forecast_digest,
            "forecast_id": self.forecast_id,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = closed(
            parse_json(_bounded(payload, "forecast"), MAX_RECORD_BYTES),
            (*_FORECAST_BODY, "forecast_digest", "forecast_id"),
            "forecast",
        )
        fixed = {
            "policy_row": PolicyRow.ROW_4.value,
            "predicted_state": TargetState.ACTIVE.value,
            "prediction_kind": PredictionKind.EXPECT_ACTIVE_WITH_PAYLOAD.value,
            "prestate": TargetState.ABSENT.value,
            "schema_version": SCHEMA_VERSION,
            "uncertainty": Uncertainty.NOT_ASSESSED.value,
        }
        for key, expected in fixed.items():
            if row[key] != expected:
                raise ValueError(f"forecast {key} is not a V1 value")
        value = cls(
            lineage_scope=_str(row["lineage_scope"]),
            incarnation_id=_str(row["incarnation_id"]),
            admitting_event_id=_str(row["admitting_event_id"]),
            admitting_sequence=_int(row["admitting_sequence"]),
            as_of_r07_sequence=_int(row["as_of_r07_sequence"]),
            decision_id=_str(row["decision_id"]),
            decision_committed_digest=_str(row["decision_committed_digest"]),
            policy_kind=parse_enum(row["policy_kind"], PolicyKind, "policy_kind"),
            selected_candidate_id=_str(row["selected_candidate_id"]),
            selected_candidate_digest=_str(row["selected_candidate_digest"]),
            action_intent_id=_str(row["action_intent_id"]),
            action_intent_digest=_str(row["action_intent_digest"]),
            obligation_group_key=_str(row["obligation_group_key"]),
            forward_key=_str(row["forward_key"]),
            target_instance_id=_str(row["target_instance_id"]),
            target_namespace=_str(row["target_namespace"]),
            target_expected_revision=_int(row["target_expected_revision"]),
            predicted_payload_digest=_str(row["predicted_payload_digest"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("forecast bytes are not canonical")
        return value


# ------------------------------------------------------- verifier's outcome


@dataclass(frozen=True, slots=True)
class VerifiedOutcomeView:
    """The witness fields R17 needs from one terminal R18 Outcome (IP-18).

    R17 reads these through the Outcome digest; it never reads R18's nonce
    records directly. ``foreign_change`` means the verifier saw a target change
    that is not in the ledger, in the revision range (expected, observed].
    """

    verified_outcome_id: str
    outcome_revision: int
    outcome_digest: str
    outcome_state: OutcomeState
    target_instance_id: str
    observed_state: ObservedState
    observed_payload_digest: str | None
    observed_revision: int
    effect_operation_key: str | None
    foreign_change: bool

    def __post_init__(self) -> None:
        identifier(self.verified_outcome_id, "verified_outcome_id")
        _revision(self.outcome_revision, "outcome_revision", minimum=1)
        digest(self.outcome_digest, "outcome_digest")
        if type(self.outcome_state) is not OutcomeState:
            raise TypeError("outcome_state must be an OutcomeState")
        canonical_uuid(self.target_instance_id, "target_instance_id")
        if type(self.observed_state) is not ObservedState:
            raise TypeError("observed_state must be an ObservedState")
        if (self.observed_payload_digest is None) != (
            self.observed_state is not ObservedState.ACTIVE
        ):
            raise ValueError("an observed payload digest exists only for ACTIVE")
        if self.observed_payload_digest is not None:
            digest(self.observed_payload_digest, "observed_payload_digest")
        _revision(self.observed_revision, "observed_revision")
        if self.effect_operation_key is not None:
            digest(self.effect_operation_key, "effect_operation_key")
        if type(self.foreign_change) is not bool:
            raise TypeError("foreign_change must be a bool")
        if (
            self.outcome_state is OutcomeState.ACTIVE_VERIFIED_AUTHORIZED
            and (
                self.observed_state is not ObservedState.ACTIVE
                or self.effect_operation_key is None
            )
        ):
            raise ValueError("an authorized effect is an observed ACTIVE effect row")
        if (
            self.outcome_state is OutcomeState.VERIFIED_ABSENT
            and self.observed_state is not ObservedState.ABSENT
        ):
            raise ValueError("VERIFIED_ABSENT observes ABSENT")


def assess(
    forecast: ExpectedObservationV1, outcome: VerifiedOutcomeView
) -> tuple[Comparison, tuple[Contributor, ...]]:
    """D-R17 §3: the comparison and the single V1 contributor."""

    if type(forecast) is not ExpectedObservationV1:
        raise TypeError("forecast must be an ExpectedObservationV1")
    if type(outcome) is not VerifiedOutcomeView:
        raise TypeError("outcome must be a VerifiedOutcomeView")
    if outcome.target_instance_id != forecast.target_instance_id:
        raise ValueError("the outcome observed a different target")
    state = outcome.outcome_state
    if state is OutcomeState.ACTIVE_VERIFIED_AUTHORIZED:
        if outcome.observed_payload_digest != forecast.predicted_payload_digest:
            raise ValueError("an authorized effect must carry the approved payload")
        comparison = Comparison.MATCH
    elif state is OutcomeState.VERIFIED_ABSENT:
        comparison = Comparison.MISMATCH
    elif outcome.observed_state is ObservedState.UNKNOWN:
        comparison = Comparison.UNKNOWN
    elif outcome.observed_payload_digest == forecast.predicted_payload_digest:
        comparison = Comparison.MATCH
    else:
        comparison = Comparison.MISMATCH

    if outcome.foreign_change:
        contributor = Contributor.ENVIRONMENT
    elif (
        state is OutcomeState.ACTIVE_VERIFIED_AUTHORIZED
        and outcome.effect_operation_key == forecast.forward_key
    ):
        contributor = Contributor.SELF
    else:
        contributor = Contributor.UNKNOWN
    return comparison, (contributor,)


# --------------------------------------------------------------- assessment

_ASSESSMENT_BODY: Final = (
    "admitting_event_id",
    "admitting_sequence",
    "comparison",
    "contributors",
    "forecast_digest",
    "forecast_id",
    "outcome_digest",
    "outcome_revision",
    "outcome_state",
    "schema_version",
    "verified_outcome_id",
)


@dataclass(frozen=True, slots=True)
class OutcomeAssessmentV1:
    """Comparison and attribution for one terminal forward Outcome (D-R17 §3).

    Recorded for every terminal Outcome; a VERIFIED_ABSENT or
    UNAUTHORIZED_EFFECT assessment is R17's terminal verdict that the Outcome
    is not eligible for positive Feedback.
    """

    forecast_id: str
    forecast_digest: str
    verified_outcome_id: str
    outcome_revision: int
    outcome_digest: str
    outcome_state: OutcomeState
    comparison: Comparison
    contributors: tuple[Contributor, ...]
    admitting_event_id: str
    admitting_sequence: int

    def __post_init__(self) -> None:
        _prefixed(self.forecast_id, FORECAST_ID_PREFIX, "forecast_id")
        digest(self.forecast_digest, "forecast_digest")
        identifier(self.verified_outcome_id, "verified_outcome_id")
        _revision(self.outcome_revision, "outcome_revision", minimum=1)
        digest(self.outcome_digest, "outcome_digest")
        if type(self.outcome_state) is not OutcomeState:
            raise TypeError("outcome_state must be an OutcomeState")
        if type(self.comparison) is not Comparison:
            raise TypeError("comparison must be a Comparison")
        sequence(self.contributors, "contributors", 1, json_input=False)
        if len(self.contributors) != 1 or type(self.contributors[0]) is not Contributor:
            raise ValueError("a V1 assessment has exactly one contributor")
        contributor = self.contributors[0]
        if contributor is Contributor.OTHER:
            raise ValueError("V1 has no qualified other-agent witness")
        if (
            contributor is Contributor.SELF
            and self.outcome_state is not OutcomeState.ACTIVE_VERIFIED_AUTHORIZED
        ):
            raise ValueError("SELF needs an authorized verified effect")
        if (
            self.outcome_state is OutcomeState.ACTIVE_VERIFIED_AUTHORIZED
            and self.comparison is not Comparison.MATCH
        ):
            raise ValueError("an authorized effect always matches in V1")
        if (
            self.outcome_state is OutcomeState.VERIFIED_ABSENT
            and self.comparison is not Comparison.MISMATCH
        ):
            raise ValueError("VERIFIED_ABSENT is a mismatch against an ACTIVE forecast")
        _event(self.admitting_event_id, self.admitting_sequence, "admitting")
        _bounded(self.canonical_bytes(), "assessment")

    @classmethod
    def issue(
        cls,
        forecast: ExpectedObservationV1,
        outcome: VerifiedOutcomeView,
        admitting_event_id: str,
        admitting_sequence: int,
    ) -> Self:
        if admitting_sequence <= forecast.admitting_sequence:
            raise ValueError("an assessment follows its forecast")
        comparison, contributors = assess(forecast, outcome)
        return cls(
            forecast.forecast_id,
            forecast.forecast_digest,
            outcome.verified_outcome_id,
            outcome.outcome_revision,
            outcome.outcome_digest,
            outcome.outcome_state,
            comparison,
            contributors,
            admitting_event_id,
            admitting_sequence,
        )

    def _body(self) -> dict[str, object]:
        return {
            "admitting_event_id": self.admitting_event_id,
            "admitting_sequence": self.admitting_sequence,
            "comparison": self.comparison.value,
            "contributors": [item.value for item in self.contributors],
            "forecast_digest": self.forecast_digest,
            "forecast_id": self.forecast_id,
            "outcome_digest": self.outcome_digest,
            "outcome_revision": self.outcome_revision,
            "outcome_state": self.outcome_state.value,
            "schema_version": SCHEMA_VERSION,
            "verified_outcome_id": self.verified_outcome_id,
        }

    @property
    def assessment_id(self) -> str:
        return ASSESSMENT_ID_PREFIX + checksum(_ASSESSMENT_ID_DOMAIN, self._body())

    @property
    def assessment_digest(self) -> str:
        return checksum(_ASSESSMENT_DOMAIN, self._body())

    @property
    def positive_feedback_eligible(self) -> bool:
        return self.outcome_state is OutcomeState.ACTIVE_VERIFIED_AUTHORIZED

    def canonical_value(self) -> dict[str, object]:
        return {
            **self._body(),
            "assessment_digest": self.assessment_digest,
            "assessment_id": self.assessment_id,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = closed(
            parse_json(_bounded(payload, "assessment"), MAX_RECORD_BYTES),
            (*_ASSESSMENT_BODY, "assessment_digest", "assessment_id"),
            "assessment",
        )
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError("assessment schema_version is not V1")
        contributors = sequence(row["contributors"], "contributors", 1, json_input=True)
        value = cls(
            forecast_id=_str(row["forecast_id"]),
            forecast_digest=_str(row["forecast_digest"]),
            verified_outcome_id=_str(row["verified_outcome_id"]),
            outcome_revision=_int(row["outcome_revision"]),
            outcome_digest=_str(row["outcome_digest"]),
            outcome_state=parse_enum(row["outcome_state"], OutcomeState, "outcome_state"),
            comparison=parse_enum(row["comparison"], Comparison, "comparison"),
            contributors=tuple(
                parse_enum(item, Contributor, "contributor") for item in contributors
            ),
            admitting_event_id=_str(row["admitting_event_id"]),
            admitting_sequence=_int(row["admitting_sequence"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("assessment bytes are not canonical")
        return value


def feedback_state_of(assessment: OutcomeAssessmentV1) -> FeedbackState:
    """D-R17 §5: the FeedbackState a positive-eligible assessment yields."""

    if type(assessment) is not OutcomeAssessmentV1:
        raise TypeError("assessment must be an OutcomeAssessmentV1")
    if assessment.comparison is Comparison.MISMATCH:
        return FeedbackState.MISMATCH
    if assessment.comparison is Comparison.UNKNOWN:
        return FeedbackState.UNKNOWN
    if assessment.contributors == (Contributor.SELF,):
        return FeedbackState.MATCH_SELF
    return FeedbackState.MATCH_OTHER_OR_UNKNOWN


# ----------------------------------------------------------------- feedback

_FEEDBACK_BODY: Final = (
    "admitting_event_id",
    "admitting_sequence",
    "assessment_digest",
    "assessment_id",
    "feedback_revision",
    "feedback_state",
    "forecast_digest",
    "forecast_id",
    "outcome_digest",
    "outcome_revision",
    "schema_version",
    "supersedes",
    "verified_outcome_id",
)


@dataclass(frozen=True, slots=True)
class DecisionFeedbackV1:
    """Advisory Feedback for the next R16 Decision (D-R17 §4).

    It never mutates a Goal, Value, SelfModel or model weights. The stored
    ``feedback_state`` is a record of the value at issue; the projection
    recomputes it from the bound assessment and treats a difference as INVALID.
    """

    assessment_id: str
    assessment_digest: str
    forecast_id: str
    forecast_digest: str
    verified_outcome_id: str
    outcome_revision: int
    outcome_digest: str
    feedback_state: FeedbackState
    admitting_event_id: str
    admitting_sequence: int

    def __post_init__(self) -> None:
        _prefixed(self.assessment_id, ASSESSMENT_ID_PREFIX, "assessment_id")
        digest(self.assessment_digest, "assessment_digest")
        _prefixed(self.forecast_id, FORECAST_ID_PREFIX, "forecast_id")
        digest(self.forecast_digest, "forecast_digest")
        identifier(self.verified_outcome_id, "verified_outcome_id")
        _revision(self.outcome_revision, "outcome_revision", minimum=1)
        digest(self.outcome_digest, "outcome_digest")
        if type(self.feedback_state) is not FeedbackState:
            raise TypeError("feedback_state must be a FeedbackState")
        if self.feedback_state in {FeedbackState.ABSENT, FeedbackState.INVALID}:
            raise ValueError("ABSENT and INVALID are projections, never a record")
        _event(self.admitting_event_id, self.admitting_sequence, "admitting")
        _bounded(self.canonical_bytes(), "feedback")

    @classmethod
    def issue(
        cls,
        assessment: OutcomeAssessmentV1,
        admitting_event_id: str,
        admitting_sequence: int,
    ) -> Self:
        if type(assessment) is not OutcomeAssessmentV1:
            raise TypeError("assessment must be an OutcomeAssessmentV1")
        if not assessment.positive_feedback_eligible:
            raise ValueError("only an authorized verified effect yields Feedback")
        if admitting_sequence <= assessment.admitting_sequence:
            raise ValueError("Feedback is a later event than its assessment")
        return cls(
            assessment.assessment_id,
            assessment.assessment_digest,
            assessment.forecast_id,
            assessment.forecast_digest,
            assessment.verified_outcome_id,
            assessment.outcome_revision,
            assessment.outcome_digest,
            feedback_state_of(assessment),
            admitting_event_id,
            admitting_sequence,
        )

    @property
    def feedback_revision(self) -> int:
        return V1_FEEDBACK_REVISION

    def _body(self) -> dict[str, object]:
        return {
            "admitting_event_id": self.admitting_event_id,
            "admitting_sequence": self.admitting_sequence,
            "assessment_digest": self.assessment_digest,
            "assessment_id": self.assessment_id,
            "feedback_revision": V1_FEEDBACK_REVISION,
            "feedback_state": self.feedback_state.value,
            "forecast_digest": self.forecast_digest,
            "forecast_id": self.forecast_id,
            "outcome_digest": self.outcome_digest,
            "outcome_revision": self.outcome_revision,
            "schema_version": SCHEMA_VERSION,
            "supersedes": None,
            "verified_outcome_id": self.verified_outcome_id,
        }

    @property
    def feedback_id(self) -> str:
        return FEEDBACK_ID_PREFIX + checksum(_FEEDBACK_ID_DOMAIN, self._body())

    @property
    def feedback_digest(self) -> str:
        return checksum(_FEEDBACK_DOMAIN, self._body())

    def policy_ref(self) -> PolicyRef:
        """The SuccessorPolicyInput ``feedback_ref`` (D-R17 §5)."""
        return PolicyRef(
            FEEDBACK_REF_KIND, self.feedback_id, self.feedback_revision, self.feedback_digest
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            **self._body(),
            "feedback_digest": self.feedback_digest,
            "feedback_id": self.feedback_id,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = closed(
            parse_json(_bounded(payload, "feedback"), MAX_RECORD_BYTES),
            (*_FEEDBACK_BODY, "feedback_digest", "feedback_id"),
            "feedback",
        )
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError("feedback schema_version is not V1")
        if row["feedback_revision"] != V1_FEEDBACK_REVISION or row["supersedes"] is not None:
            raise ValueError("V1 Feedback is revision 1 with no supersession")
        value = cls(
            assessment_id=_str(row["assessment_id"]),
            assessment_digest=_str(row["assessment_digest"]),
            forecast_id=_str(row["forecast_id"]),
            forecast_digest=_str(row["forecast_digest"]),
            verified_outcome_id=_str(row["verified_outcome_id"]),
            outcome_revision=_int(row["outcome_revision"]),
            outcome_digest=_str(row["outcome_digest"]),
            feedback_state=parse_enum(row["feedback_state"], FeedbackState, "feedback_state"),
            admitting_event_id=_str(row["admitting_event_id"]),
            admitting_sequence=_int(row["admitting_sequence"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("feedback bytes are not canonical")
        return value


# --------------------------------------------------------------- projection


@dataclass(frozen=True, slots=True)
class TerminalOutcomeRevision:
    """A terminal forward Outcome revision and the sequence that made it so."""

    verified_outcome_id: str
    outcome_revision: int
    terminal_sequence: int

    def __post_init__(self) -> None:
        identifier(self.verified_outcome_id, "verified_outcome_id")
        _revision(self.outcome_revision, "outcome_revision", minimum=1)
        event_sequence(self.terminal_sequence, "terminal_sequence")


def select_feedback(
    outcome_revisions: Iterable[TerminalOutcomeRevision],
    feedbacks: Iterable[DecisionFeedbackV1],
    before_sequence: int,
) -> DecisionFeedbackV1 | None:
    """D-R17 §5: the latest terminal forward Outcome revision first, then its
    latest Feedback; none for that revision means no Feedback (ABSENT).

    Callers pass only the forward_key's Outcome revisions and Feedback: an
    inverse Outcome is never a revision of the forward Outcome.
    """

    event_sequence(before_sequence, "before_sequence")
    revisions = tuple(outcome_revisions)
    records = tuple(feedbacks)
    if any(type(item) is not TerminalOutcomeRevision for item in revisions):
        raise TypeError("outcome_revisions must be TerminalOutcomeRevision values")
    if any(type(item) is not DecisionFeedbackV1 for item in records):
        raise TypeError("feedbacks must be DecisionFeedbackV1 values")
    outcome_ids = {item.verified_outcome_id for item in revisions} | {
        item.verified_outcome_id for item in records
    }
    if len(outcome_ids) > 1:
        raise ValueError("pass the revisions and Feedback of one forward Outcome only")
    terminal = [item for item in revisions if item.terminal_sequence < before_sequence]
    if not terminal:
        return None
    latest = max(terminal, key=lambda item: item.outcome_revision)
    if sum(1 for item in terminal if item.outcome_revision == latest.outcome_revision) != 1:
        raise ValueError("one forward Outcome revision appears twice")
    matching = [
        item
        for item in records
        if item.admitting_sequence < before_sequence
        and item.verified_outcome_id == latest.verified_outcome_id
        and item.outcome_revision == latest.outcome_revision
    ]
    if not matching:
        return None
    if len(matching) != 1:
        # V1 issues one Feedback revision per Outcome revision (UNIQUE key).
        raise ValueError("more than one V1 Feedback for one Outcome revision")
    return matching[0]


def project_feedback_state(
    ledger: LedgerState,
    forecast: ExpectedObservationV1 | None,
    feedback: DecisionFeedbackV1 | None,
    assessment: OutcomeAssessmentV1 | None,
    outcome: VerifiedOutcomeView | None,
    *,
    action_intent_id: str,
    forward_key: str,
    claim_sequence: int | None,
) -> FeedbackState:
    """D-R17 §5: the trusted projection to SuccessorPolicy FeedbackState.

    Inputs, all as of the successor Decision's horizon and read by the issuer
    (IP-19):

    - ``ledger``: the current R18 ledger state of this forward operation;
    - ``forecast``: the terminal forecast found for this forward_key, if any;
    - ``feedback``: the record ``select_feedback`` chose, and ``assessment``
      the one it binds;
    - ``outcome``: the latest terminal forward Outcome revision, if any;
    - ``action_intent_id`` / ``forward_key``: the ActionIntent the ledger is
      about; ``claim_sequence``: the EXECUTION_CLAIM event, once claimed.

    Only this adapter emits INVALID. A read failure is not an input here: the
    issuer turns it into UNAVAILABLE (HOLD) and never calls this function.
    In V1 only MATCH_SELF, MATCH_OTHER_OR_UNKNOWN, ABSENT and INVALID can
    come out: Feedback needs an authorized effect, and an authorized effect
    always MATCHes, so MISMATCH / UNKNOWN are structurally unreachable here.
    """

    if type(ledger) is not LedgerState:
        raise TypeError("ledger must be a LedgerState")
    claimed = ledger not in _PRE_CLAIM_LEDGER
    if claimed and claim_sequence is None and ledger is not LedgerState.CORRUPT:
        raise ValueError("a claimed ledger needs its claim sequence")
    if forecast is None:
        if claimed:
            return FeedbackState.INVALID  # #367 D2-2: claimed without a forecast
        return FeedbackState.ABSENT if feedback is None else FeedbackState.INVALID
    if forecast.action_intent_id != action_intent_id or forecast.forward_key != forward_key:
        return FeedbackState.INVALID
    if claim_sequence is not None and forecast.admitting_sequence >= claim_sequence:
        return FeedbackState.INVALID  # the forecast must precede the effect's claim
    if feedback is None:
        return FeedbackState.ABSENT
    if assessment is None or outcome is None or ledger not in _FEEDBACK_LEDGER:
        return FeedbackState.INVALID
    if (
        feedback.forecast_id != forecast.forecast_id
        or feedback.forecast_digest != forecast.forecast_digest
        or assessment.forecast_id != forecast.forecast_id
        or assessment.forecast_digest != forecast.forecast_digest
        or feedback.assessment_id != assessment.assessment_id
        or feedback.assessment_digest != assessment.assessment_digest
        or feedback.admitting_sequence <= assessment.admitting_sequence
        or assessment.admitting_sequence <= forecast.admitting_sequence
    ):
        return FeedbackState.INVALID
    for record in (feedback, assessment):
        if (
            record.verified_outcome_id != outcome.verified_outcome_id
            or record.outcome_revision != outcome.outcome_revision
            or record.outcome_digest != outcome.outcome_digest
        ):
            return FeedbackState.INVALID
    if (
        outcome.outcome_state is not OutcomeState.ACTIVE_VERIFIED_AUTHORIZED
        or assessment.outcome_state is not outcome.outcome_state
    ):
        # Feedback from an Outcome that was not authorized when issued,
        # including an inverse after UNAUTHORIZED_EFFECT.
        return FeedbackState.INVALID
    recomputed = feedback_state_of(assessment)
    if recomputed is not feedback.feedback_state:
        return FeedbackState.INVALID
    return recomputed
