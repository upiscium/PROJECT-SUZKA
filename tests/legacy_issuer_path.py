"""IP-20a: the PRE-refactor PLAN_PREVIEW issuer path, verbatim (IP-20 D1 r3 §5-1).

``_Legacy`` is ``_DecisionIssuer`` with ``_issue``, ``_issue_qualified``,
``_ledger``, ``_check_deadline`` and ``_committed`` exactly as they were at
``origin/rebuild/develop@37b018d`` (before ``_evaluate``). The equivalence test
runs it next to the refactored issuer on the same event and compares the
results byte for byte with one fixed clock. Delete this file only together with
the equivalence test.
"""

# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

import suzka.r16.decision_issuer as _module

globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})


class _Legacy(_DecisionIssuer):
    def _issue(self, event: AgentEvent, goal_id: str) -> object:
        from suzka.r16.plan_preview import _PlanSourceConflict, _PlanSourceUnavailable

        self._pending = None
        initial = self._preview._state_snapshot()
        state: RequiredSourceState | None
        try:
            before = self._preview._qualify(goal_id, event)
            after = self._preview._qualify(goal_id, event)
            if after != before or self._preview._state_snapshot() != initial:
                raise _PlanSourceUnavailable("Decision source changed")
        except _PlanSourceConflict:
            state = RequiredSourceState.CONFLICT
        except _PlanSourceUnavailable:
            state = RequiredSourceState.UNAVAILABLE
        else:
            state = None
        try:
            if state is not None:
                return self._unqualified(event, state)
            return self._issue_qualified(event, before, initial)
        except (HistoryStoreError, AgentStateLoadError):
            # Review P2-3: a temporary store/state read failure is a HOLD.
            hold = _IssuerHold(R16OwnerReason.SOURCE_UNAVAILABLE)
            LOGGER.info("R16 issuer result=hold reason=%s", hold.reason.value)
            return _DecisionIssueResult(
                DecisionIssueStatus.HOLD, None, None, None, hold.reason
            )
        except _IssuerHold as hold:
            LOGGER.info("R16 issuer result=hold reason=%s", hold.reason.value)
            return _DecisionIssueResult(
                DecisionIssueStatus.HOLD, None, None, None, hold.reason
            )

    def _issue_qualified(
        self, event: AgentEvent, q: _QualifiedPreview, initial: object
    ) -> object:
        scope = lineage_scope_for(q.experience_id)
        snapshot_hash, as_of = self._snapshot()
        now_utc = utc_timestamp(event.requested_at)
        inspection = self._journal.inspect()
        lineage, lineage_ref, incarnation, head, attested, classified = self._lineage(
            inspection, scope
        )
        commitment, commitment_ref, deadline = self._commitment(q)
        interpretation, stage = self._read_interpretation(event, q)
        predecessor, predecessor_ref = self._predecessor(
            head, incarnation, scope, as_of, snapshot_hash, attested, q.self_origin
        )
        try:
            ledger = self._ledger(classified, predecessor, q, _sequence(event))
        except _LedgerSourceUnavailable:
            return self._unqualified(event, RequiredSourceState.UNAVAILABLE)
        value = SuccessorPolicyInputV2(
            required_source=RequiredSourceState.QUALIFIED,
            source_refs=_required_refs(q),
            predecessor=predecessor,
            predecessor_ref=predecessor_ref,
            lineage=lineage,
            lineage_ref=lineage_ref,
            commitment=commitment,
            commitment_ref=commitment_ref,
            commitment_deadline_utc=deadline,
            ledger=ledger.state,
            ledger_refs=ledger.refs,
            inverse_authorization=ledger.inverse_authorization,
            feedback=ledger.feedback,
            feedback_ref=ledger.feedback_ref,
            verified_outcome_ref=ledger.verified_outcome_ref,
            interpretation=interpretation,
            interpretation_stage=stage,
            as_of_r07_sequence=as_of,
            as_of_snapshot_hash=snapshot_hash,
            event_id=event.event_id,
            event_sequence=_sequence(event),
            now_utc=now_utc,
            version=POLICY_KIND,
        )
        result = evaluate(value)
        if (
            predecessor is PredecessorState.QUALIFIED
            and head is not None
            and head.decision is not None
        ):
            existing = self._compare_head(head.decision, value, result)
            if existing is not None:
                return existing
        if result.is_hold:
            return self._hold_result(result)
        assert incarnation is not None
        committed = self._committed(event, q, value, result, incarnation, scope, head)
        self._check_slots(committed, incarnation)
        self._check_deadline(q)
        # The sources must not move while we built the Decision.
        if self._preview._qualify(q.goal_id, event) != q or (
            self._preview._state_snapshot() != initial
        ):
            raise _IssuerHold(R16OwnerReason.SOURCE_UNAVAILABLE)
        transaction_id = TransactionCoordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        )
        self._pending = _PendingIssue(
            event,
            q,
            incarnation,
            snapshot_hash,
            operations_digest(self._store.operations(incarnation)),
            transaction_id,
            committed,
            initial,
        )
        LOGGER.info("R16 issuer result=issued row=%s", result.row.value)
        participant = SubjectHistoryParticipant(
            self._store,
            incarnation,
            committed,
            JournalCommitProofSource(self._journal, self._recovery),
        )
        return CoordinatedResult(
            _DecisionIssueResult(
                DecisionIssueStatus.ISSUED,
                committed.decision_id,
                result.row,
                None,
                None,
                committed,
            ),
            (participant,),
        )

    def _ledger(
        self,
        classified: ReaderClassification | None,
        predecessor: PredecessorState,
        q: _QualifiedPreview,
        before_sequence: int | None = None,
    ) -> LedgerRead:
        """IP-16b §5: the ledger of this lineage's obligation group.

        Read only when the predecessor is usable: a blocked, conflicting or
        unavailable head is row 0 / row 1 whatever the ledger says.  With no
        R17 / R18 data in the store (any incarnation, §10-1) nothing else is
        read and the witness is exactly the pre-IP-16b one.
        """

        if (
            classified is None
            or classified.global_block is not None
            or predecessor
            not in (PredecessorState.ABSENT_INITIAL, PredecessorState.QUALIFIED)
        ):
            return NO_LEDGER
        # IP-18b (§9-4 / §9-5): both views are read once per evaluation.
        facts = self._target_facts()
        blocked = self._blocked_target()
        has_obligation = q.commitment_id is not None
        if (
            not self._store.has_record_operations()
            and not self._store.has_live_claim_keys()
            and not facts
        ):
            # No R17 / R18 data and no facts: the real ledger is NONE with no
            # prior claim, so no group key is needed to decide the §9-5 witness.
            return with_target_blocked(
                NO_LEDGER, blocked=blocked, has_obligation=has_obligation, has_prior_claim=False
            )
        try:
            group_key = self._group_key(q)
        except _LedgerSourceUnavailable:
            if blocked and has_obligation:
                # A lost target usually takes the obligation secret next to it:
                # without this the subject would be held (row 1) by the very
                # outage the witness exists to keep it free of.
                return TARGET_UNAVAILABLE_LEDGER
            raise
        forward_key = operation_key(group_key, Direction.FORWARD)
        # N5: facts, once complete, take precedence; ``derive_ledger`` is
        # skipped (it would report CORRUPT on a forward record, or CLAIMED
        # with the prior-claim ref).
        fact_ledger = facts_ledger(facts, forward_key)
        if fact_ledger is not None:
            return fact_ledger
        prior = None
        owner = self._store.live_key_owner(f"claim:{forward_key}")
        if owner is not None and owner[0] != classified.incarnation:
            # Another incarnation claimed this obligation's forward effect.
            rows = self._store.record_rows("execution_claims", owner[0], owner[1])
            prior = prior_claim_ref(rows[0] if len(rows) == 1 else None, forward_key)
            if prior is None:
                LOGGER.info("R16 issuer ledger=CORRUPT reason=prior_claim_row")
                return CORRUPT_LEDGER
        records = tuple(
            RecordAt(item.record, item.processing_sequence) for item in classified.records
        )
        real = derive_ledger(
            records,
            group_key,
            # IP-17a (§3a / §12-3): the issuer's clock closes an unclaimed
            # forward Intent whose approval window has ended.  IP-20 note: a
            # Wake-origin evaluation at a frozen time T must pass T here, not
            # the live clock, or its closure would not match its sealed T.
            self._clock(),
            prior_claim=prior,
        )
        if prior is None and before_sequence is not None and real.state is not LedgerState.CORRUPT:
            # IP-19 (#367 D2-1, D-R17 §5): the Feedback witness, as of this
            # Decision's own event.  A claim of another incarnation, a facts
            # state and CORRUPT carry no R17 record of this incarnation, so
            # they keep ABSENT.  IP-20 note: a Wake-origin evaluation passes
            # its frozen horizon here, like the clock above.
            feedback = read_feedback(records, forward_key, real.state, before_sequence)
            real = replace(real, feedback=feedback.state, feedback_ref=feedback.ref)
        # §9-5: substitute LAST, and only over an empty ledger.
        return with_target_blocked(
            real, blocked=blocked, has_obligation=has_obligation, has_prior_claim=prior is not None
        )

    def _committed(
        self,
        event: AgentEvent,
        q: _QualifiedPreview,
        value: SuccessorPolicyInputV2,
        result: SuccessorPolicyResult,
        incarnation: str,
        scope: str,
        head: HistoryHead | None,
    ) -> CommittedDecisionV1:
        predecessor = (
            head.decision
            if head is not None and head.status is HeadStatus.ADOPTED
            else None
        )
        predecessor_key = GENESIS if predecessor is None else predecessor.decision_id
        policy_bytes = value.canonical_bytes()
        intent = checksum(
            _INTENT_DOMAIN,
            [incarnation, predecessor_key, value.evidence_fingerprint(), value.version.value],
        )
        identity = decision_id_for(
            incarnation,
            event.event_id,
            _sequence(event),
            intent,
            policy_input_digest(policy_bytes),
        )
        record = _decision_record(identity, q, result)
        if q.self_origin and record.selected.candidate_id == "candidate-b":
            # Explicit, not ``assert`` (D5 review P3-3): never issue Plan B
            # for a self-origin Goal.
            raise _IssuerHold(R16OwnerReason.PLAN_SOURCE_CONFLICT)
        return CommittedDecisionV1(
            incarnation_id=incarnation,
            decision_id=identity,
            issuer_event_id=event.event_id,
            issuer_sequence=_sequence(event),
            lineage_scope=scope,
            review_intent_key=intent,
            predecessor_key=predecessor_key,
            predecessor_digest=None if predecessor is None else predecessor.content_digest,
            decision=record,
            # IP-13: the label always follows the sealed bytes.
            policy_kind=value.version,
            policy_row=result.history_row,
            policy_input_bytes=policy_bytes,
            source_refs=_history_refs(q),
        )

    def _check_deadline(self, q: _QualifiedPreview) -> None:
        if q.commitment_id is None:
            return
        record = self._preview._loop._commitment_system.get(q.commitment_id)
        at = None if record is None else record.deadline.at
        if at is not None and at - self._clock() < DEADLINE_MARGIN:
            # Not yet elapsed, but too close to commit safely (D2 P2-2).
            raise _IssuerHold(R16OwnerReason.DEADLINE_ELAPSED)


def _rebind_to_live_module_globals() -> None:
    """Run the verbatim methods against the LIVE ``decision_issuer`` globals, so
    a test that monkeypatches a module-level name (``POLICY_KIND``, a helper)
    patches the legacy path exactly as it patches the refactored one."""

    import types

    for name in ("_issue", "_issue_qualified", "_ledger", "_check_deadline", "_committed"):
        function = _Legacy.__dict__[name]
        rebound = types.FunctionType(
            function.__code__,
            _module.__dict__,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        rebound.__kwdefaults__ = function.__kwdefaults__
        rebound.__annotations__ = function.__annotations__
        rebound.__qualname__ = function.__qualname__
        setattr(_Legacy, name, rebound)


_rebind_to_live_module_globals()
