# G1-P: Externally terminal Context projection

Status: isolated implementation candidate. This is NOT the G1-C event-time capture participant, a G2 source witness or a NarrativeSelf producer.

Tracking: https://github.com/upiscium/PROJECT-SUZKA/issues/288#issuecomment-6107935748

## Semantics

Public GET /api/contexts and GET /api/contexts/{id} use a single private immutable Context snapshot. They never dereference the mutable worker ContextRegistry. Public reads linearize when a READY snapshot is admitted under one short lock. The response is entirely materialized from that generation, and an already-admitted GET may finish transmitting after a newer event begins.

A pending normal CHAT or CONTEXT_UPDATE leaves the last EXTERNALLY TERMINAL Context as the public view. Internally committing AgentState does not update it. Before the event terminal Journal append, public Context is marked PENDING. New GET admissions then return generic 503. After the durable terminal and receipt-retention checkpoint, the entire new view is atomically published before the event future returns. Any uncertain failure in preparation, internal commit, finalization or terminal/publication leaves public GET DEGRADED (503). A verified ordinary handler failure restores the old snapshot and allows the previous ready view.

Context POST transition/relate results capture both the frame and current pointer inside the single serialized event. Returning the result waits until event terminal; later worker mutations cannot change its is_current value.

Public responses have terminal-sequence, Context-revision and visibility headers. These are an API read-generation label, NOT authority to say any episode has a genuine event-time Context capture.

Startup requires reconciled existing R07 Journal plus AgentState. When B has never been installed, it is NOT REQUIRED for this public-only projection and is NEVER backfilled. Older legacy snapshots without any Context root require an empty restored registry. Future installed-but-dirty B baselines cannot be treated as missing; introduction of B needs its own separate Human implementation gate.

## Exact reader scope

The two Context GET routes are rewritten. Context POST transition/relate replies are assembled inside active events. Serialized CHAT/debug handling and their worker-private ContextPromptView and R14 Attention decisions are unchanged. AgentState capture/restore remains unchanged. Existing Value/Belief/R13/Attention read models are currently published at internal commit and are NOT silently shifted by this G1-P change. Debug chat response is sent only after event terminal.

No public generic Context source proof or arbitrary historical lookup is added. No R08 or R14 policy changes, R07 record format changes, memory/DB access changes, B quota/ticket/participant, R15 root changes or private transcript persistence.

## Focused evidence

Tests cover: uncommitted Context GET suppression, pre-model completion GET, internal-commit versus external-finalization GET, PENDING 503, terminal swap, handler recovery, failed terminal and degraded public status, terminal-success-before-publication fault and restart, legacy absence of B, admin mutation response generation and private read-model state transitions.

Focused commands:
    nix develop --command uv run pytest -q tests/test_g1p_context_publication.py tests/test_context_api.py
    nix develop --command uv run ruff check suzka/api/context_read_model.py suzka/api/server.py suzka/api/routes/contexts.py
    nix develop --command uv run mypy suzka/api/context_read_model.py suzka/api/server.py suzka/api/routes/contexts.py

Required before Ready or Merge: all backend tests, exact-head CI, independent correctness and security review, CP7 complete public/private Context-derived reader inventory, and separate Human integration approval. Do not claim G2 physically pure source reading or P1 genuine autobiographical reinterpretation.

Design is bounded to G1-P. If the reader inventory reveals out-of-scope R14 or AgentState mutation needed, stop and request explicit Human rescoping.
