# HARD-band retention ladder (F15 DC-6)

Applies only with `degraded_continuation.mode: continue` (the default is
`refuse`: nothing below exists and the Journal HARD band behaves as before) and
`degraded_continuation.hard_band_auto_shrink: true` (the default; `false`
restores the old HARD-band behaviour on its own).

## What it does

When the Journal's live bytes are over `event_journal.max_total_bytes` (the
HARD band) no new event is admitted. Before DC-6 nothing could leave that
state: compaction ran only at start-up and after a terminal completion, and a
HARD band produces no terminal completion.

After `degraded_continuation.t_hard_s` (600 s) of continuous HARD, measured by
the supervisor across cycle rebuilds and process-start-in-HARD, the process
lowers the retention window stepwise and tries one compaction per step
(`hard_ladder_days`, default `[14, 7, 3, 1, 0]`). Steps that are not below the
configured `event_journal.retention_days` are dropped, never refused. It stops
at the first step after which the band is not HARD. A running Runtime is
compacted only while idle (`AgentRuntime.run_at_idle`); at start-up the Runtime
is started by a new cycle afterwards.

Every attempt is one line in `<quarantine_dir>/hard_band.jsonl` (default
`<journal dir>/quarantine/`): time, step, outcome, live bytes before and after.
The COMPACTION record carries the step's `retention_days`.

## What it does not do

- It archives only what `EventJournal.compact` already proves safe: whole
  rotated segments, never the active one, at a clean boundary, with the
  Semantic guard and the SubjectHistory attestation unchanged and byte-verified
  archives. It cannot free bytes that are not in rotated segments (then the
  state stays `HARD_STUCK`).
- Any reason other than `nothing_due` that does not compact
  (`previous_compaction_pending`, `history_attestation_*`, `deferred_*`,
  `boundary_not_clean`, `clock_not_monotonic`, `compaction_failed`,
  `journal_schema_below_4`, a Runtime that never becomes idle) stops the ladder:
  `dc_state = HARD_STUCK`, the reason in `dc_hard_reason`. It is retried only on
  a growing delay (1, 4, 16, 60, 300 s), never in a loop.

## Consequences (this overrides #381 D2 "the window is not shortened automatically")

- R15 narrative sources must be RETAINED in the Journal. A shorter window
  lowers R15 / R16 capability for events older than the effective window.
- A later true rollback is more likely to be refused
  (`participant_history_archived`) and end `RECONCILING`.
- Older Decision proof becomes "derived evidence" through the archive.
- **The window never returns by itself.** The segments are archived; the next
  ordinary compaction uses the configured window again but restores nothing.
  `/health` and `/health/dc` show `dc_hard_window_days` (the smallest window a
  step applied in this process) and `dc_hard_window_note` until the process
  restarts; the archives and the `hard_band.jsonl` lines are the lasting record.
- To refuse all of this: `hard_band_auto_shrink: false`, or raise
  `max_total_bytes`, or set `retention_days` low yourself.

## Health

`dc_state` is `HARD_WAIT` (counting `t_hard_s`), `HARD_LADDER` (a pass is
running) or `HARD_STUCK` (see `dc_hard_reason`: a compaction reason,
`ladder_empty`, `ladder_exhausted`, `run_at_idle_timeout`) with
`dc_reason = journal_hard_limit`, while the subject is SERVING or RECONCILING.
Chat stays refused (503) for new events until the band is cleared.

## Bounds

The per-step 60 s bounds the wait for the Runtime to become idle, not the
compaction itself (an in-flight compaction cannot be interrupted). A cycle
teardown waits for the watcher thread; one that does not end is reported by the
stop proof (`threads_not_stopped`). `hard_band.jsonl` is append-only and is not
rotated.

## Order of attempts, log, disk

- Before any shrinking step the ladder first tries ONE compaction with the
  configured `retention_days` ("rung 0", logged as `configured_window`). That
  is not a shrink: it archives only what the configured policy already allows
  and shows no `dc_hard_window_days`. If it clears HARD, nothing is shortened.
  This also applies when the filtered ladder is empty (`retention_days` at or
  below the lowest step); only a band still HARD after it ends `HARD_STUCK`
  (`ladder_empty` / `ladder_exhausted`).
- `hard_band.jsonl` gets one line per applied compaction and per CHANGE of a
  step's outcome (a permanently stuck ladder does not append every retry). It
  is capped at 256 KiB: past that the oldest half of its lines is dropped, the
  newest kept; nothing else is deleted.
- Archiving moves rotated segments from the Journal into `archive_dir`. It does
  NOT free disk space: HARD is about the Journal's live bytes (active plus
  rotated segments), which archives do not count.
- The watcher's stop join is `min(30 s, T_teardown_max / 8)` and is inside the
  `T_teardown_max` budget; it is instant unless a compaction is in flight.
- An unexpected error in the watcher is logged and retried on the same growing
  delay; it is never shown as `HARD_STUCK`. A watcher that cannot start leaves
  the cycle serving without a ladder (a missing ladder never drops the subject).
