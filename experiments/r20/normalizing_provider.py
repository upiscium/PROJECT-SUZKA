"""COMPROMISED model-call-layer normalisation of the S1 / S2 replies (HD-23, option 1).

The production R15 source gate (``suzka.r15.qualified_proposal``) re-parses the response STORED in
the subject's Experience.  Normalising only in the harness transport therefore passes the harness
check but leaves the raw reply in memory, and the gate refuses it.  So the experiment wraps the
model provider itself: for a registered S1 / S2 request the wrapper applies
``reply_normalizer.normalize_reply`` BEFORE the app sees the text, and the subject stores the
normalised text.  The model's literal reply is kept only in a private 0600 file.

Nothing under ``suzka/`` is touched: ``serve`` wraps an injected provider, or replaces the loader
name the app's lifespan calls (so every rebuilt cycle gets a wrapper again).
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import threading
from typing import Any

from suzka.persona.prompt_builder import CURRENT_MESSAGE_HEADING
from suzka.r15.request_templates import s1_request, s2_request

from suzka.r15.qualified_proposal import prior_referents  # PUBLIC: the production gate's own referent reader

from experiments.r15_p0.pilot import Case
from experiments.r15_p1.protocol import parse_candidate
from experiments.r20.reply_normalizer import normalize_reply

# HD-23 option (a): up to this many TOTAL attempts for one S1 / S2 request (COMPROMISE, preregistered).
REPLY_ATTEMPTS_MAX = 3
# HD-23 (E cap): the E reply (turn 2) alone may be regenerated up to this many TOTAL attempts (COMPROMISE, preregistered).
E_ATTEMPTS_MAX = 6

RAW_NAME = "p1.model_raw.jsonl"  # private: the model's literal text
NAMES_NAME = "p1.model_names.jsonl"  # private too, but names only


def _common_prefix(first: str, second: str) -> str:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return first[:length]


# The fixed text of each registered request, up to where its quoted JSON begins.  Derived from the
# registry's own PUBLIC builders (never copied): two builds that differ only in their inputs share
# exactly the fixed prefix.  ``suzka.r15.request_templates`` exposes no public prefix / matcher that
# works without the observed turns, so this is the minimal stable prefix check.
S1_PREFIX = _common_prefix(s1_request("a", "b"), s1_request("c", "d"))
S2_PREFIX = _common_prefix(
    s2_request("a", "b", "c", e2_input="d"), s2_request("e", "f", "g", e2_input="h")
)
_USER_TURN = "\nUser: "
_COMPLETION_CUE = "\nAssistant:"
# The P1 protocol's second prompt (the E turn), from the pilot's own builder.  Its text does not depend on the
# case id / run id (only target / lane / variant / control), so the S0 case's E text is the registered one.
E_REQUEST = Case("r20-e-detect", "B", "forced", 0, "normal").prompts()["E"]


def _is_current_e(text: str) -> bool:
    """Exact match of the CURRENT message with the E prompt (the last thing in ``text``)."""

    body = text.removesuffix(_COMPLETION_CUE)  # the built prompt ends "User: <message>\nAssistant:"
    if not body.endswith(E_REQUEST):
        return False
    before = body[: len(body) - len(E_REQUEST)]
    return before == "" or before.endswith(("\n", ": "))


def registered_phase(text: str) -> str | None:
    """``"S1"`` / ``"S2"`` / ``"E"`` when the CURRENT message of ``text`` is a registered request.

    ``text`` is the built prompt (``generate``) or the user part (``generate_chat``); the current
    message is the last thing in it.  A registered request that only sits in an earlier memory line
    (before the current message) does not count.
    """

    if _is_current_e(text):
        return "E"
    best: tuple[int, str] | None = None
    for phase, prefix in (("S1", S1_PREFIX), ("S2", S2_PREFIX)):
        index = text.rfind(prefix)
        if index >= 0 and (best is None or index > best[0]):
            best = (index, phase)
    if best is None:
        return None
    tail = text[best[0] :]
    if _USER_TURN in tail or CURRENT_MESSAGE_HEADING in tail:
        return None
    return best[1]


def acceptable(text: str, phase: str) -> bool:
    """True when the (already normalised) reply parses under the experiment's strict P1 parser.

    The production source gate (``suzka.r15.qualified_proposal._parse_candidate``, private) applies
    the same closed schema with the same duplicate-key / constant refusals; it is NOT imported here.
    ``tests/test_r20_reply_retry.py`` asserts that both agree on a sample set.
    """

    if phase == "E":  # exactly one prior referent, read by the production gate's public function
        return len(prior_referents(text)) == 1
    try:
        parse_candidate(text, phase=phase, has_reveal=phase == "S2")
    except ValueError:  # TrialRefused is a ValueError
        return False
    return True


def _append(path: Path, row: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class NormalizingProvider:
    """Delegates everything to ``inner``; only ``generate`` / ``generate_chat`` are overridden.

    ``generate_chat`` exists on the wrapper only when the inner provider has it (the app probes it).
    """

    def __init__(self, inner: Any, private_dir: Path) -> None:
        self._inner = inner
        self._raw_path = Path(private_dir) / RAW_NAME
        self._names_path = Path(private_dir) / NAMES_NAME
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:  # only called for names not found on the wrapper
        if name == "_inner":
            raise AttributeError(name)
        target = getattr(self._inner, name)  # AttributeError when the inner has none: callers probe with getattr
        return self._generate_chat if name == "generate_chat" else target

    def _finish(self, request_text: str, call: Callable[[], Any]) -> str:
        """Call ``inner`` once, or (registered S1 / S2 / E only) up to its cap times (E_ATTEMPTS_MAX for E)."""

        phase = registered_phase(request_text)
        if phase is None:
            return str(call())
        text, applied, accepted, attempts = "", [], False, 0
        cap = E_ATTEMPTS_MAX if phase == "E" else REPLY_ATTEMPTS_MAX  # E: 6, S1 / S2: 3
        while attempts < cap and not accepted:
            attempts += 1
            raw = str(call())
            text, applied = (raw, []) if phase == "E" else normalize_reply(raw, phase)  # E: regenerate only
            accepted = acceptable(text, phase)
            with self._lock:
                _append(
                    self._raw_path,
                    {"phase": phase, "attempt": attempts, "raw": raw, "normalizations": list(applied), "accepted": accepted},
                )
        with self._lock:  # the LAST attempt's text is returned when none is acceptable (never fabricated)
            _append(
                self._names_path,
                {"phase": phase, "normalizations": list(applied), "attempts": attempts, "accepted": accepted},
            )
        return text

    def generate(self, prompt: str) -> str:
        return self._finish(prompt, lambda: self._inner.generate(prompt))

    def _generate_chat(self, system_text: str, user_text: str) -> str:
        return self._finish(user_text, lambda: self._inner.generate_chat(system_text, user_text))


def wrap_loader(loader: Callable[..., Any], private_dir: Path) -> Callable[..., Any]:
    """A drop-in for ``load_model_provider``: every provider it builds is wrapped."""

    def load(*arguments: Any, **keywords: Any) -> NormalizingProvider:
        return NormalizingProvider(loader(*arguments, **keywords), private_dir)

    return load


def read_names(private_dir: Path) -> dict[str, list[str]]:
    """phase -> the names applied to the LAST registered reply of that phase (names only)."""

    path = Path(private_dir) / NAMES_NAME
    names: dict[str, list[str]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            names[str(row["phase"])] = [str(name) for name in row["normalizations"]]
    return names


def read_attempts(private_dir: Path) -> dict[str, dict[str, Any]]:
    """phase -> ``{"attempts": int, "accepted": bool}`` of the LAST registered request of that phase."""

    path = Path(private_dir) / NAMES_NAME
    found: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            found[str(row["phase"])] = {"attempts": int(row["attempts"]), "accepted": bool(row["accepted"])}
    return found


__all__ = [
    "E_ATTEMPTS_MAX", "NAMES_NAME", "RAW_NAME", "REPLY_ATTEMPTS_MAX", "acceptable", "read_attempts", "S1_PREFIX", "S2_PREFIX", "NormalizingProvider", "read_names",
    "registered_phase", "wrap_loader",
]
