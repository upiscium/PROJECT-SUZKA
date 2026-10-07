"""Narrow runtime persistence port for process-local Attention continuity."""

from __future__ import annotations

from threading import get_ident

from suzka.attention.contracts import AttentionContinuity
from suzka.attention.system import AttentionSystem, _Bundle
from suzka.runtime.attention_state_codec import validated_attention_continuity


_TRANSACTION_SEAL = object()


class AttentionRestoreTransaction:
    """Sealed lock-held publication handle for one staged Attention restore."""

    _completed: bool
    _closed: bool
    _owner: AttentionSystem
    _owner_thread: int
    _port: AttentionStatePort
    _prior_bundle: _Bundle
    _published: bool
    _target_bundle: _Bundle

    __slots__ = (
        "_completed",
        "_closed",
        "_owner",
        "_owner_thread",
        "_port",
        "_prior_bundle",
        "_published",
        "_target_bundle",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Attention restore transactions are created by their state port")

    @classmethod
    def _create(
        cls,
        port: AttentionStatePort,
        owner: AttentionSystem,
        owner_thread: int,
        prior_bundle: _Bundle,
        target_bundle: _Bundle,
        *,
        _seal: object,
    ) -> AttentionRestoreTransaction:
        if _seal is not _TRANSACTION_SEAL:
            raise TypeError("Attention restore transaction factory is private")
        transaction = object.__new__(cls)
        transaction._port = port
        transaction._owner = owner
        transaction._owner_thread = owner_thread
        transaction._prior_bundle = prior_bundle
        transaction._target_bundle = target_bundle
        transaction._published = False
        transaction._completed = False
        transaction._closed = False
        return transaction

    def _require_open_owner_thread(self) -> None:
        if get_ident() != self._owner_thread:
            raise RuntimeError("Attention restore transaction belongs to another thread")
        if self._closed:
            raise RuntimeError("Attention restore transaction is closed")
        if self._port._system is not self._owner:
            raise RuntimeError("Attention restore transaction owner no longer matches its port")

    def publish(self) -> None:
        """Publish the already validated immutable bundle as one final swap."""

        self._require_open_owner_thread()
        if self._published or self._completed:
            raise RuntimeError("Attention restore transaction was already published")
        if self._owner._bundle is not self._prior_bundle:
            raise RuntimeError("Attention owner changed while its restore lock was held")
        self._published = True
        self._owner._bundle = self._target_bundle

    def complete(self) -> None:
        """Mark success only after the staged bundle has been published."""

        self._require_open_owner_thread()
        if not self._published or self._completed:
            raise RuntimeError("Attention restore transaction is not publishable as complete")
        if self._owner._bundle is not self._target_bundle:
            raise RuntimeError("published Attention bundle changed before completion")
        self._completed = True

    def close(self) -> None:
        """Roll back by reference unless complete, then release the owner lock."""

        if self._closed:
            return
        if get_ident() != self._owner_thread:
            raise RuntimeError("Attention restore transaction belongs to another thread")
        if not self._completed:
            self._owner._bundle = self._prior_bundle
        self._closed = True
        self._owner._lock.release()


class AttentionStatePort:
    """Persistence-only access to one exact, trusted AttentionSystem owner."""

    __slots__ = ("_system",)
    _system: AttentionSystem

    def __init__(self, system: AttentionSystem) -> None:
        if type(system) is not AttentionSystem:
            raise TypeError("AttentionStatePort requires an exact AttentionSystem owner")
        self._system = system

    def export_attention_state(self) -> AttentionContinuity:
        """Return a validated detached continuity value, never a system bundle."""

        return validated_attention_continuity(self._system.snapshot())

    def prepare_attention_restore(
        self,
        snapshot: AttentionContinuity,
    ) -> AttentionRestoreTransaction:
        """Stage a complete immutable bundle while holding the actual owner lock."""

        owner = self._system
        owner._lock.acquire()
        try:
            prior_bundle = owner._bundle
            if type(snapshot) is not AttentionContinuity:
                raise TypeError("snapshot must be an exact AttentionContinuity")
            validated = validated_attention_continuity(snapshot)
            staged_system = AttentionSystem(validated)
            target_bundle = staged_system._bundle
            return AttentionRestoreTransaction._create(
                self,
                owner,
                get_ident(),
                prior_bundle,
                target_bundle,
                _seal=_TRANSACTION_SEAL,
            )
        except BaseException:
            owner._lock.release()
            raise


__all__ = ["AttentionRestoreTransaction", "AttentionStatePort"]
