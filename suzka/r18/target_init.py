"""IP-16a: the operator's one-time target creation (D1 §3-5, Human decision §8-2).

    python -m suzka.r18.target_init /abs/path/target.sqlite3

Creates the fixed target (IP-15) and the obligation secret (§3a) in the
target's private directory. Nothing creates them at startup, and neither is
ever replaced: an existing target or secret is refused (#380 D2 addendum 3).
For the lab only while HP-6 keeps R18 off in production.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from suzka.r18.obligation import (
    SECRET_FILE,
    TARGET_NAMESPACE,
    ObligationSecretError,
    check_secret_directory,
    create_obligation_secret,
)
from suzka.r18.target import TargetError, TargetStore


def initialize(path: Path) -> tuple[str, Path]:
    """Create the target and its secret; return the target instance id."""

    check_secret_directory(path.parent)
    if os.path.lexists(path.parent / SECRET_FILE):
        raise ObligationSecretError("the obligation secret already exists; it is never replaced")
    store = TargetStore.create(path, target_namespace=TARGET_NAMESPACE)
    try:
        try:
            instance = store.meta().target_instance_id
        finally:
            store.close()
        secret = create_obligation_secret(path.parent)
    except (TargetError, ObligationSecretError, OSError) as error:
        # The target is empty; the operator removes it and runs this again.
        raise ObligationSecretError(
            f"target {path} was created but its obligation secret was not ({error}); "
            "remove the empty target and run target_init again"
        ) from error
    return instance, secret


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m suzka.r18.target_init /abs/path/target.sqlite3", file=sys.stderr)
        return 2
    try:
        instance, secret = initialize(Path(args[0]))
    except (TargetError, ObligationSecretError) as error:
        print(f"target creation refused: {error}", file=sys.stderr)
        return 1
    print(f"created target {instance} and obligation secret {secret}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["initialize", "main"]
