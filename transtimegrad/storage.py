"""Transactional publication of owned directory artifacts."""

import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Optional


def atomic_directory_write(
    destination: Path,
    writer: Callable[[Path], None],
    validate_existing: Callable[[Path], None],
    validate_staged: Optional[Callable[[Path], None]] = None,
) -> None:
    """Stage an artifact beside its destination, then publish with rollback.

    The caller owns the format-specific validation. It must reject unrelated
    content and symlinks before this helper replaces any existing directory.
    """
    destination = Path(destination).absolute()
    if destination.is_symlink():
        raise ValueError("Artifact destination must not be a symlink")
    if destination.exists() and not destination.is_dir():
        raise ValueError("Artifact destination must be a directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Resolve only the parent; resolving the leaf would follow a forbidden link.
    destination = destination.parent.resolve() / destination.name
    if destination.exists():
        validate_existing(destination)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-stage-", dir=destination.parent))
    backup = None
    try:
        writer(stage)
        (validate_staged or validate_existing)(stage)
        # Recheck ownership immediately before the first mutation of old state.
        if destination.is_symlink():
            raise ValueError("Artifact destination became a symlink")
        if destination.exists():
            validate_existing(destination)
            backup = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}-backup-", dir=destination.parent)
            )
            try:
                destination.rename(backup)
            except BaseException:
                backup.rmdir()
                backup = None
                raise
        try:
            stage.rename(destination)
        except BaseException:
            if backup is not None:
                try:
                    backup.rename(destination)
                    backup = None
                except BaseException as error:
                    raise RuntimeError(
                        f"Artifact publication and rollback failed; previous artifact retained at {backup}"
                    ) from error
            raise
        if backup is not None:
            try:
                shutil.rmtree(backup)
                backup = None
            except OSError:
                warnings.warn(
                    f"Artifact saved; previous artifact retained at {backup}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    finally:
        if stage.exists():
            shutil.rmtree(stage)
