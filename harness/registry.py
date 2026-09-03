"""Provider loading by reference string.

Provider identity is a "package.module:factory" string on purpose: strings
survive multiprocessing spawn (module-global hooks do not, measured in phase
1), pickle cleanly into episode specs, and enter content hashes unchanged.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any


def load_provider(ref: str, params: Mapping[str, Any] | None = None) -> Any:
    if ":" not in ref:
        raise ValueError(f"provider ref must be 'module:factory', got {ref!r}")
    module_name, attr = ref.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attr)
    return factory(**dict(params or {}))


def load_attr(ref: str) -> Any:
    """Import ``module:attr`` and return the ATTRIBUTE itself (not a call).

    The read half of the same crossing ``load_provider`` uses for factories:
    card-authored data (a catalogue of ``type`` objects, an oracle tuple, a
    docs table) is named by ref in a manifest or a brief and imported here,
    never carried as JSON. A ref without ``:`` is refused like a provider ref.
    """
    if ":" not in ref:
        raise ValueError(f"attribute ref must be 'module:attr', got {ref!r}")
    module_name, attr = ref.split(":", 1)
    return getattr(importlib.import_module(module_name), attr)
