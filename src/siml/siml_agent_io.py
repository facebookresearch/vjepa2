# src/siml/siml_agent_io.py
"""
SIML agent context. Provides surprise(...) for candidate action sequences.
Designed to be inference-only and optional: if not loaded, gates become no-ops.
"""

from typing import Optional, Callable, Any, Dict
import os
import importlib

class SimlContext:
    def __init__(
        self,
        path_bin: str = "./agent_memory.bin",
        path_bak: Optional[str] = None,
        surprise_fn: Optional[Callable[[Any, Any, Optional[dict]], float]] = None,
    ):
        self.path_bin = path_bin
        self.path_bak = path_bak
        self._loaded = False
        self._surprise_fn = surprise_fn

        # mark ready if a callable was provided
        if callable(self._surprise_fn):
            self._loaded = True
            return

        # or mark ready if any memory file exists (you will wire the callable later)
        for p in (path_bin, path_bak):
            if p and os.path.exists(p):
                # not "ready" for calling until a function is bound
                self._loaded = False
                break

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "SimlContext":
        """
        Optional builder that supports:
          cfg['agent_memory'] : path to memory file(s)
          cfg['surprise_entry']: 'module.path:callable' providing surprise(a_seq, s_k, extras) -> float
        """
        path = cfg.get("agent_memory", "./agent_memory.bin")
        bak  = cfg.get("agent_memory_bak", None)
        fn = None
        entry = cfg.get("surprise_entry")
        if entry:
            if ":" not in entry:
                raise ValueError(f"surprise_entry must be 'module.path:callable', got {entry!r}")
            mod_name, func_name = entry.split(":", 1)
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, func_name)
            if not callable(fn):
                raise TypeError(f"surprise_entry {entry!r} is not callable")
        return cls(path_bin=path, path_bak=bak, surprise_fn=fn)

    def bind_surprise(self, fn: Callable[[Any, Any, Optional[dict]], float]) -> None:
        """Attach a callable post-init."""
        if not callable(fn):
            raise TypeError("surprise must be callable")
        self._surprise_fn = fn
        self._loaded = True

    def ready(self) -> bool:
        """True when surprise(...) can be called safely."""
        return callable(self._surprise_fn)

    def surprise(self, a_seq, s_k, extras: Optional[dict] = None) -> float:
        """
        Return a scalar free-energy/surprise for the proposed action sequence.
        Must be wired to your SIML agent (see from_cfg(...)/bind_surprise(...)).
        """
        if not self.ready():
            raise RuntimeError("SimlContext.surprise called but no surprise_fn is bound. "
                               "Provide cfg['surprise_entry'] or call bind_surprise().")
        return float(self._surprise_fn(a_seq, s_k, extras))
