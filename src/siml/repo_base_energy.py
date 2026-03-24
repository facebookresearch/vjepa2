# src/siml/repo_base_energy.py
"""
Expose your repository's Eq.(5) latent L1 energy as a callable:
    base_energy(a_seq, z_k, s_k, z_g) -> float

Resolution order:
1) If cfg['predictor_entry'] is provided (e.g., "pkg.mod:factory"), import & call it.
2) Search the repo tree for a Python file that defines a predictor factory or callable:
     - Function names tried: make_ac_predictor, make_eq5_predictor, make_predictor, predict_latent
     - Likely files: scaffold.py, *predict*.py inside eval*/ or src/eval*/ subdirs.
3) If nothing found: raise RuntimeError (no silent fallbacks).
"""

from typing import Any, Dict, Callable, Optional
import importlib
import importlib.util
import os
import sys
import inspect
import numpy as np

# ---------------- utilities ----------------

def _mean_abs(a, b) -> float:
    try:
        import torch  # lazy
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            a = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
            b = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
            return float(torch.abs(a - b).mean().item())
    except Exception:
        pass
    a = np.asarray(a); b = np.asarray(b)
    return float(np.mean(np.abs(a - b)))

def _load_entrypoint(entry: str) -> Callable[..., Any]:
    if ":" not in entry:
        raise ValueError(f'Entry must be "module.path:callable", got {entry!r}')
    mod_name, func_name = entry.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise TypeError(f"Entrypoint {entry!r} is not callable")
    return fn

def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _repo_root_from_this_file() -> str:
    # This file is at src/siml/repo_base_energy.py → repo_root is two levels up from src/
    here = os.path.abspath(os.path.dirname(__file__))         # .../src/siml
    src_dir = os.path.abspath(os.path.join(here, ".."))       # .../src
    root = os.path.abspath(os.path.join(src_dir, ".."))       # repo root
    return root

# ---------------- predictor resolution ----------------

_PRED_NAMES = ("make_ac_predictor", "make_eq5_predictor", "make_predictor", "predict_latent")
_PRED_FILE_HINTS = ("scaffold.py", "predict", "planner", "eval")

def _resolve_predictor_from_entry(cfg: Dict[str, Any]) -> Optional[Any]:
    entry = cfg.get("predictor_entry")
    if not entry:
        return None
    factory = _load_entrypoint(entry)
    try:
        return factory(cfg)
    except TypeError:
        return factory()

def _resolve_predictor_by_tree_search() -> Any:
    root = _repo_root_from_this_file()
    # Allow importing modules relative to repo root (if they expect that)
    if root not in sys.path:
        sys.path.insert(0, root)

    candidates = []
    for dirpath, _, filenames in os.walk(root):
        # Skip venvs, git, and build folders
        low = dirpath.lower()
        if any(x in low for x in (".git", ".venv", "venv", "__pycache__", "build", "dist")):
            continue
        # Prefer eval*/ and src/eval*/ subtrees
        preferred = ("eval", "evals")
        score = 0
        if any(p in low.split(os.sep) for p in preferred):
            score += 2
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            # Heuristic: filenames containing our hints or named scaffold.py
            if fname == "scaffold.py" or any(h in fname.lower() for h in _PRED_FILE_HINTS):
                candidates.append((score, os.path.join(dirpath, fname)))

    # Sort by heuristic score (higher first) then shorter path
    candidates.sort(key=lambda x: (-x[0], len(x[1])))

    for _, path in candidates:
        try:
            mod = _import_from_path(f"_siml_autoload_{os.path.basename(path).replace('.py','')}", path)
        except Exception:
            continue
        # Find a usable predictor factory/callable
        for name in _PRED_NAMES:
            fn = getattr(mod, name, None)
            if callable(fn):
                # Return either a callable(a_seq,z_k,s_k) or an object with .predict(...)
                try:
                    sig = inspect.signature(fn)
                    if len(sig.parameters) == 0:
                        pred = fn()
                    else:
                        # We can't pass cfg safely here; assume no-arg factory
                        pred = fn()
                except Exception:
                    # If factory expects cfg or other args, just try calling with no args;
                    # if that fails, skip this candidate.
                    continue
                return pred
    raise RuntimeError(
        "Could not auto-locate a predictor in the repo. "
        "Expose one of the following in any eval/planner file: "
        f"{', '.join(_PRED_NAMES)}, or pass --predictor_entry module:path:func"
    )

def _call_predictor(predictor: Any, a_seq, z_k, s_k):
    if callable(predictor):
        return predictor(a_seq, z_k, s_k)
    if hasattr(predictor, "predict"):
        return predictor.predict(a_seq, z_k, s_k)
    raise TypeError("Predictor must be callable or implement .predict(a_seq, z_k, s_k)")

def make_base_energy_from_repo(cfg: Dict[str, Any]):
    predictor = (
        _resolve_predictor_from_entry(cfg)    # explicit
        or _resolve_predictor_by_tree_search()# auto
    )

    def base_energy(a_seq, z_k, s_k, z_g) -> float:
        z_pred = _call_predictor(predictor, a_seq, z_k, s_k)
        return _mean_abs(z_pred, z_g)

    return base_energy
