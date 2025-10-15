# src/siml/planner_hook.py
"""
Call your repository's existing MPC/CEM with our EnergyWithGate.

Resolution order:
1) If cfg['mpc_entry'] provided (e.g., "pkg.mod:run_mpc"), import & call it.
2) Search the repo tree for a file that defines one of:
     run_mpc(score_fn, cfg) | plan_with_goal(...) | run_planner(...)
   in eval*/ or src/eval*/ subdirs, scaffold.py or *plan*.py files preferred.
3) If nothing found: raise RuntimeError (no silent CEM fallback).
"""

from typing import Any, Dict, Callable, Optional
import importlib
import importlib.util
import os
import sys
import inspect

from siml.energy_gate import EnergyWithGate

# ---------------- utilities ----------------

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
    here = os.path.abspath(os.path.dirname(__file__))         # .../src/siml
    src_dir = os.path.abspath(os.path.join(here, ".."))       # .../src
    root = os.path.abspath(os.path.join(src_dir, ".."))       # repo root
    return root

# ---------------- MPC runner resolution ----------------

_RUN_NAMES = ("run_mpc", "plan_with_goal", "run_planner")
_RUN_FILE_HINTS = ("scaffold.py", "plan", "planner", "eval")

def _try_entrypoint(entry: str, score_fn: Callable, cfg: Dict[str, Any]) -> bool:
    try:
        fn = _load_entrypoint(entry)
    except Exception:
        return False
    try:
        fn(score_fn, cfg)               # positional
        return True
    except TypeError:
        try:
            fn(score_fn=score_fn, cfg=cfg)  # keyword
            return True
        except Exception:
            return False
    except Exception:
        return False

def _run_repo_mpc_by_tree_search(score_fn: Callable, cfg: Dict[str, Any]) -> bool:
    root = _repo_root_from_this_file()
    if root not in sys.path:
        sys.path.insert(0, root)

    candidates = []
    for dirpath, _, filenames in os.walk(root):
        low = dirpath.lower()
        if any(x in low for x in (".git", ".venv", "venv", "__pycache__", "build", "dist")):
            continue
        score = 0
        if any(p in low.split(os.sep) for p in ("eval", "evals")):
            score += 2
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if fname == "scaffold.py" or any(h in fname.lower() for h in _RUN_FILE_HINTS):
                candidates.append((score, os.path.join(dirpath, fname)))

    candidates.sort(key=lambda x: (-x[0], len(x[1])))

    for _, path in candidates:
        try:
            mod = _import_from_path(f"_siml_autoload_{os.path.basename(path).replace('.py','')}", path)
        except Exception:
            continue
        for name in _RUN_NAMES:
            fn = getattr(mod, name, None)
            if callable(fn):
                # Try positional then keyword calling conventions
                try:
                    fn(score_fn, cfg)
                    return True
                except TypeError:
                    try:
                        fn(score_fn=score_fn, cfg=cfg)
                        return True
                    except Exception:
                        continue
                except Exception:
                    continue
    return False

def run_mpc_with_energy(energy: EnergyWithGate, cfg: Dict[str, Any]):
    # 1) explicit entrypoint if provided
    entry = cfg.get("mpc_entry")
    if entry and _try_entrypoint(entry, energy.score, cfg):
        print(f"[SIML planner] Used explicit MPC entrypoint: {entry}")
        return

    # 2) search the repo tree for a runner
    if _run_repo_mpc_by_tree_search(energy.score, cfg):
        print("[SIML planner] Used MPC entrypoint discovered by repo tree search.")
        return

    # 3) no runner found → hard fail with guidance
    raise RuntimeError(
        "Could not locate an MPC/CEM runner in the repository.\n"
        "Expose one of: run_mpc(score_fn, cfg) | plan_with_goal(...) | run_planner(...)\n"
        "in any eval/planner file under eval*/ or src/eval*/ (e.g., scaffold.py), "
        "or pass --mpc_entry module.path:callable"
    )
