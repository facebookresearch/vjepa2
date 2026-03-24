# src/siml/adapters.py
"""
Local adapters so you can point SIML to the real V-JEPA 2 predictor & MPC
without changing the main repo.
"""
from evals.scaffold import make_predictor as _repo_make_predictor
from evals.scaffold import run_mpc       as _repo_run_mpc

def _resolve_predictor_factory():
    from evals.scaffold import make_predictor as _repo_make_predictor
    return _repo_make_predictor

def _resolve_mpc_runner():
    from evals.scaffold import run_mpc as _repo_run_mpc
    return _repo_run_mpc

# --------- SIML-facing entrypoints (do not edit) --------- #
def make_predictor(cfg):
    """
    Return either:
      - callable(a_seq, z_k, s_k) -> z_pred
      - or an object with .predict(a_seq, z_k, s_k)
    """
    factory = _resolve_predictor_factory()
    try:
        return factory(cfg)
    except TypeError:
        return factory()

def run_mpc(score_fn, cfg):
    """
    Call your existing planner. It should internally call `score_fn(a_seq, z_k, s_k, z_g)`
    when ranking candidates. Keep CEM settings/horizon identical to the paper.
    """
    runner = _resolve_mpc_runner()
    try:
        return runner(score_fn, cfg)             # positional
    except TypeError:
        return runner(score_fn=score_fn, cfg=cfg)  # keyword
