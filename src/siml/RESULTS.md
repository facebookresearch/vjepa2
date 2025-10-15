
# Results
## Reaching (H=1, CEM 128×5, L1-ball r=0.075)

| Setting                  | Final err (m) | Monotonicity | Gate accept | Steps≤4 cm | Latency (ms) |
|--------------------------|---------------|--------------|-------------|------------|--------------|
| JEPA-only (baseline)     | ~0.14         | ~0.23        | –           | ~1%        | ~1,040       |
| FEP-gate τ=0.060, λ=0.5  | **0.103**     | **0.370**    | 0.816       | ~1.7%      | ~1,084       |
| FEP-gate τ=0.073, λ=0.5  | 0.117         | 0.277        | 0.839       | ~1.0%      | ~1,083       |
| FEP-gate τ=0.085, λ=0.5  | 0.128         | 0.244        | 0.982       | ~1.3%      | ~1,093       |

λ sweep @ τ=0.073:
- λ=0.1 → 0.1172 m, accept 0.836
- λ=0.5 → 0.1170 m, accept 0.838
- λ=1.0 → 0.1169 m, accept 0.834

**Takeaway:** modestly stricter τ (0.060) keeps acceptance ~0.82 and gives the best accuracy & progress.
