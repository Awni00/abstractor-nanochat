# Hadamard Dual Attention Sanity Run

Use this tiny run as a smoke test for the `hadamard_dual` integration before larger experiments.

## Command

```bash
python -m scripts.base_train \
  --depth=4 \
  --max-seq-len=128 \
  --device-type=cpu \
  --device-batch-size=1 \
  --total-batch-size=256 \
  --num-iterations=20 \
  --eval-every=-1 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1 \
  --attention-impl=hadamard_dual \
  --dual-rel-head-proportion=0.5 \
  --use-residual-augmentation \
  --residual-stride=2 \
  --residual-gate-channels=32
```

## Acceptance Criteria

- Run completes without exceptions.
- Training loss trends downward over the short run.
- Checkpoint save/load path (when enabled) preserves the new config fields.

## Notes

- `hadamard_dual` intentionally does not support KV cache in this phase.
- Keep `--sample-every=-1` during this sanity run to avoid generation code paths that require KV cache.
