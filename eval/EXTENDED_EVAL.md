# Extended 1,000-episode evaluation

This runner reuses the paper evaluator and adds interruption-safe episode CSVs,
summary statistics, paired comparisons, and plots. It does not modify adapters.

Preflight (no model inference):

```bash
PYTHONHASHSEED=0 python -m eval.extended_eval --list \
  --lora-dir runs/<session_id>/lora \
  --out-dir results/extended_eval_1000
```

When the requested checkpoints and the base model are locally cached, run:

```bash
PYTHONHASHSEED=0 python -m eval.extended_eval \
  --episodes 1000 --seed 42 \
  --lora-dir runs/<session_id>/lora \
  --variants base,filter_on,filter_off,core,aux,all,rw,merge \
  --out-dir results/extended_eval_1000
```

The fixed `PYTHONHASHSEED` is required because the original evaluator derives
seeds with Python's `hash(game)`. Re-running the same command resumes from
`episode_rewards.csv`. To rebuild statistics and plots without inference:

```bash
python -m eval.extended_eval --analyse-only \
  --out-dir results/extended_eval_1000
```
