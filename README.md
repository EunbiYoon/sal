# Beyond Frontier Action Distillation: Verifier-Guided Training for Small Agentic Models
## 📖 Overview

SARL trains language-model agents to act strategically across cooperative and competitive games. The repository provides the complete pipeline for preparing preference data, training Qwen2.5-3B LoRA adapters with Direct Preference Optimization (DPO), and reproducing Tables 1–7.

The pipeline includes three stages:

1. **Preference Data Construction:** Merge, deduplicate, and deterministically sample game trajectories into six training datasets.
2. **Strategic Agent Training:** Compare one untrained base model with six DPO/LoRA variants and optionally construct a merged adapter.
3. **Game Evaluation:** Evaluate in-distribution and held-out strategic games using the paper protocol.

### 🎮 Featured Games

- **Social dilemmas:** Prisoner's Dilemma and Stag Hunt
- **Coordination and competition:** Battle of the Sexes and Matching Pennies
- **Sequential interaction:** Negotiation and Tic-Tac-Toe
- **Held-out games:** Auction, Divide-the-Dollar, Beauty Contest, and one-stage IPD

---

## 🚀 Method

SARL uses preference pairs of the form `prompt / chosen / rejected` to optimize a 4-bit Qwen2.5-3B base model with DPO. The seven primary names are evaluation variants, not seven separate training jobs: `base` is the unchanged reference model and the other six are trained LoRA adapters.

The teacher is the oracle policy that produces preferred responses. The student is both the blind rollout policy and the model updated by LoRA/DPO; no separate blind-model setting is used.

| Variant | Training pairs | Description |
|---|---:|---|
| `base` | 0 | Unmodified Qwen2.5-3B reference; no LoRA training |
| `filter_on` | 388 | Filtered preference data |
| `filter_off` | 407 | Unfiltered preference data |
| `core` | 503 | Core A+β subset |
| `aux` | 613 | Auxiliary A+β subset |
| `all` | 1,338 | Full A+β set |
| `rw` | 1,749 | Reward-weighted extension |
| `merge` | — | Equal-weight merge of `aux` and `all` |

The variant-to-data mapping used by `train/train.sh` is:

```text
base       -> no dataset and no training
filter_on  -> data/paper/filter_on.jsonl
filter_off -> data/paper/filter_off.jsonl
core       -> data/paper/a_beta_core.jsonl
aux        -> data/paper/a_beta_aux.jsonl
all        -> data/paper/a_beta_all.jsonl
rw         -> data/paper/a_beta_rw.jsonl
merge      -> 0.5 * aux adapter + 0.5 * all adapter
```

---

## 🛠️ Installation

The project requires Python 3.10 or newer, an NVIDIA GPU, and CUDA 12.6. On the UMass cluster, create the environment once:

```bash
module load conda/latest cuda/12.6
conda create -n sal python=3.11 -y
conda activate sal

pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

cp .env.example .env
# Set HF_TOKEN in .env.
```

The Hugging Face account must have access to `Qwen/Qwen2.5-3B-Instruct`. The train and eval entrypoints load CUDA, Conda, and `.env` automatically.

---

## ⚙️ Configuration

All hyperparameters are managed in `.env`; training and evaluation commands need no arguments. The main settings are:

```dotenv
STUDENT_MODEL=Qwen/Qwen2.5-3B-Instruct
TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct

TRAIN_MAX_STEPS=100
TRAIN_MAX_STEPS_CORE=620
TRAIN_MAX_STEPS_AUX=760
TRAIN_MAX_STEPS_RW=1940
TRAIN_EPOCHS=10
TRAIN_NUM_GPUS=3
LEARNING_RATE=5e-5
DPO_BETA=0.1
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET=all-linear
MAX_LENGTH=12288
TRAIN_BATCH_SIZE=1
GRAD_ACCUM=8
WARMUP_RATIO=0.05
LR_SCHEDULER=cosine
LOGGING_STEPS=10
SAVE_STEPS=10
SAVE_TOTAL_LIMIT=100

EVAL_EPISODES=12
EVAL_SEED=42
EVAL_MAX_TOKENS=192
EVAL_VARIANTS=base,core
EVAL_GAMES=all
MERGE_ALPHA=0.5
```

`TRAIN_MAX_STEPS_<VARIANT>` takes precedence over the shared `TRAIN_MAX_STEPS`, which
in turn takes precedence over `TRAIN_EPOCHS`. Leave both step settings empty to train
that variant by epoch. Supported suffixes are `FILTER_ON`, `FILTER_OFF`, `CORE`, `AUX`,
`ALL`, and `RW`. See [.env.example](.env.example) for quantization, precision, CUDA, and
cache settings.

---

## ⚡ Training

Prepared preference datasets must exist under `data/paper/`.

```bash
./train/train.sh
```

If `RUN_ID` is omitted, the launcher always creates a new UTC timestamp such as `20260828_035102`; a stale `RUN_ID` in `.env` is ignored. Pass `RUN_ID=my_session` explicitly on the command line only when a stable session name is required.

By default, this trains `filter_on,filter_off,core,aux,all,rw`. `base` is not included because it has no adapter to train. This runs one worker per GPU. With `TRAIN_NUM_GPUS=3`, GPU 0 trains `filter_on → aux`, GPU 1 trains `filter_off → all`, and GPU 2 trains `core → rw`. After all workers succeed, it creates the merged adapter when both `aux` and `all` exist and stores everything under `runs/<session-id>/`. Every setting is read from `.env`; per-variant logs are written to `runs/<session-id>/train_logs/`.

Select one or more training variants with `TRAIN_VARIANTS`:

```bash
# One variant
TRAIN_NUM_GPUS=1 TRAIN_VARIANTS=core ./train/train.sh

# A subset
TRAIN_NUM_GPUS=3 TRAIN_VARIANTS=core,aux,all ./train/train.sh

# All six trainable variants (the default)
TRAIN_VARIANTS=filter_on,filter_off,core,aux,all,rw ./train/train.sh
```

For one variant with DistributedDataParallel across several GPUs:

```bash
DDP_NUM_GPUS=3 ./train/ddp_train.sh --variant core
```

### Training code layout

The `python -m train` entrypoint is implemented under `train/` and is grouped by the three A+β components:

```text
train/
├── solver/          # load solver-labelled prompt/chosen/rejected pairs
├── frontier/        # prepare the model and construct the DPO trainer
├── counterfactual/  # manage runs/checkpoints for filtered training data
├── cli.py           # resolve variant arguments and orchestrate training
└── __main__.py      # python -m train entrypoint
```

The folders describe boundaries in the LoRA training consumer. Preference-pair generation itself happens before this entrypoint; `train` consumes the already constructed JSONL files.

During DPO tokenization, game-generated blocks beginning with `[round continues]` and ending immediately before the next `<think>` block remain in the model input as context but receive label `-100`, so they do not contribute to chosen/rejected log-probability loss. Agent `<think>` and `<action>` tokens remain supervised.

### Monitoring

Track training with TensorBoard:

```bash
tensorboard --logdir runs --port 6006
```

---

## 🧪 Evaluation

After training finishes, run:

```bash
./eval/eval.sh
```

This automatically evaluates the latest training session in paper mode. Model, variants, games, episode count, seed, generation length, LoRA settings, and merge weight are read from `.env`. Results are written under `runs/<session-id>/eval/`.

For the separate 1,000-episode robustness protocol, see [Extended Evaluation](eval/EXTENDED_EVAL.md).

---

## 📊 Outputs

All artifacts from one experiment share a session directory:

```text
runs/<session-id>/
├── lora/
│   ├── filter_on/
│   ├── filter_off/
│   ├── core/
│   ├── aux/
│   ├── all/
│   ├── rw/
│   └── merge/
└── eval/
    ├── staging/
    ├── tables/
    ├── metrics/
    ├── logs/
    ├── result.md
    └── latex.md
```

Each trainable variant stores its resumable checkpoints under `adapter/checkpoint-*`. `runs/latest.json` points to the most recently completed training session.

---

## 🔧 Troubleshooting

- For `AutoProcessor`, Torch, or TorchVision import errors, reinstall the matched Torch/TorchVision versions listed above.
- For Hugging Face HTTP 401/403 errors, verify `HF_TOKEN` in `.env` and confirm model-license access.
