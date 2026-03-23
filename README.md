# RLHF Roast Generator (GPT-2 + Preference Model + PPO)

This project implements a full toy RLHF loop for roast generation:

1. Collect preference data (`chosen` witty roast vs `rejected` mean roast)
2. Supervised fine-tune GPT-2 on witty examples (SFT)
3. Train a reward model on preference pairs
4. Run PPO to align the policy toward high reward
5. Analyze reward/toxicity/diversity/wit and inspect failure modes

The target behavior is: funny, clever, on-topic roasts that avoid lazy toxicity.

## Project Structure

- `scripts/generate_dataset.py`: builds SFT, preference, and PPO prompt datasets
- `scripts/sft_train.py`: SFT training
- `scripts/reward_model.py`: reward model training (pairwise preference loss)
- `scripts/ppo_train.py`: manual PPO loop with KL regularization
- `scripts/analysis.py`: quantitative comparison + overoptimization checks
- `app.py`: Gradio UI for side-by-side model comparison
- `run_pipeline.py`: end-to-end runner

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Full pipeline:

```bash
venv/bin/python run_pipeline.py
```

Only analysis (after models already exist):

```bash
venv/bin/python run_pipeline.py --steps 5
```

Launch UI:

```bash
venv/bin/python app.py
```

## Latest Results (Run on March 23, 2026)

From `outputs/analysis_metrics.json`:

| Model | Avg Reward | Toxicity | Wit Proxy | Distinct-2 | Avg Length |
|---|---:|---:|---:|---:|---:|
| Base GPT-2 | -1.3751 | 0.0000 | 0.0190 | 0.8377 | 54.4 |
| SFT Model | -1.1202 | 0.0000 | 0.1333 | 0.6989 | 67.2 |
| PPO Model | -0.2328 | 0.0000 | 0.1524 | 0.6856 | 69.1 |

Alignment deltas (PPO vs SFT):

- Reward: `+0.8874`
- Toxicity: `+0.0000`
- Distinct-2: `-0.0133`
- Wit proxy: `+0.0190`

Interpretation:

- PPO improved the learned reward objective.
- Toxicity remained flat (good).
- Diversity stayed close to SFT.
- Reward and wit proxy improved, toxicity stayed flat, but absolute reward remains negative and text quality is still inconsistent.

## Misalignment / Overoptimization Notes

This project intentionally tracks overoptimization risks:

- Reward increases that do not correspond to better human-perceived quality
- Repetition / mode collapse
- Toxicity increases

In current results, reward improves without toxicity increase, but wit-quality remains inconsistent. This is a realistic RLHF failure mode on small-data toy setups.

## Outputs

Generated artifacts:

- `models/sft_model/`
- `models/reward_model/`
- `models/ppo_model/`
- `outputs/reward_training_log.json`
- `outputs/ppo_training_log.json`
- `outputs/analysis_metrics.json`
- `outputs/analysis_dashboard.png`
- `outputs/reward_model_training.png`
