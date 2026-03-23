# Roast Generator (Beginner-Friendly Explanation)

## What is this project?

This project is a small experiment that teaches a computer to write **funny roasts** (playful jokes about someone’s habits) instead of writing random or overly mean insults.

Think of it like this:

- At first, the computer says messy nonsense.
- Then we show it examples of better jokes.
- Then we give it a “judge” that scores jokes.
- Then the computer practices and tries to get better scores.

The goal is to make jokes that are:

- more clever,
- less toxic,
- and more relevant to the person’s trait (for example: “always late to meetings”).

---

## Why was this built?

This was built for a Reinforcement Learning challenge.

The challenge required:

1. collect preference data (better joke vs worse joke),
2. train a scoring model (“judge”),
3. run an improvement loop (PPO),
4. report metrics and discuss where the model still fails.

This project does all of those steps.

---

## The 5 steps in simple terms

### 1) Build examples

We create pairs like:

- **Good version**: witty, clever, less offensive
- **Bad version**: mean, lazy, or low-quality

So the system learns what “better” means.

### 2) Basic training

We take a small language model (GPT-2) and train it on the better examples so it learns roast style.

### 3) Train a judge

A separate model learns to score outputs.
It should give higher scores to the better roast in each pair.

### 4) Improvement loop (PPO)

The roast model generates jokes, the judge scores them, and the model updates itself to improve.

### 5) Evaluate results

We compare three versions:

- Base GPT-2 (no training)
- SFT model (basic training only)
- PPO model (trained with feedback loop)

We track quality, toxicity, and repetition/diversity.

---

## Important truth about current results

This project works as a **learning pipeline**, but roast quality is still inconsistent.

Sometimes outputs are funny, but sometimes they are confusing or don’t fully match the prompt.
That is expected because this is still a toy setup and optimization can over-focus on reward shortcuts.

So this is best viewed as:

- a successful RLHF workflow demo,
- not a production-ready comedy model.

---

## Latest run (March 23, 2026)

Latest evaluation on unseen prompts produced:

- **Base GPT-2**: on-topic `0.1176`, contradiction `0.0125`, distinct-2 `0.6833`
- **SFT**: on-topic `0.9879`, contradiction `0.0000`, distinct-2 `0.2840`
- **PPO**: on-topic `0.9900`, contradiction `0.0000`, distinct-2 `0.0894`

What this means:

- Relevance improved a lot from base model to trained models.
- Contradictions were reduced to near-zero.
- PPO introduced **mode collapse** (too repetitive), so diversity and wit dropped.

So the current version is better at staying on topic, but still needs stronger anti-repetition controls during RL training.

---

## How to run it (minimal instructions)

From the project folder:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
venv/bin/python run_pipeline.py
venv/bin/python app.py
```

Then open the local URL shown in terminal.

---

## What is included vs not included on GitHub

GitHub includes code only.

GitHub does **not** include large generated folders:

- `data/`
- `models/`
- `outputs/`
- `venv/`

Those are created on your machine when you run the pipeline.

---

## One-line summary

This project teaches a small model to move from random/mean text toward more witty roast-style text using a human-preference training loop, while showing where alignment still falls short.
