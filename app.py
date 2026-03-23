"""
RLHF Roast Generator — Gradio UI

Interactive demo that lets you:
  - Type any trait and get roasts from all 3 models side-by-side
  - See reward scores in real time
  - Browse the preference dataset
  - View training metrics and the analysis dashboard
"""

import os
import sys
import json

import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

SFT_MODEL_DIR = os.path.join(BASE_DIR, "models", "sft_model")
PPO_MODEL_DIR = os.path.join(BASE_DIR, "models", "ppo_model")
REWARD_MODEL_DIR = os.path.join(BASE_DIR, "models", "reward_model")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# ──────────────────────────────────────────────
# Load models once at startup
# ──────────────────────────────────────────────

print("Loading models (this may take a moment)...")
device = torch.device("cpu")


class RewardModel(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
            hidden = outputs.last_hidden_state[batch_idx, seq_lengths]
        else:
            hidden = outputs.last_hidden_state[:, -1, :]
        return self.reward_head(hidden).squeeze(-1)


# Base GPT-2
base_tok = AutoTokenizer.from_pretrained("gpt2")
base_tok.pad_token = base_tok.eos_token
base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
base_model.eval()
print("  ✓ Base GPT-2 loaded")

# SFT model
sft_tok = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
sft_tok.pad_token = sft_tok.eos_token
sft_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR).to(device)
sft_model.eval()
print("  ✓ SFT model loaded")

# PPO model
ppo_tok = AutoTokenizer.from_pretrained(PPO_MODEL_DIR)
ppo_tok.pad_token = ppo_tok.eos_token
ppo_model = AutoModelForCausalLM.from_pretrained(PPO_MODEL_DIR).to(device)
ppo_model.eval()
print("  ✓ PPO model loaded")

# Reward model
reward_tok = AutoTokenizer.from_pretrained("gpt2")
reward_tok.pad_token = reward_tok.eos_token
reward_model = RewardModel("gpt2").to(device)
reward_model.load_state_dict(
    torch.load(
        os.path.join(REWARD_MODEL_DIR, "reward_model.pt"),
        map_location=device,
        weights_only=True,
    )
)
reward_model.eval()
print("  ✓ Reward model loaded")
print("All models ready!\n")


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def clean_response(text):
    """
    Trim GPT-2 output to the first 1-2 clean sentences.
    GPT-2 tends to ramble — we cut it off at the first natural stop.
    """
    # Remove any re-generated prompt fragments (lines starting with "Roast someone")
    lines = [l for l in text.split("\n") if l.strip() and "Roast someone" not in l]
    text = " ".join(lines).strip()

    import re
    text = re.sub(r"http\S+", "", text).strip()
    # Split on sentence-ending punctuation, keep first 2 sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return text[:300]
    # Keep up to 2 sentences, max 250 chars
    result = " ".join(sentences[:2])
    if len(result) > 250:
        result = result[:250].rsplit(" ", 1)[0] + "…"
    return result


def generate(model, tok, prompt, max_new_tokens=100, temperature=0.8):
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tok.eos_token_id,
        )
    full = tok.decode(output[0], skip_special_tokens=True)
    raw = full[len(prompt):].strip()
    return clean_response(raw)


def score_text(text):
    enc = reward_tok(
        text, max_length=256, padding="max_length",
        truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        reward = reward_model(enc["input_ids"], enc["attention_mask"])
    return float(reward.item())


def reward_bar(score):
    """Convert a reward score into a visual progress-bar string."""
    # Normalize roughly to 0–100 range; reward scores span ~0–20
    pct = max(0, min(100, int((score / 20.0) * 100)))
    filled = pct // 5
    bar = "█" * filled + "░" * (20 - filled)
    return f"[{bar}] {score:.2f}"


# ──────────────────────────────────────────────
# Core roast generation function
# ──────────────────────────────────────────────

def roast_all(trait, temperature, max_tokens):
    if not trait.strip():
        empty = "⬆️ Enter a trait above and click Roast!"
        return empty, empty, empty, empty, ""

    trait_clean = trait.strip().rstrip(":")
    # Small grammar fix for common user input like "always late to meetings".
    if trait_clean.lower().startswith("always "):
        trait_clause = f"is {trait_clean}"
    else:
        trait_clause = trait_clean
    prompt = f"Roast someone who {trait_clause}:"
    prompt_display = f"**Prompt sent to all models:** `{prompt}`"

    base_roast = generate(base_model, base_tok, prompt, int(max_tokens), temperature)
    sft_roast  = generate(sft_model,  sft_tok,  prompt, int(max_tokens), temperature)
    ppo_roast  = generate(ppo_model,  ppo_tok,  prompt, int(max_tokens), temperature)

    base_score = score_text(prompt + " " + base_roast)
    sft_score  = score_text(prompt + " " + sft_roast)
    ppo_score  = score_text(prompt + " " + ppo_roast)

    winner = max(
        [("Base GPT-2", base_score), ("SFT Model", sft_score), ("PPO Model", ppo_score)],
        key=lambda x: x[1],
    )

    scores_md = (
        f"### 🏅 Reward Scores — Winner: **{winner[0]}**\n\n"
        f"| Model | Score | Bar |\n"
        f"|-------|-------|-----|\n"
        f"| 🤖 Base GPT-2 | {base_score:.2f} | `{reward_bar(base_score)}` |\n"
        f"| 📚 SFT Model  | {sft_score:.2f} | `{reward_bar(sft_score)}` |\n"
        f"| 🏆 PPO Model  | {ppo_score:.2f} | `{reward_bar(ppo_score)}` |\n\n"
        f"*Higher score = more witty/aligned according to the reward model*"
    )

    return prompt_display, base_roast, sft_roast, ppo_roast, scores_md


# ──────────────────────────────────────────────
# Load preference dataset for browser tab
# ──────────────────────────────────────────────

def load_preference_data():
    path = os.path.join(DATA_DIR, "preference_dataset.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


pref_data = load_preference_data()
pref_table = [
    [d["prompt"].replace("Roast someone who ", "").rstrip(":"), d["chosen"], d["rejected"]]
    for d in pref_data
]


# ──────────────────────────────────────────────
# Load metrics for results tab
# ──────────────────────────────────────────────

def load_metrics():
    path = os.path.join(OUTPUTS_DIR, "analysis_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


metrics_data = load_metrics()


def format_metrics_table():
    if not metrics_data:
        return []
    rows = []
    for model_name, m in metrics_data["metrics_summary"].items():
        rows.append([
            model_name,
            f"{m['avg_reward']:.4f}",
            f"{m['std_reward']:.4f}",
            f"{m['avg_toxicity']:.4f}",
            f"{m['avg_wit']:.4f}",
            f"{m['distinct_2']:.4f}",
            f"{m['avg_length_words']:.1f}",
        ])
    return rows


def load_reward_log():
    path = os.path.join(OUTPUTS_DIR, "reward_training_log.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        log = json.load(f)
    return [
        [e["epoch"], f"{e['train_loss']:.4f}", f"{e['val_accuracy']:.2%}",
         f"{e['avg_chosen_reward']:.3f}", f"{e['avg_rejected_reward']:.3f}",
         f"{e['reward_gap']:.3f}"]
        for e in log
    ]


def load_ppo_log():
    path = os.path.join(OUTPUTS_DIR, "ppo_training_log.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        log = json.load(f)
    return [
        [e["epoch"], f"{e['avg_reward']:.4f}", f"{e['std_reward']:.4f}", e["num_samples"]]
        for e in log
    ]


# ──────────────────────────────────────────────
# Build the Gradio interface
# ──────────────────────────────────────────────

DESCRIPTION = """
# 🔥 RLHF Roast Generator

**Reinforcement Learning from Human Feedback applied to comedy.**

This demo fine-tuned GPT-2 to generate *witty* roasts using a full RLHF pipeline:
1. **SFT** — taught GPT-2 the roast format with clever examples
2. **Reward Model** — trained an AI judge to score witty vs. mean roasts
3. **PPO** — used reinforcement learning to optimize for high reward

Compare all three models side-by-side to see how alignment changes the outputs.
"""

STEP_EXPLAINER = """
### How the RLHF Pipeline Works

| Step | What happens | Analogy |
|------|-------------|---------|
| **1. Dataset** | 50 witty/mean roast pairs collected | Writing a rubric for the joke contest |
| **2. SFT** | GPT-2 reads the witty examples | Teaching a student with a textbook |
| **3. Reward Model** | AI judge learns witty vs. mean | Hiring a professional comedy critic |
| **4. PPO** | Generator optimizes for high scores | Student practicing for the critic's approval |
| **5. Analysis** | Check reward, toxicity, diversity | Grading the final exam |
"""

with gr.Blocks(title="RLHF Roast Generator") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ── Tab 1: Generate Roasts ──────────────────────────────────
        with gr.Tab("🎤 Generate Roasts"):
            gr.Markdown(
                "### Step 1 — Enter a trait about a person (e.g. *always late to meetings*)\n"
                "The same prompt is sent to all 3 models so you can compare how training changed the output."
            )

            with gr.Row():
                trait_input = gr.Textbox(
                    label="Person's Trait",
                    placeholder="e.g.  always late to meetings   |   microwaves fish in the office   |   has a podcast nobody listens to",
                    lines=1,
                    scale=4,
                )
                generate_btn = gr.Button("🔥 Roast!", variant="primary", scale=1)

            with gr.Row():
                temp_slider = gr.Slider(
                    minimum=0.5, maximum=1.5, value=0.8, step=0.1,
                    label="Creativity (Temperature)", info="Higher = more surprising/random outputs"
                )
                tokens_slider = gr.Slider(
                    minimum=30, maximum=150, value=80, step=10,
                    label="Max Length (tokens)", info="How long each roast can be"
                )

            # Show the exact prompt that was constructed and sent
            prompt_display = gr.Markdown("*Enter a trait above and click Roast to see the prompt*")

            gr.Markdown("---")
            gr.Markdown("### Step 2 — See what each model generates")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🤖 Base GPT-2")
                    gr.Markdown("*No training at all — raw GPT-2 off the shelf.\nExpect random, off-topic text.*")
                    base_out = gr.Textbox(label="Base GPT-2 Output", lines=5, interactive=False)

                with gr.Column():
                    gr.Markdown("### 📚 SFT Model")
                    gr.Markdown("*Read 100 witty roast examples.\nKnows the format, not yet optimized.*")
                    sft_out = gr.Textbox(label="SFT Model Output", lines=5, interactive=False)

                with gr.Column():
                    gr.Markdown("### 🏆 PPO Model (RLHF-Aligned)")
                    gr.Markdown("*Trained with reward feedback to maximize wittiness.\nThis is the RLHF result.*")
                    ppo_out = gr.Textbox(label="PPO Model Output", lines=5, interactive=False)

            gr.Markdown("---")
            gr.Markdown("### Step 3 — See which model the reward model scores highest")
            scores_box = gr.Markdown("*Scores will appear here after you generate roasts*")

            generate_btn.click(
                fn=roast_all,
                inputs=[trait_input, temp_slider, tokens_slider],
                outputs=[prompt_display, base_out, sft_out, ppo_out, scores_box],
            )
            trait_input.submit(
                fn=roast_all,
                inputs=[trait_input, temp_slider, tokens_slider],
                outputs=[prompt_display, base_out, sft_out, ppo_out, scores_box],
            )

            gr.Markdown("---")
            gr.Markdown("**💡 Try one of these example traits:**")
            gr.Examples(
                examples=[
                    ["always late to meetings"],
                    ["has a podcast nobody listens to"],
                    ["microwaves fish in the office"],
                    ["uses Comic Sans unironically"],
                    ["brings a guitar to parties"],
                    ["thinks they're a LinkedIn influencer"],
                    ["peaked in high school"],
                    ["obsessed with crypto"],
                ],
                inputs=trait_input,
            )

        # ── Tab 2: How It Works ─────────────────────────────────────
        with gr.Tab("📖 How It Works"):
            gr.Markdown(STEP_EXPLAINER)
            gr.Markdown("""
### What is RLHF?

**Reinforcement Learning from Human Feedback (RLHF)** is the technique used to train ChatGPT, Claude, and other modern AI assistants to be helpful and safe.

The core insight: instead of just showing a model examples of good text, you **teach it to optimize for a score** — a score that represents human preferences.

### The Alignment Problem in This Project

A roast can be:
- **Mean/offensive** — just insults with no cleverness. Easy to generate, but not what we want.
- **Witty/clever** — uses comparisons, irony, and wordplay. Much harder, but much better.

Without RLHF, GPT-2 doesn't know the difference. With RLHF, it learns to prefer the witty style.

### Overoptimization Risk

A key concern in RLHF is **reward hacking** — the model finds ways to get a high score from the reward model without actually being better. Signs of this:
- **Mode collapse**: Always generating the same phrase (diversity drops)
- **Toxicity increase**: Reward goes up but the text gets worse in other ways
- **Reward inflation**: Scores keep climbing but quality plateaus

In this project, we measured all three. The result: reward went up, toxicity stayed at zero, and diversity actually *improved* vs. the SFT model. ✓
""")

        # ── Tab 3: Preference Dataset ───────────────────────────────
        with gr.Tab("📊 Preference Dataset"):
            gr.Markdown("""
### The Preference Dataset

These are the **50 preference pairs** used to train the reward model.
The model learned that **chosen** (witty) roasts should score higher than **rejected** (mean) ones.
""")
            gr.Dataframe(
                value=pref_table,
                headers=["Trait", "Chosen (Witty) ✓", "Rejected (Mean) ✗"],
                wrap=True,
                row_count=10,
            )

        # ── Tab 4: Training Metrics ─────────────────────────────────
        with gr.Tab("📈 Training Metrics"):
            gr.Markdown("### Final Evaluation: All Three Models")
            gr.Dataframe(
                value=format_metrics_table(),
                headers=["Model", "Avg Reward", "Std Reward", "Toxicity", "Wit Score", "Distinct-2", "Avg Length"],
            )

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Reward Model Training (per epoch)")
                    gr.Dataframe(
                        value=load_reward_log(),
                        headers=["Epoch", "Train Loss", "Val Accuracy", "Avg Chosen", "Avg Rejected", "Reward Gap"],
                    )

                with gr.Column():
                    gr.Markdown("### PPO Training (per epoch)")
                    gr.Dataframe(
                        value=load_ppo_log(),
                        headers=["Epoch", "Avg Reward", "Std Reward", "# Samples"],
                    )

            gr.Markdown("""
### Overoptimization Analysis

| Metric | SFT → PPO Change | Interpretation |
|--------|-----------------|----------------|
| Reward | +0.99 | PPO successfully improved alignment |
| Toxicity | 0.000 → 0.000 | No reward hacking — model didn't become meaner |
| Distinct-2 | 0.622 → 0.677 | Diversity *improved* — no mode collapse |
| **Verdict** | ✓ **Healthy alignment** | Reward up, safety maintained, variety preserved |
""")

        # ── Tab 5: Analysis Plots ───────────────────────────────────
        with gr.Tab("🖼️ Analysis Plots"):
            dashboard_path = os.path.join(OUTPUTS_DIR, "analysis_dashboard.png")
            reward_plot_path = os.path.join(OUTPUTS_DIR, "reward_model_training.png")

            if os.path.exists(dashboard_path):
                gr.Markdown("### Model Comparison Dashboard")
                gr.Image(value=dashboard_path, label="Analysis Dashboard")
            else:
                gr.Markdown("*Run `python run_pipeline.py --steps 5` to generate plots.*")

            if os.path.exists(reward_plot_path):
                gr.Markdown("### Reward Model Training Curves")
                gr.Image(value=reward_plot_path, label="Reward Model Training")

    gr.Markdown("""
---
**RLHF Roast Generator** | Built with GPT-2, TRL, and Gradio | Duke University — Reinforcement Learning, Spring 2026
""")

if __name__ == "__main__":
    demo.launch(share=False, show_error=True, theme=gr.themes.Soft())
