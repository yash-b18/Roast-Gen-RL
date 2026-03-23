"""
Step 4: PPO Alignment Training (Manual Implementation)

Implements Proximal Policy Optimization from scratch to align the SFT-tuned
GPT-2 toward generating witty roasts (high reward) while staying close to
the SFT reference policy (KL penalty prevents reward hacking).

RLHF Loop:
1. Generate roast completions from the policy model
2. Score them with the trained reward model
3. Compute advantages using the reward signal
4. Update policy with clipped PPO objective + KL penalty

Note: TRL v0.29+ removed PPOTrainer, so this is a clean manual implementation
that makes the PPO mechanics fully transparent.
"""

import os
import json
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_from_disk
import numpy as np


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
SFT_MODEL_DIR = os.path.join(BASE_DIR, "models", "sft_model")
REWARD_MODEL_DIR = os.path.join(BASE_DIR, "models", "reward_model")
PPO_MODEL_DIR = os.path.join(BASE_DIR, "models", "ppo_model")


class RewardModel(nn.Module):
    """Same architecture as training — loads the trained weights."""

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


class ValueHead(nn.Module):
    """Value head on top of GPT-2 for estimating state values (critic)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        return self.linear(hidden_states).squeeze(-1)


def get_log_probs(model, input_ids, attention_mask):
    """Compute per-token log probabilities under the model."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # Shift: predict next token
    target_ids = input_ids[:, 1:]  # Shifted targets
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    # Mask padding
    mask = attention_mask[:, 1:]
    return token_log_probs * mask


def compute_kl_divergence(policy_logprobs, ref_logprobs, mask):
    """Compute KL(policy || reference) per token, then average."""
    kl = policy_logprobs - ref_logprobs  # Approximation: sum of log-ratio
    kl = kl * mask
    return kl.sum() / mask.sum()


def ppo_step(
    policy_model,
    ref_model,
    value_head,
    optimizer,
    input_ids,
    attention_mask,
    prompt_lengths,
    rewards,
    clip_eps=0.2,
    kl_coef=0.1,
    vf_coef=0.1,
    ppo_epochs=4,
):
    """
    Perform one PPO update step.

    Args:
        policy_model: The model being optimized
        ref_model: Frozen reference (SFT) model for KL computation
        value_head: Critic network
        optimizer: Optimizer for policy + value head
        input_ids: Full sequences (prompt + response) [B, T]
        attention_mask: Attention mask [B, T]
        prompt_lengths: Length of prompt portion for each sample [B]
        rewards: Scalar reward for each sample [B]
        clip_eps: PPO clipping epsilon
        kl_coef: KL penalty coefficient
        vf_coef: Value function loss coefficient
        ppo_epochs: Number of PPO optimization epochs per batch
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Create response mask (1 for response tokens, 0 for prompt tokens)
    response_mask = torch.zeros_like(attention_mask, dtype=torch.float)
    for i in range(batch_size):
        response_mask[i, prompt_lengths[i] :] = attention_mask[i, prompt_lengths[i] :]

    # Compute old log probs and values (detached, no gradient)
    with torch.no_grad():
        old_log_probs = get_log_probs(policy_model, input_ids, attention_mask)

        # Get hidden states for value estimation
        outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
        old_values = value_head(outputs.hidden_states[-1][:, :-1, :] if hasattr(outputs, 'hidden_states') and outputs.hidden_states else outputs.logits[:, :-1, :].mean(-1).unsqueeze(-1).expand(-1, -1, policy_model.config.hidden_size))

        ref_log_probs = get_log_probs(ref_model, input_ids, attention_mask)

    # Compute per-token rewards: KL penalty + terminal reward
    # The reward is assigned to the last response token
    response_mask_shifted = response_mask[:, 1:]  # Align with log_probs shift
    token_rewards = torch.zeros_like(old_log_probs)
    kl_per_token = old_log_probs - ref_log_probs
    token_rewards -= kl_coef * kl_per_token  # KL penalty on each token

    # Add the scalar reward to the last response token
    for i in range(batch_size):
        resp_indices = response_mask_shifted[i].nonzero(as_tuple=True)[0]
        if len(resp_indices) > 0:
            token_rewards[i, resp_indices[-1]] += rewards[i]

    # Compute advantages using GAE (simplified: no discount since single-step reward)
    # For simplicity, advantage = reward - value
    advantages = token_rewards - old_values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO optimization
    total_policy_loss = 0
    total_value_loss = 0
    total_kl = 0

    for _ in range(ppo_epochs):
        new_log_probs = get_log_probs(policy_model, input_ids, attention_mask)

        # Policy loss with clipping
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * response_mask_shifted).sum() / response_mask_shifted.sum()

        # Value loss (simplified)
        outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use logits mean as a simple value proxy
        new_values = value_head(outputs.logits[:, :-1, :].mean(-1).unsqueeze(-1).expand(-1, -1, policy_model.config.hidden_size))
        value_loss = F.mse_loss(new_values, token_rewards.detach())

        # Total loss
        loss = policy_loss + vf_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy_model.parameters()) + list(value_head.parameters()), 1.0
        )
        optimizer.step()

        # Track KL
        with torch.no_grad():
            kl = compute_kl_divergence(
                new_log_probs, ref_log_probs, response_mask_shifted
            )

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_kl += kl.item()

    return {
        "policy_loss": total_policy_loss / ppo_epochs,
        "value_loss": total_value_loss / ppo_epochs,
        "kl_divergence": total_kl / ppo_epochs,
    }


def main():
    print("=" * 60)
    print("Step 4: PPO Alignment Training (Manual Implementation)")
    print("=" * 60)

    device = torch.device("cpu")  # Use CPU for stability with manual PPO
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load policy model (SFT checkpoint)
    print("Loading SFT model as policy...")
    policy_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR).to(device)
    policy_model.train()

    # Freeze all but the last 2 transformer blocks + lm_head for efficiency
    for name, param in policy_model.named_parameters():
        param.requires_grad = False
    # Unfreeze last 2 blocks and lm_head
    for name, param in policy_model.named_parameters():
        if any(
            k in name
            for k in ["transformer.h.10", "transformer.h.11", "transformer.ln_f", "lm_head"]
        ):
            param.requires_grad = True

    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy_model.parameters())
    print(f"Policy params: {total:,} total, {trainable:,} trainable")

    # Load reference model (frozen copy of SFT)
    print("Loading reference model (frozen SFT)...")
    ref_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load reward model
    print("Loading reward model...")
    reward_model = RewardModel("gpt2").to(device)
    reward_model.load_state_dict(
        torch.load(
            os.path.join(REWARD_MODEL_DIR, "reward_model.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    reward_model.eval()

    # Value head (critic)
    value_head = ValueHead(policy_model.config.n_embd).to(device)

    # Load prompts
    ppo_dataset = load_from_disk(os.path.join(DATA_DIR, "ppo_prompts"))
    prompts = [ppo_dataset[i]["prompt"] for i in range(len(ppo_dataset))]
    print(f"Loaded {len(prompts)} prompts")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, policy_model.parameters()))
        + list(value_head.parameters()),
        lr=1e-5,
        weight_decay=0.01,
    )

    # Training loop
    num_epochs = 3
    batch_size = 4
    max_new_tokens = 60
    kl_coef = 0.15
    clip_eps = 0.2
    training_log = []

    print(f"\nStarting PPO training: {num_epochs} epochs, batch_size={batch_size}")
    print(f"KL coef: {kl_coef}, Clip eps: {clip_eps}")

    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_kl = []
        epoch_examples = []
        np.random.seed(epoch)
        indices = np.random.permutation(len(prompts))

        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_indices = indices[batch_start:batch_end]
            batch_prompts = [prompts[i] for i in batch_indices]
            actual_batch_size = len(batch_prompts)

            # Tokenize prompts
            prompt_encodings = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1).tolist()

            # Generate responses
            policy_model.eval()
            with torch.no_grad():
                gen_outputs = policy_model.generate(
                    **prompt_encodings,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            policy_model.train()

            # Pad all to same length
            max_len = gen_outputs.shape[1]
            attention_mask = torch.ones_like(gen_outputs)
            for i in range(actual_batch_size):
                # Mask padding tokens
                pad_mask = gen_outputs[i] == tokenizer.pad_token_id
                # Only mask leading pads (from prompt padding)
                for j in range(gen_outputs.shape[1]):
                    if gen_outputs[i, j] != tokenizer.pad_token_id:
                        break
                    attention_mask[i, j] = 0

            # Decode and score with reward model
            full_texts = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            reward_encodings = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            with torch.no_grad():
                rewards = reward_model(
                    reward_encodings["input_ids"], reward_encodings["attention_mask"]
                )

            epoch_rewards.extend(rewards.tolist())

            # Log examples
            for p, ft, r in zip(batch_prompts, full_texts, rewards.tolist()):
                response = ft[len(p):].strip()
                epoch_examples.append({"prompt": p, "response": response, "reward": r})

            # PPO step
            prompt_lengths_tensor = [int(pl) for pl in prompt_lengths]
            stats = ppo_step(
                policy_model=policy_model,
                ref_model=ref_model,
                value_head=value_head,
                optimizer=optimizer,
                input_ids=gen_outputs,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths_tensor,
                rewards=rewards,
                clip_eps=clip_eps,
                kl_coef=kl_coef,
                ppo_epochs=2,
            )
            epoch_kl.append(stats["kl_divergence"])

        avg_reward = np.mean(epoch_rewards)
        std_reward = np.std(epoch_rewards)
        avg_kl = np.mean(epoch_kl) if epoch_kl else 0

        log_entry = {
            "epoch": epoch + 1,
            "avg_reward": float(avg_reward),
            "std_reward": float(std_reward),
            "avg_kl": float(avg_kl),
            "num_samples": len(epoch_rewards),
            "examples": epoch_examples[:3],
        }
        training_log.append(log_entry)

        print(
            f"\nEpoch {epoch+1}/{num_epochs} | "
            f"Avg Reward: {avg_reward:.4f} (±{std_reward:.4f}) | "
            f"Avg KL: {avg_kl:.4f} | "
            f"Samples: {len(epoch_rewards)}"
        )
        for ex in epoch_examples[:2]:
            print(f"  [{ex['reward']:.3f}] {ex['prompt'][:50]}... -> {ex['response'][:80]}...")

    # Save PPO model
    print(f"\nSaving PPO-aligned model to {PPO_MODEL_DIR}")
    os.makedirs(PPO_MODEL_DIR, exist_ok=True)
    policy_model.save_pretrained(PPO_MODEL_DIR)
    tokenizer.save_pretrained(PPO_MODEL_DIR)

    # Save training log
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "outputs", "ppo_training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print("\nPPO training complete!")
    return training_log


if __name__ == "__main__":
    main()
