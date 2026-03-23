"""
Step 3: Train a Reward Model on Preference Data

Trains a reward model that learns to score "witty" roasts higher than
"mean/offensive" roasts. Uses GPT-2 as backbone with a scalar value head.

The reward model is the core alignment signal — it encodes human preferences
about what makes a good roast (clever > cruel).
"""

import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
import numpy as np
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "reward_model")


class RewardModel(nn.Module):
    """GPT-2 backbone + linear head that outputs a scalar reward score."""

    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)
        # Freeze early layers, fine-tune later ones + head
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use last non-padding token's hidden state
        if attention_mask is not None:
            # Get index of last non-pad token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
            hidden = outputs.last_hidden_state[batch_idx, seq_lengths]
        else:
            hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(hidden).squeeze(-1)
        return reward


class PreferenceDataset(Dataset):
    """Dataset of (prompt+chosen, prompt+rejected) pairs for reward model."""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_text = f"{item['prompt']} {item['chosen']}"
        rejected_text = f"{item['prompt']} {item['rejected']}"

        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "pair_weight": torch.tensor(float(item.get("pair_weight", 1.0)), dtype=torch.float),
            "rejection_type": item.get("rejection_type", "unknown"),
        }


def preference_loss(chosen_rewards, rejected_rewards, pair_weight=None):
    """
    Bradley-Terry preference loss.
    We want: reward(chosen) > reward(rejected)
    Loss = -log(sigmoid(r_chosen - r_rejected))
    """
    losses = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards)
    if pair_weight is not None:
        losses = losses * pair_weight
    return losses.mean()


def main():
    print("=" * 60)
    print("Step 3: Reward Model Training")
    print("=" * 60)

    # Load preference data
    pref_dataset = load_from_disk(os.path.join(DATA_DIR, "preference_dataset"))
    pref_data = [pref_dataset[i] for i in range(len(pref_dataset))]
    print(f"Loaded {len(pref_data)} preference pairs")

    # Split into train/val
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(pref_data))
    split_idx = int(0.9 * len(pref_data))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    train_data = [pref_data[i] for i in train_idx]
    val_data = [pref_data[i] for i in val_idx]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RewardModel("gpt2").to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # Dataloaders
    train_dataset = PreferenceDataset(train_data, tokenizer)
    val_dataset = PreferenceDataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5,
        weight_decay=0.01,
    )

    # Training loop
    num_epochs = 4
    best_val_acc = 0
    training_log = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            pair_weight = batch["pair_weight"].to(device)

            chosen_rewards = model(chosen_ids, chosen_mask)
            rejected_rewards = model(rejected_ids, rejected_mask)

            loss = preference_loss(chosen_rewards, rejected_rewards, pair_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_chosen_rewards = []
        val_rejected_rewards = []
        val_by_type = {}

        with torch.no_grad():
            for batch in val_loader:
                chosen_ids = batch["chosen_input_ids"].to(device)
                chosen_mask = batch["chosen_attention_mask"].to(device)
                rejected_ids = batch["rejected_input_ids"].to(device)
                rejected_mask = batch["rejected_attention_mask"].to(device)
                rejection_types = batch["rejection_type"]

                c_rew = model(chosen_ids, chosen_mask)
                r_rew = model(rejected_ids, rejected_mask)

                correct_mask = c_rew > r_rew
                val_correct += correct_mask.sum().item()
                val_total += c_rew.size(0)
                val_chosen_rewards.extend(c_rew.cpu().numpy())
                val_rejected_rewards.extend(r_rew.cpu().numpy())
                for i, rej_type in enumerate(rejection_types):
                    if rej_type not in val_by_type:
                        val_by_type[rej_type] = {"correct": 0, "total": 0}
                    val_by_type[rej_type]["correct"] += int(correct_mask[i].item())
                    val_by_type[rej_type]["total"] += 1

        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_loss = np.mean(epoch_losses)
        avg_chosen = np.mean(val_chosen_rewards)
        avg_rejected = np.mean(val_rejected_rewards)
        reward_gap = avg_chosen - avg_rejected

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "avg_chosen_reward": float(avg_chosen),
            "avg_rejected_reward": float(avg_rejected),
            "reward_gap": float(reward_gap),
            "val_accuracy_by_rejection_type": {
                k: (v["correct"] / v["total"] if v["total"] else 0.0)
                for k, v in sorted(val_by_type.items())
            },
        }
        training_log.append(log_entry)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"Chosen: {avg_chosen:.3f} | "
            f"Rejected: {avg_rejected:.3f} | "
            f"Gap: {reward_gap:.3f}"
        )
        if val_by_type:
            by_type_str = ", ".join(
                [
                    f"{k}:{(v['correct'] / v['total'] if v['total'] else 0.0):.2%}"
                    for k, v in sorted(val_by_type.items())
                ]
            )
            print(f"  Val by rejection type -> {by_type_str}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "reward_model.pt"))
            tokenizer.save_pretrained(MODEL_DIR)

    # Save training log
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "outputs", "reward_training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to {MODEL_DIR}")
    print("Reward model training complete!")

    return training_log


if __name__ == "__main__":
    main()
