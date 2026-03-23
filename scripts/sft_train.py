"""
Step 2: Supervised Fine-Tuning (SFT) of GPT-2 on Roast Data

Fine-tunes GPT-2 on the witty roast dataset so it learns the format
and style of roast generation before RLHF alignment.
"""

import os
import sys

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "sft_model")


def main():
    print("=" * 60)
    print("Step 2: Supervised Fine-Tuning (SFT) on Roast Data")
    print("=" * 60)

    # Load dataset
    dataset = load_from_disk(os.path.join(DATA_DIR, "sft_dataset"))
    print(f"Loaded {len(dataset)} training examples")

    # Load model and tokenizer
    model_name = "gpt2"
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model parameters: {model.num_parameters():,}")

    # SFT Training config
    output_dir = os.path.join(BASE_DIR, "outputs", "sft_training")
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=20,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=False,  # MPS/CPU don't support fp16 well
        bf16=False,
        use_cpu=True,
        report_to="none",  # Set to "wandb" if you want logging
        max_length=256,
        dataset_text_field="text",
        seed=42,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting SFT training...")
    train_result = trainer.train()

    # Save model
    print(f"\nSaving SFT model to {MODEL_DIR}")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Log metrics
    metrics = train_result.metrics
    print("\n--- SFT Training Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Quick generation test — reload from saved checkpoint to avoid device issues
    print("\n--- Quick Generation Test ---")
    gen_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    gen_model.eval()
    test_prompts = [
        "Roast someone who always late to meetings:",
        "Roast someone who has a podcast nobody listens to:",
        "Roast someone who microwaves fish in the office:",
    ]
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")

    print("\nSFT training complete!")
    return metrics


if __name__ == "__main__":
    main()
