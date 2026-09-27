import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling
import torch

def main():
    model_id = "google/gemma-2-2b-it"
    out_dir = "adapters/chess-lora-2b"

    print(f"Loading tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading data...")
    dataset = load_dataset("json", data_files={"train": "data/chess/train.jsonl", "valid": "data/chess/valid.jsonl"})

    dataset = dataset.map(lambda x: {"text": x["prompt"] + x["completion"]})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading model {model_id} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    response_template = "Best move:"
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=60,
        save_strategy="steps",
        save_steps=60,
        learning_rate=1e-4,
        max_steps=1000,
        logging_steps=50,
        bf16=True,
        optim="paged_adamw_8bit",
        max_length=512,
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        args=args,
    )

    print("Starting training...")
    trainer.train()
    
    trainer.save_model(out_dir)
    print(f"Saved adapter to {out_dir}")

if __name__ == "__main__":
    main()
