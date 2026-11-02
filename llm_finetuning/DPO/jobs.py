# dpo_training.py
# /// script
# dependencies = [
#     "trl[dpo]>=0.7.0",
#     "transformers>=4.36.0", 
#     "datasets>=2.14.0",
#     "accelerate>=0.24.0",
#     "torch>=2.0.0"
# ]
# ///

from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    # Load preference dataset
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    
    # Take a reasonable subset for training
    train_dataset = dataset.select(range(10000))
    
    # Load SmolLM3-3B model (pre-trained with SFT)
    model_name = "HuggingFaceTB/SmolLM3-3B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure DPO training
    training_args = DPOConfig(
        # Core DPO parameters
        beta=0.1,                           # Preference optimization strength
        max_prompt_length=512,              # Maximum prompt length
        max_length=1024,                    # Maximum total sequence length
        
        # Training configuration
        learning_rate=5e-7,                 # Lower than SFT for stability
        per_device_train_batch_size=2,      # Adjust for GPU memory
        gradient_accumulation_steps=8,      # Effective batch size = 16
        max_steps=1000,                     # Sufficient for good alignment
        
        # Optimization
        warmup_steps=100,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,        # Memory efficiency
        bf16=True,                          # Mixed precision
        
        # Logging and saving
        logging_steps=50,
        save_steps=250,
        output_dir="./smollm3-dpo-aligned",
        
        # Hub integration
        push_to_hub=False,
        # hub_model_id="./smollm3-dpo-aligned",  # Change this!
        report_to="trackio",
        
        # Remove unused columns for cleaner training
        remove_unused_columns=False,
    )
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Start training
    print("Starting DPO training...")
    trainer.train()
    
    print("Training completed! Model saved and pushed to Hub.")

if __name__ == "__main__":
    main()