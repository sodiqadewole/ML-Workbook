from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from model_config import ModelConfig
from training_config import TrainingConfig
from datasets import Dataset
import torch

def create_qlora_model(model_config: ModelConfig,
                       tokenizer: AutoTokenizer,
                       compute_dtype: torch.dtype,
                       attn_implementation: str) -> AutoModelForCausalLM:
    '''Create and configure a QLoRA model for efficient fine-tuning'''
    print(f"Create QLoRA model: {model_config.model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading quantized model")

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map={"":0},
        attn_implementation=attn_implementation,
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(
        model,
        gradient_checkpointing_kwargs={'use_reentrant': True}
    )

    # configure tokenizer setting in model
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False # Disable cache for training

    print("QLoRA model prepared successfully")
    print("Model memory footprint: ~{model.get_memory_footprint() / 1e9:.1f}")
    
    return model


# QLoRA Training setup
def create_lora_config(training_config: TrainingConfig) -> LoraConfig:
    target_modules = [
        'k_proj', 'q_proj', 'v_proj', 'o_proj', # Attention projection
        'gate_proj', 'down_proj', 'up_proj' # Feed-forward projections
    ]

    lora_config = LoraConfig(
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        r=training_config.r,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=target_modules
    )
    return lora_config


# Training Execution
def train_qlora_model(dataset: Dataset,
                      model: AutoModelForCausalLM,
                      training_config: TrainingConfig,
                      compute_dtype: torch.dtype) -> SFTTrainer:

    peft_config = create_lora_config(training_config)

    # configure training arguments
    training_arguments = SFTConfig(
        output_dir=training_config.output_dir,
        optim='adamw_8bit',
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        log_level='info',
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
        learning_rate=training_config.learning_rate,
        fp16=compute_dtype==torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        max_steps = training_config.max_steps,
        warmup_ratio=training_config.warmup_ratio,
        lr_scheduler_type='linear',
        dataset_text_field='text',
        max_length=training_config.max_seq_length,
        remove_unused_columns=False,

        # Additional stability and performance settings
        dataloader_drop_last=True,      # Drop incomplete batches
        gradient_checkpointing=True,    # Enable gradient checkpoint
        save_total_limit=3,
        load_best_model_at_end=False,        
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_arguments
    )

    return trainer