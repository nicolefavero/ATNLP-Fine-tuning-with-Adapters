from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model

# 1) Load T5 model and tokenizer
model_name_or_path = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

# 2) Create a LoRA configuration
#    - The 'target_modules' must match layer names in T5 where you want to inject LoRA.
#    - Below are example names for T5's attention projections ('q', 'k', 'v', 'o').
#    - Adjust as needed depending on the exact T5 variant and layer naming.
lora_config = LoraConfig(
    r=8,                          # Rank of the LoRA update matrices
    lora_alpha=32,                # Scaling factor
    lora_dropout=0.05,            # Dropout to apply to LoRA layers
    bias="none",                  # Typically "none" or "lora_only"
    target_modules=["q", "k", "v", "o"],  # Potential T5 attention projection names
    task_type="SEQ_2_SEQ_LM",     # T5 is a seq2seq model
)

# 3) Wrap the T5 model with the LoRA configuration
model = get_peft_model(model, lora_config)

# 4) Set model to train mode
model.train()

print("LoRA added and model is ready for training!")
