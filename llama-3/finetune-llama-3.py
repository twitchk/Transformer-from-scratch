# finetune-llama-3.py
# Translated from Chinese to English with G Translate
# More models at https://huggingface.co/unsloth

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

max_seq_length = 2048  #You can customize to any window length, and the model window size is automatically scaled according to the RoPE encoding.
dtype = None  #Set to None to automatically obtain. Currently Float16 supports the following GPU types: Tesla T4, V100; Bfloat16 supports the following GPU types: Ampere+
load_in_4bit = True  # Use 4-bit quantization to reduce memory usage. Can be set to False.

# Call unsloth to pre-quantize the 4-bit model
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",  # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",  # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit",  # [NEW] 15 Trillion token Llama-3
]

# Call unsloth to pre-quantize the 4-bit model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,  # Load the model in 4bit
)

# Below are the default parameters for the PEFT model, you can adjust them as needed.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # The special symbol EOS_TOKEN must be added, otherwise the generation will loop infinitely.


def formatting_prompts_func(examples):
    instructions = examples["instruction_zh"]
    inputs = examples["input_zh"]
    outputs = examples["output_zh"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN  # The special symbol EOS_TOKEN must be added, otherwise the generation will loop infinitely.
        texts.append(text)
    return {"text": texts, }


# Loading data from a dataset
from datasets import load_dataset

# You can use this dataset for experimentation:
# https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True, )

# Below are the default parameters for the PEFT model, you can adjust them as needed.
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # It can increase the training speed of small context windows by more than 5 times
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=500,  # Fine-tune the number of cycles
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=1337,
        output_dir="outputs",
    ),
)

# Training the model
trainer_stats = trainer.train()

# inference
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "You answer questions in Chinese",  # instruction
            "How do plants breathe?",  # input
            "",  # output - leave this blank for generation!
        )
    ], return_tensors="pt").to("cuda")


# Generate text using Stream
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

# Saving the fine-tuned model
model.save_pretrained("llama-3-zh_lora")  # Save in local folder llama-3-zh_lora
