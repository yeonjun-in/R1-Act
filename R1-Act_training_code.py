import torch
import argparse, requests, re, torch, os, json
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='R1-Act')
parser.add_argument('--llm', default='DeepSeek-R1-Distill-Llama-8B-bnb-4bit', choices=['DeepSeek-R1-Distill-Llama-8B-bnb-4bit', 'DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit', 'DeepSeek-R1-Distill-Qwen-7B-bnb-4bit', 'DeepSeek-R1-Distill-Qwen-14B-bnb-4bit', 'DeepSeek-R1-Distill-Qwen-32B-bnb-4bit'])
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--r', type=int, default=16)
parser.add_argument('--alpha', type=int, default=16)

args, _ = parser.parse_known_args()

print(args)

max_seq_length = 32768
dtype = None 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"unsloth/{args.llm}", 
    max_seq_length = max_seq_length,
    load_in_4bit=True,
    dtype=dtype,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = args.r, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = args.alpha,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 1995,
    use_rslora = False,  
    loftq_config = None, 
)


dataset = load_dataset("Yeonjun/R1-Act-train")

texts = []
for a in dataset:
    
    messages = [
       {"role": "system", "content": 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' if 'Qwen' in args.llm else 'You are Llama, created by Meta AI. You are a helpful assistant.'},
       {"role": "user", "content": a['instruction']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    text += a['response']
    texts.append(text)

dataset = dataset.add_column("text", texts)

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,  
    args=TrainingArguments(
        per_device_train_batch_size = args.bs,
        gradient_accumulation_steps = 1,

        num_train_epochs=15,

        learning_rate=1e-5,
        save_strategy="steps",
        save_steps=5000,
        warmup_steps=5,

        lr_scheduler_type="cosine",

        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=1e-4,

        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        logging_steps=1,
        seed=1995,

        output_dir=f"outputs/{args.method}_{args.llm}_r{args.r}_alpha{args.alpha}",

        report_to="none",
    ),
)


from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part=(
        "<｜User｜>"        
    ),
    response_part=(
        "<｜Assistant｜>"               
    )
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()