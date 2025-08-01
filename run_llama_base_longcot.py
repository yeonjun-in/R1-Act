import argparse, requests, re, torch, os, json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from copy import deepcopy
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import warnings
from utils import parse_jbbbehaviours, parse_wildchat
warnings.filterwarnings("ignore")

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='debug')
parser.add_argument('--llm', default='unsloth/Meta-Llama-3.1-8B-Instruct')
parser.add_argument('--dataset', default='')
parser.add_argument('--temperature', default=0)
parser.add_argument('--top_p', default=None)
parser.add_argument('--num_beams', default=1)
parser.add_argument('--max_new_tokens', default=512)
parser.add_argument('--output_dir', default='./preds/$data/$model/$llm/')

args, _ = parser.parse_known_args()
if 'checkpoint' in args.llm:
    save_file = os.path.join(args.output_dir.replace('$model', args.model).replace('$data', args.dataset).replace('$llm', '_'.join(args.llm.replace('outputs/', '').split('/'))), 'outputs.json')
else:
    save_file = os.path.join(args.output_dir.replace('$model', args.model).replace('$data', args.dataset).replace('$llm', args.llm.split('/')[-1]), 'outputs.json')

ensure_directory_exists(save_file)

print(args)

model_name = args.llm
max_seq_length = 32768
dtype = None 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"{args.llm}", 
    max_seq_length = max_seq_length,
    load_in_4bit=True,
    dtype=dtype,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

FastLanguageModel.for_inference(model)

if 'strongreject' == args.dataset:
    data = pd.read_csv('strongreject_dataset/strongreject_dataset.csv')['forbidden_prompt'].tolist()
elif 'wildjailbreak' == args.dataset:
    data = load_dataset('UWNSL/WildJailbreakEval', '250')['train']['instruction']
elif 'xstest' == args.dataset:
    data = pd.read_csv('xstest_prompts.csv')[lambda x: x['label'] == 'safe']['prompt'].tolist()
elif 'jbb' == args.dataset:
    data = parse_jbbbehaviours()


results = []
for i, query in enumerate(tqdm(data)):

    messages = [
       {"role": "system", "content": 'You are Llama, created by Meta AI. You are a helpful assistant.'},
       {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_length = model_inputs.input_ids.shape[1]
    with torch.inference_mode():
        output_ids = model.generate(**model_inputs,
                                       do_sample=True if args.temperature > 0 else False,
                                       temperature=args.temperature,
                                       top_p=args.top_p,
                                       num_beams=args.num_beams,
                                       max_new_tokens=1024,
                                       use_cache=True,
                                       )

    outputs = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()
    results.append({'prompt':query, 'response':outputs})
    

with open(save_file, 'w') as f:
    json.dump(results, f, indent=4)