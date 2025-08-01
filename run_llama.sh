export CUDA_VISIBLE_DEVICES=0

for dataset in jbb wildjailbreak strongreject xstest 
do
for model in base_longcot 
do
for backbone in DeepSeek-R1-Distill-Llama-8B-bnb-4bit 
do
for llm in outputs/R1-Act_{llm}_r16_alpha16
do
python run_llama_$model.py --model $model --dataset $dataset --llm $llm
done
done
done
done