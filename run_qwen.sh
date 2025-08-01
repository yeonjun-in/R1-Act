export CUDA_VISIBLE_DEVICES=0

for dataset in jbb wildjailbreak strongreject xstest 
do
for model in base_longcot 
do
for backbone in DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit DeepSeek-R1-Distill-Qwen-7B-bnb-4bit DeepSeek-R1-Distill-Qwen-14B-bnb-4bit
do
for llm in outputs/R1-Act_{args.llm}_r16_alpha16
do
python run_qwen_$model.py --model $model --dataset $dataset --llm $llm
done
done
done
done