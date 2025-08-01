llm=$1
bs=$2
cuda=$3
export CUDA_VISIBLE_DEVICES=$cuda

echo "$llm-bnb-4bit"
python R1-Act_training_code.py --llm $llm-bnb-4bit --bs $bs


# 1.5B -> bs 16
# 7B, 8B -> bs 8
# 14B -> bs 8