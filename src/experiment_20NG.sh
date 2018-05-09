# Running with max sequence length 1000
# Using dataset 20NG
# Comparison of skip and baseline at 10k steps
# Batch size 16
python3 seq_classification.py --dataset 20NG --steps 10000 --cell_type baseline --model_dir 20NG_max1000_baseline
python3 seq_classification.py --dataset 20NG --steps 10000 --cell_type skip --model_dir 20NG_max1000_skip
