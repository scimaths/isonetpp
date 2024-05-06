cd ..

declare -A dataset_seeds=(
   ["aids"]="7366"
   ["mutag"]="4929"
   ["ptc_fm"]="7366"
   ["ptc_fr"]="7474"
   ["ptc_mm"]="7474"
   ["ptc_mr"]="7366"
)

gpus=(0 1 2 3 4 5)
overall_counter=0

for config_file in \
   "configs/rq7_efficiency/scoring=agg___tp=attention_pp=identity_when=post_K=8.yaml" \
   "configs/rq7_efficiency/scoring=agg___tp=attention_pp=identity_when=post_K=10.yaml" \
   "configs/rq7_efficiency/scoring=agg___tp=attention_pp=identity_when=post_K=12.yaml" \
; do
   for dataset in "${!dataset_seeds[@]}"; do
      seed="${dataset_seeds[$dataset]}"
      gpu_index=$(( (overall_counter) % ${#gpus[@]} ))

      CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
         --experiment_id rq7_efficiency \
         --experiment_dir experiments/ \
         --model_config_path $config_file \
         --dataset_name $dataset \
         --seed $seed \
         --dataset_size large \
         &

      ((overall_counter++))
      sleep 10s
   done
done
