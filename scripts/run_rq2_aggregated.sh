cd ..

# 1704, 4929, 7366, 7474, 7762

# declare -A dataset_seeds=(
#    ["aids"]="7474"
#    ["mutag"]="7474"
#    ["ptc_fm"]="4929"
#    ["ptc_fr"]="7366"
#    ["ptc_mm"]="7762"
#    ["ptc_mr"]="7366"
# )

# declare -A dataset_seeds=(
#    ["aids"]="1704"
#    ["mutag"]="1704"
#    ["ptc_fm"]="1704"
#    ["ptc_fr"]="1704"
#    ["ptc_mm"]="1704"
#    ["ptc_mr"]="1704"
# )

# declare -A dataset_seeds=(
#    ["aids"]="4929"
#    ["mutag"]="4929"
#    ["ptc_fm"]="7366"
#    ["ptc_fr"]="4929"
#    ["ptc_mm"]="4929"
#    ["ptc_mr"]="4929"
# )

declare -A dataset_seeds=(
   ["aids"]="7366"
   ["mutag"]="7366"
   ["ptc_fm"]="7474"
   ["ptc_fr"]="7474"
   ["ptc_mm"]="7366"
   ["ptc_mr"]="7474"
)

gpus=(0 1 2)
overall_counter=0

for config_file in \
   "configs/rq2_aggregated/scoring=agg___tp=attention_pp=lrl_when=post.yaml" \
   "configs/rq2_aggregated/scoring=agg___tp=masked_attention_pp=lrl_when=post.yaml" \
   "configs/rq2_aggregated/scoring=agg___tp=sinkhorn_pp=lrl_when=post.yaml" \
; do
   for dataset in "${!dataset_seeds[@]}"; do
      seed="${dataset_seeds[$dataset]}"
      gpu_index=$(( (overall_counter) % ${#gpus[@]} ))

      CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
         --experiment_id rq2_aggregated \
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
