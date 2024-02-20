cd ..

declare -A dataset_seeds=(
   ["aids"]="356"
   ["mutag"]="1704"
   ["ptc_fm"]="3133"
   ["ptc_fr"]="4929"
   ["ptc_mm"]="7366"
   ["ptc_mr"]="7762"
)

gpus=(0 1 5 6 7)
config_counter=0

for config_file in \
   "configs/rq1/scoring=agg___tp=masked_attention_pp=lrl_when=post.yaml" \
   "configs/rq1/scoring=attention_pp=lrl___tp=masked_attention_pp=lrl_when=post.yaml" \
   "configs/rq1/scoring=masked_attention_pp=lrl___tp=masked_attention_pp=lrl_when=post.yaml" \
   "configs/rq1/scoring=sinkhorn_pp=lrl___tp=masked_attention_pp=lrl_when=post.yaml" \
; do
   dataset_counter=0
   for dataset in "${!dataset_seeds[@]}"; do
      seed="${dataset_seeds[$dataset]}"
      gpu_index=$(( (dataset_counter + config_counter) % ${#gpus[@]} ))

      CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
         --experiment_id rq1 \
         --experiment_dir experiments/ \
         --model_config_path $config_file \
         --dataset_name $dataset \
         --seed $seed \
         --dataset_size large \
         &

      ((dataset_counter++))
      sleep 30s
   done
   ((config_counter++))
done
