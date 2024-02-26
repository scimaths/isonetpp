cd ..

declare -A dataset_seeds=(
   ["aids"]="7474"
   ["mutag"]="7474"
   ["ptc_fm"]="4929"
   ["ptc_fr"]="7366"
   ["ptc_mm"]="7762"
   ["ptc_mr"]="7366"
)

gpus=(0 1 2 3)
overall_counter=0

for config_file in \
   "configs/rq1/scoring=agg___tp=masked_attention_pp=lrl_when=post.yaml" \
   "configs/rq1/scoring=attention_pp=lrl___tp=masked_attention_pp=lrl_when=post.yaml" \
   "configs/rq1/scoring=masked_attention_pp=lrl___tp=masked_attention_pp=lrl_when=post.yaml" \
   "configs/rq1/scoring=masked_attention_pp=lrl___tp=masked_attention_pp=lrl_when=post___unify=true.yaml" \
   "configs/rq1/scoring=sinkhorn_pp=lrl___tp=masked_attention_pp=lrl_when=post.yaml" \
; do
   for dataset in "${!dataset_seeds[@]}"; do
      seed="${dataset_seeds[$dataset]}"
      gpu_index=$(( (overall_counter) % ${#gpus[@]} ))

      CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
         --experiment_id rq1 \
         --experiment_dir experiments/ \
         --model_config_path $config_file \
         --dataset_name $dataset \
         --seed $seed \
         --dataset_size large \
         &

      ((overall_counter++))
      sleep 30s
   done
done
