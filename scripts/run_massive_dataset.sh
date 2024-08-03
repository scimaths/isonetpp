cd ..

declare -A dataset_seeds=(
   ["dhfr"]="0"
   ["cox2"]="0"
)

gpus=(0 1 3)
overall_counter=0

for config_file in \
   "configs/rq4_baselines/scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true.yaml" \
   "configs/rq4_iterative/iterative___scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true.yaml" \
   "configs/edge_early_variants/edge_early_interaction_baseline_1_pre.yaml" \
   "configs/edge_early_variants/edge_early_interaction_1_pre.yaml" \
   "configs/rq4_baselines/scoring=agg___tp=attention_pp=identity_when=post.yaml" \
   "configs/isonet.yaml" \
   "configs/node_align_node_loss.yaml" \
; do
   for dataset in "${!dataset_seeds[@]}"; do
      seed="${dataset_seeds[$dataset]}"
      gpu_index=$(( (overall_counter) % ${#gpus[@]} ))

      CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
         --experiment_id massive_dataset \
         --experiment_dir experiments/ \
         --model_config_path $config_file \
         --dataset_name $dataset \
         --seed $seed \
         --dataset_size massive \
         &

      ((overall_counter++))
      sleep 10s
   done
done
