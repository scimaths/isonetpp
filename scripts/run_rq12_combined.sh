cd ..

# 1704, 4929, 7762

declare -A dataset_seeds=(
   ["aids"]="1704"
   ["mutag"]="1704"
   ["ptc_fm"]="1704"
   ["ptc_fr"]="1704"
   ["ptc_mm"]="1704"
   ["ptc_mr"]="1704"
)

gpus=(0 1 2 3 4 5)
overall_counter=0

for config_file in \
    "configs/rq12_combined/scoring=agg___tp=attention_pp=lrl_when=post.yaml" \
    "configs/rq12_combined/scoring=agg___tp=masked_attention_pp=lrl_when=post.yaml" \
    "configs/rq12_combined/scoring=agg___tp=sinkhorn_pp=lrl_when=post.yaml" \
    "configs/rq12_combined/scoring=attention_pp=lrl___tp=attention_pp=lrl_when=post.yaml" \
    "configs/rq12_combined/scoring=attention_pp=lrl___tp=attention_pp=lrl_when=post___unify=true.yaml" \
    "configs/rq12_combined/scoring=attention_pp=lrl___tp=masked_attention_pp=lrl_when=post.yaml" \
    "configs/rq12_combined/scoring=attention_pp=lrl___tp=sinkhorn_pp=lrl_when=post.yaml" \
    "configs/rq12_combined/scoring=masked_attention_pp=lrl___tp=masked_attention_pp=lrl_when=post.yaml" \
    "configs/rq12_combined/scoring=masked_attention_pp=lrl___tp=masked_attention_pp=lrl_when=post___unify=true.yaml" \
    "configs/rq12_combined/scoring=sinkhorn_pp=lrl___tp=attention_pp=lrl_when=post.yaml" \
    "configs/rq12_combined/scoring=sinkhorn_pp=lrl___tp=masked_attention_pp=lrl_when=post.yaml" \
    "configs/rq12_combined/scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=post.yaml" \
    "configs/rq12_combined/scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=post___unify=true.yaml" \
; do
   for dataset in "${!dataset_seeds[@]}"; do
      seed="${dataset_seeds[$dataset]}"
      gpu_index=$(( (overall_counter) % ${#gpus[@]} ))

      CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
         --experiment_id rq12_combined \
         --experiment_dir experiments/ \
         --model_config_path $config_file \
         --dataset_name $dataset \
         --seed $seed \
         --dataset_size large \
         --dataset_path_override large_dataset_split_1/ \
         --max_epochs 10 \
         --wandb_config_path configs/wandb_benchmark_paper.yaml \
         &

      ((overall_counter++))
      sleep 10s
   done
done
