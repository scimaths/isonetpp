cd ..

# best seeds for Node-Early
declare -A dataset_seeds=(
   ["aids"]="7762"
   ["mutag"]="7762"
   ["ptc_fm"]="7474"
   ["ptc_fr"]="7762"
   ["ptc_mm"]="7762"
   ["ptc_mr"]="7366"
)

gpus=(0 1 2 3 4 5)
overall_counter=0

for config_file in \
   "configs/edge_early_variants/edge_early_interaction_1_post.yaml" \
   "configs/edge_early_variants/edge_early_interaction_1_pre.yaml" \
   "configs/edge_early_variants/edge_early_interaction_baseline_1_post.yaml" \
   "configs/edge_early_variants/edge_early_interaction_baseline_1_pre.yaml" \
; do
    for dataset in "${!dataset_seeds[@]}"; do
        seed="${dataset_seeds[$dataset]}"
        gpu_index=$(( (overall_counter) % ${#gpus[@]} ))

        CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
            --experiment_id rq6_edge_early_variants \
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
