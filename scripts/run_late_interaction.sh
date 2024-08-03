cd ..

declare -A dataset_seeds=(
   ["aids"]="7762"
   ["mutag"]="7762"
   ["ptc_fm"]="7762"
   ["ptc_fr"]="7762"
   ["ptc_mm"]="7762"
   ["ptc_mr"]="7762"
)

gpus=(2 3)
overall_counter=0

for config_file in \
   "configs/isonet.yaml" \
   "configs/nanl_consistency.yaml" \
; do
   # "configs/..." \
   for dataset in "${!dataset_seeds[@]}"; do
      seed="${dataset_seeds[$dataset]}"
      gpu_index=$(( (overall_counter) % ${#gpus[@]} ))

      CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
         --experiment_id rqY_late_interaction \
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
