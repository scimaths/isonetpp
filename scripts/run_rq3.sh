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

gpus=(5)
overall_counter=0

# Not running PRE for 1st paper on dog
   # "configs/rq3/scoring=attention_pp=lrl___tp=attention_pp=lrl_when=pre___unify=true.yaml" \
   # "configs/rq3/scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true.yaml" \
   # "configs/rq3/scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=post___unify=true.yaml" \
for config_file in \
   "configs/rq3/scoring=attention_pp=lrl___tp=attention_pp=lrl_when=post___unify=true.yaml" \
; do
   for dataset in "${!dataset_seeds[@]}"; do
      seed="${dataset_seeds[$dataset]}"
      gpu_index=$(( (overall_counter) % ${#gpus[@]} ))

      CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
         --experiment_id rq3 \
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
