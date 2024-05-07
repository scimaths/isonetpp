if [ -z "$1" ]; then
    echo Give EXPERIMENT argument
    exit
else
    EXPERIMENT=$1
fi

if [ -z "$2" ]; then
    echo Give MODEL_CONFIG_PATH argument
    exit
else
    MODEL_CONFIG_PATH=$2
fi

# datasets=('aids' 'mutag' 'ptc_fm' 'ptc_fr' 'ptc_mm' 'ptc_mr')
datasets=('ptc_fr')
cuda=('3')
# cuda=('6' '6' '6' '7' '7' '7')
seeds=('7366')
# seeds=('7762' '1704' '1704' '7474' '1704' '1704')

for ((idx=0; idx<${#datasets[@]}; idx++)); do
    dataset="${datasets[$idx]}"
    seed="${seeds[$idx]}"
    WANDB_MODE=disabled CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=4 python3 -m subgraph_matching.train \
    --experiment_id ${EXPERIMENT} \
    --experiment_dir experiments/ \
    --model_config_path ${MODEL_CONFIG_PATH} \
    --dataset_name ${dataset} \
    --seed 7474 \
    --dataset_size large \
    --margin 0.5 &
    sleep 3s
done
