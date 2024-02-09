if [ -z "$1" ]; then
    echo Give EXPERIMENT argument
    exit
else
    EXPERIMENT=$1
fi

if [ -z "$2" ]; then
    echo Give MODEL argument
    exit
else
    MODEL=$2
fi

if [ -z "$3" ]; then
    echo Give MODEL_CONFIG_PATH argument
    exit
else
    MODEL_CONFIG_PATH=$3
fi

datasets=('aids' 'mutag' 'ptc_fm' 'ptc_fr' 'ptc_mm' 'ptc_mr')
cuda=('3' '3' '3' '3' '3' '3')

for ((idx=0; idx<${#datasets[@]}; idx++)); do
    dataset="${datasets[$idx]}"
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=${cuda[$idx]} python3 -m subgraph_matching.train \
    --experiment_id ${EXPERIMENT} \
    --experiment_dir experiments/ \
    --model_config_path ${MODEL_CONFIG_PATH} \
    --dataset_name ${dataset} \
    --seed 0 \
    --dataset_size large &
    sleep 3s
done


