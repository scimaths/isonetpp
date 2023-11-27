if [ -z "$1" ]; then
    echo Give experiment_group argument
    exit
else
    experiment_group=$1
fi

if [ -z "$2" ]; then
    echo Give TASK argument
    exit
else
    TASK=$2
fi

if [ "$TASK" == "node_early_interaction" ] || [ "$TASK" == "node_early_interaction_interpretability" ]; then
    if [ -z "$3" ]; then
        echo Give time_updates argument
        exit
    else
        time_updates=$3
    fi
else
    time_updates=0
fi

# datasets=('aids' 'mutag' 'ptc_fr' 'ptc_fm' 'ptc_mr' 'ptc_mm')
datasets=('aids')
lambd=('0' '0.2' '0.5' '0.8' '1')

for ((idx=0; idx<${#datasets[@]}; idx++)); do
    dataset="${datasets[$idx]}"
    for seed in 0; do
        for ((idxl=0; idxl<${#lambd[@]}; idxl++)); do
            lambd="${lambd[$idxl]}"
            CUDA_VISIBLE_DEVICES=$((idxl)) python -m subgraph.iso_matching_models \
            --experiment_group=${experiment_group} \
            --TASK=${TASK} \
            --time_updates=${time_updates} \
            --NOISE_FACTOR=0 \
            --MARGIN=0.5 \
            --filters_1=10 \
            --filters_2=10 \
            --filters_3=10 \
            --transform_dim=16 \
            --FEAT_TYPE="One" \
            --DATASET_NAME=${dataset} \
            --lambd=${lambd} \
            --SEED=${seed} &
            sleep 10s
        done
    done
done
