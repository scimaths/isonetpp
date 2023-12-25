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

if [ "$TASK" == "node_early_interaction" ] || [ "$TASK" == "node_early_interaction_interpretability" ] || [ "$TASK" == "edge_early_interaction" ] || [ "$TASK" == "node_edge_early_interaction_with_consistency" ] || [ "$TASK" == "node_edge_early_interaction_with_consistency_and_two_sinkhorns" ]; then
    if [ -z "$3" ]; then
        echo Give time_updates argument
        exit
    else
        time_updates=$3
    fi
else
    time_updates=0
fi

if [ "$TASK" == "node_edge_early_interaction_with_consistency" ] || [ "$TASK" == "node_edge_early_interaction_with_consistency_and_two_sinkhorns" ]; then
    if [ -z "$4" ]; then
        echo Give consistency_lambda argument
        exit
    else
        consistency_lambda=$4
    fi
else
    consistency_lambda=0
fi

# datasets=('aids')
datasets=('aids' 'mutag' 'ptc_fr' 'ptc_fm' 'ptc_mr' 'ptc_mm')

for ((idx=0; idx<${#datasets[@]}; idx++)); do
    dataset="${datasets[$idx]}"
    for seed in 0 1 2 3 4; do
        CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$(((idx * 6 + seed) % 3 + 2)) python -m subgraph.iso_matching_models \
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
        --consistency_lambda=${consistency_lambda} \
        --SEED=${seed} &
        sleep 10s
    done
done
