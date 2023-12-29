if [ -z "$1" ]; then
    echo Give CUDA device
    exit
else
    cuda_device=$1
fi

if [ -z "$2" ]; then
    echo Give experiment_group argument
    exit
else
    experiment_group=$2
fi

if [ -z "$3" ]; then
    echo Give TASK argument
    exit
else
    TASK=$3
fi

if [ -z "$4" ]; then
    echo Give time_updates argument
    exit
else
    time_updates=$4
fi

if [ -z "$5" ]; then
    echo Give consistency_lambda argument
    exit
else
    consistency_lambda=$5
fi

# datasets=('aids')
datasets=('aids' 'mutag' 'ptc_fr' 'ptc_fm' 'ptc_mr' 'ptc_mm')

for ((idx=0; idx<${#datasets[@]}; idx++)); do
    dataset="${datasets[$idx]}"
        CUDA_VISIBLE_DEVICES=${cuda_device} python -m subgraph.iso_matching_models \
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
        --consistency_lambda=${consistency_lambda} &
        sleep 10s
done
