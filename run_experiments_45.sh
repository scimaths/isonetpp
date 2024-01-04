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
    echo Give consistency_lambda argument
    exit
else
    consistency_lambda=$4
fi

if [ -z "$5" ]; then
    echo Give loss_lambda argument
    exit
else
    loss_lambda=$5
fi

if [ -z "$6" ]; then
    echo Give loss_type argument
    exit
else
    loss_type=$6
fi

# datasets=('aids')
datasets=('aids' 'mutag' 'ptc_fr' 'ptc_fm' 'ptc_mr' 'ptc_mm')

for ((idx=0; idx<${#datasets[@]}; idx++)); do
    dataset="${datasets[$idx]}"
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$((idx)) python -m subgraph.iso_matching_models \
    --experiment_group=${experiment_group} \
    --TASK=${TASK} \
    --NOISE_FACTOR=0 \
    --MARGIN=0.5 \
    --filters_1=10 \
    --filters_2=10 \
    --filters_3=10 \
    --transform_dim=16 \
    --FEAT_TYPE="One" \
    --DATASET_NAME=${dataset} \
    --consistency_lambda=${consistency_lambda} \
    --loss_lambda=${loss_lambda} \
    --output_type=2 \
    --loss_type=${loss_type} \
    --no_of_query_subgraphs=300 \
    --MIN_QUERY_SUBGRAPH_SIZE=5 \
    --MAX_QUERY_SUBGRAPH_SIZE=15 \
    --MIN_CORPUS_SUBGRAPH_SIZE=16 \
    --MAX_CORPUS_SUBGRAPH_SIZE=20 &
    sleep 10s
done
