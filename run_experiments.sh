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

# datasets=('aids')
datasets=('aids' 'mutag' 'ptc_fr' 'ptc_fm' 'ptc_mr' 'ptc_mm')

for ((idx=0; idx<${#datasets[@]}; idx++)); do
    dataset="${datasets[$idx]}"
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$(((idx) % 4)) python -m subgraph.iso_matching_models \
    --experiment_group=${experiment_group} \
    --TASK=${TASK} \
    --NOISE_FACTOR=0 \
    --MARGIN=0.5 \
    --filters_1=10 \
    --filters_2=10 \
    --filters_3=10 \
    --transform_dim=16 \
    --IPLUS_LAMBDA=0.7 \
    --output_type=2 \
    --no_of_query_subgraphs=300 \
    --MIN_QUERY_SUBGRAPH_SIZE=5 \
    --MAX_QUERY_SUBGRAPH_SIZE=15 \
    --MIN_CORPUS_SUBGRAPH_SIZE=16 \
    --MAX_CORPUS_SUBGRAPH_SIZE=20 \
    --FEAT_TYPE="One" \
    --DIR_PATH="velugoti_dataset" \
    --DATASET_NAME=${dataset} &
    sleep 10s
done
