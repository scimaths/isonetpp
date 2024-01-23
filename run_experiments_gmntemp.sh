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

if [ -z "$3" ]; then
    echo Give DATASET argument
    exit
else
    DATASET=$3
fi

temps=('0.05' '0.1' '0.5' '5')
cuda=('0' '0' '0' '0')
for ((idx=0; idx<${#temps[@]}; idx++)); do
    temp="${temps[$idx]}"
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=${cuda[$idx]} python -m subgraph.iso_matching_models \
    --experiment_group=${experiment_group} \
    --TASK=${TASK} \
    --MARGIN=0.5 \
    --filters_1=10 \
    --filters_2=10 \
    --filters_3=10 \
    --NOISE_FACTOR=0 \
    --transform_dim=16 \
    --FEAT_TYPE="One" \
    --temp_gmn_scoring=${temp} \
    --DATASET_NAME=${DATASET} \
    --no_of_query_subgraphs=300 \
    --MIN_QUERY_SUBGRAPH_SIZE=5 \
    --MAX_QUERY_SUBGRAPH_SIZE=15 \
    --MIN_CORPUS_SUBGRAPH_SIZE=16 \
    --MAX_CORPUS_SUBGRAPH_SIZE=20 &
    sleep 10s
done
