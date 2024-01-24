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
datasets=('ptc_fm' 'ptc_mr' 'ptc_mm')
# datasets=('aids' 'mutag' 'ptc_fr' 'ptc_fm' 'ptc_mr' 'ptc_mm')
# cuda=('4')
cuda=('1' '3' '3')
# cuda=('3' '3' '3' '2' '2' '2')
for ((idx=0; idx<${#datasets[@]}; idx++)); do
    dataset="${datasets[$idx]}"
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
    --DATASET_NAME=${dataset} \
    --no_of_query_subgraphs=300 \
    --MIN_QUERY_SUBGRAPH_SIZE=5 \
    --MAX_QUERY_SUBGRAPH_SIZE=15 \
    --MIN_CORPUS_SUBGRAPH_SIZE=16 \
    --MAX_CORPUS_SUBGRAPH_SIZE=20 &
    sleep 10s
done
