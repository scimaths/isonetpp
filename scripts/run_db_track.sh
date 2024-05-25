cd ..

tuples=(
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 1704 aids 2"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 1704 aids 1"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 1704 aids 0"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 4929 mutag 2"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 4929 mutag 1"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 4929 mutag 0"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 4929 ptc_fm 2"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 4929 ptc_fm 1"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 4929 ptc_fm 0"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 4929 ptc_fr 2"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 7762 ptc_fr 1"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 4929 ptc_fr 0"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 1704 ptc_mm 2"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 7762 ptc_mm 1"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 4929 ptc_mm 0"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 1704 ptc_mr 2"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 4929 ptc_mr 1"
    "./configs/rq12_combined/scoring=neural___tp=sinkhorn_pp=lrl_when=post.yaml 4929 ptc_mr 0"
)

gpus=(0 1 2)
overall_counter=0

for tuple in "${tuples[@]}"; do
    IFS=' ' read -r config_path seed dataset_name dataset_split <<< "$tuple"
    gpu_index=$(( (overall_counter) % ${#gpus[@]} ))

    if [[ $dataset_split -eq 0 ]]; then
        CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
            --experiment_id rq12_final \
            --experiment_dir experiments/ \
            --model_config_path $config_path \
            --dataset_name $dataset_name \
            --seed $seed \
            --dataset_size large \
            --wandb_config_path configs/wandb_benchmark_paper_final_runs.yaml \
            &
    else
        CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
            --experiment_id rq12_final \
            --experiment_dir experiments/ \
            --model_config_path $config_path \
            --dataset_name $dataset_name \
            --seed $seed \
            --dataset_size large \
            --dataset_path_override large_dataset_split_$dataset_split/ \
            --wandb_config_path configs/wandb_benchmark_paper_final_runs.yaml \
            &
    fi

    ((overall_counter++))
    sleep 10s
done