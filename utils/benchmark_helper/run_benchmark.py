# model, split, dataset, best_seed
# 13, 3, 6
# data saved in the following format
# 
import os
import argparse
import json
import shutil
from utils.tooling import read_config

def get_models(path_to_experiment_dir):
    model_paths = []
    for model in sorted(os.listdir(path_to_experiment_dir + "/trained_models")):
        model_paths.append(path_to_experiment_dir + "/trained_models/" + model)

    config_paths = []
    for config in sorted(os.listdir(path_to_experiment_dir + "/configs")):
        config_paths.append(path_to_experiment_dir + "/configs/" + config)

    log_paths = []
    for log in sorted(os.listdir(path_to_experiment_dir + "/logs")):
        log_paths.append(path_to_experiment_dir + "/logs/" + log)

    model_config_log_pair = []
    for model_path in model_paths:
        model_name = model_path.split("/")[-1].split(".")[0]

        curr_config_path = None
        for config_path in config_paths:
            if model_name in config_path:
                curr_config_path = config_path

        for log_path in log_paths:
            if model_name in log_path:
                model_config_log_pair.append((model_path, curr_config_path, log_path))

    for model_path, config_path, log_path in model_config_log_pair:
        yield model_path, config_path, log_path


# Assumptions:
# 1. All models have either --max_epoch or fully trained, in either case have test score printed at the end
# 2. Once triggered, a model will finish training (If crashed, will have to manually set back is_trained to 0)
def create_benchmark_status(benchmark_location):
    models_to_run = json.load(
        open(os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "models.json"
            ), "r")
    )

    def get_val_map_score(log_path):
        log_content = open(log_path, "r").readlines()
        curr_map = 0
        for line in log_content:
            if "VAL ap_score:" in line:
                curr_map = max(curr_map, float(line.split(" ")[-3]))
            if "Run: 9 VAL" in line:
                return curr_map
        raise ValueError("No validation map score found")

    relevant_models = {}
    for model_path, local_config_path, log_path in get_models("experiments/rq12_combined"):
        # Read details from model config saved in experiments
        model_params, _ = read_config(local_config_path, with_dict=True)
        model_name = model_params.model
        dataset_name = model_params.dataset[:-6]
        seed = int(model_params.seed)

        # Assumption 1
        log_content = open(log_path, "r").read()
        if (
            model_name not in models_to_run
        ) or (
            "TEST - ap_score" not in log_content
        ):
            continue

        # Best validation map score till 10th epoch
        best_val_map_score = get_val_map_score(log_path)

        # Is this fully trained?
        is_trained = "--max_epochs 10" not in log_content

        # Which split?
        if "split_1" in log_content:
            split = 1
        elif "split_2" in log_content:
            split = 2
        else:
            split = 0
        
        if model_name not in relevant_models:
            relevant_models[model_name] = {}
        if dataset_name not in relevant_models[model_name]:
            relevant_models[model_name][dataset_name] = {}
        if split not in relevant_models[model_name][dataset_name]:
            relevant_models[model_name][dataset_name][split] = []
        relevant_models[model_name][dataset_name][split].append({
            "seed": seed,
            "val_map": best_val_map_score,
            "is_trained": is_trained,
            "log_path": log_path,
            "config_path": local_config_path,
            "model_path": model_path,
        })

    best_model_mappings = {}
    for model in relevant_models:
        best_model_mappings[model] = {}
        for dataset in relevant_models[model]:
            best_model_mappings[model][dataset] = {}
            for split in relevant_models[model][dataset]:
                number_of_models = len(relevant_models[model][dataset][split])
                print(f"Model: {model}, Dataset: {dataset}, Split: {split}, Number of models: {number_of_models}")
                best_model_mappings[model][dataset][split] = max(relevant_models[model][dataset][split], key=lambda x: x["val_map"])

    with open(benchmark_location, "w") as f:
        json.dump(best_model_mappings, f, indent=4)

def run_models(max_runs_to_start):
    def load_config():
        model_name_to_config_map = {}
        for root, _, files in os.walk("./configs/"):
            for file in files:
                config_path = os.path.join(root, file)
                model_params, _ = read_config(config_path, with_dict=True)
                if 'model_config' in model_params and 'name' in model_params:
                    model_name_to_config_map[model_params.name] = config_path
        return model_name_to_config_map

    model_name_to_config_map = load_config()
    benchmark_status = json.load(open(benchmark_location, "rb"))
    for model in benchmark_status:
        for dataset in benchmark_status[model]:
            for split in benchmark_status[model][dataset]:
                if benchmark_status[model][dataset][split]["is_trained"]:
                    continue
                if max_runs_to_start == 0:
                    break
                max_runs_to_start -= 1
                seed = benchmark_status[model][dataset][split]["seed"]
                config_path = model_name_to_config_map[model]

                print(f"\t\"{config_path} {seed} {dataset} {split}\"")
                # os.system(f"python train.py --config {config_path} --model {model_path} --seed {seed} --gpu {gpus[0]}")

                benchmark_status[model][dataset][split]["is_trained"] = True
                with open(benchmark_location, "w") as f:
                    json.dump(benchmark_status, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark models")
    parser.add_argument("--max_runs_to_start", type=int, default=1)
    parser.add_argument("--override", type=bool, default=False)
    args = parser.parse_args()

    max_runs_to_start = args.max_runs_to_start

    benchmark_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), "benchmark_status.json")
    if not os.path.exists(benchmark_location) or args.override:
        create_benchmark_status(benchmark_location)

    run_models(max_runs_to_start)
