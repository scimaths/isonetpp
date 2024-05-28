import os
import json
import shutil
from utils.tooling import read_config


def get_models():
    for experiment_dir in paths_to_experiment_dir:
        for experiment in os.listdir(experiment_dir):

            model_paths = []
            for model in sorted(os.listdir(experiment_dir + experiment + "/trained_models")):
                model_paths.append(experiment_dir + experiment + "/trained_models/" + model)

            config_paths = []
            for config in sorted(os.listdir(experiment_dir + experiment + "/configs")):
                config_paths.append(experiment_dir + experiment + "/configs/" + config)

            log_paths = []
            for log in sorted(os.listdir(experiment_dir + experiment + "/logs")):
                log_paths.append(experiment_dir + experiment + "/logs/" + log)

            model_config_log_pair = []
            for model_path in model_paths:
                model_name = model_path.split("/")[-1].split(".")[0]

                index = 0
                curr_config_path = None
                for config_path in config_paths:
                    if model_name in config_path:
                        index += 1
                        curr_config_path = config_path

                if index > 1:
                    raise ValueError("More than one config file for model")
                if index == 0:
                    raise ValueError("No config file for model")

                index = 0
                for log_path in log_paths:
                    if model_name in log_path:
                        index += 1
                        model_config_log_pair.append((model_path, curr_config_path, log_path))

                if index > 1:
                    raise ValueError("More than one log file for model")
                if index == 0:
                    raise ValueError("No log file for model")

            for model_path, config_path, log_path in model_config_log_pair:
                yield model_path, config_path, log_path

def collect_models(models_to_run):

    for model_path, local_config_path, log_path in get_models():
        model_params, _ = read_config(local_config_path, with_dict=True)

        model_name = model_params.model
        dataset_name = model_params.dataset[:-6]
        seed = int(model_params.seed)

        if (
            model_name not in models_to_run
        ) or (
            seed != models_to_run[model_name]["seeds"][dataset_name]
        ):
            continue

        if "relevant_models" not in models_to_run[model_name]:
            models_to_run[model_name]["relevant_models"] = []

        models_to_run[model_name]["relevant_models"].append({
            "name": model_name,
            "seed": seed,
            "dataset": dataset_name,
            "model_path": model_path,
            "log_path": log_path,
            "config_path": local_config_path,
            "margin": "0.5" if "margin" not in models_to_run[model_name] or dataset_name not in models_to_run[model_name]["margin"] else "0.1",
            "map_score": "0",
            "hits@20": "0"
        })


    print("Total relevant models found")
    for model_name, metadata in models_to_run.items():
        print(model_name, ":", len(metadata["relevant_models"]) if "relevant_models" in metadata else 0)

    for model_name in models_to_run:
        if "relevant_models" not in models_to_run[model_name]:
            continue
        for idx, relevant_model in enumerate(models_to_run[model_name]["relevant_models"]):

            # Extract the filename from the source file path
            model_file_name = os.path.basename(relevant_model["model_path"])
            log_file_name = os.path.basename(relevant_model["log_path"])
            config_file_name = os.path.basename(relevant_model["config_path"])

            # check sanity: margin present in file if 0.1 and TEST - ap_score present
            log_content = open(relevant_model["log_path"], "r").read()
            if "TEST - ap_score" not in log_content:
                continue
            
            if relevant_model["margin"] == "0.1" and "--margin 0.1" not in log_content:
                continue
            elif relevant_model["margin"] == "0.5" and "--margin 0.1" in log_content:
                continue

            # Construct the destination file path
            if not os.path.exists(collection_path + relevant_model["name"]):
                os.makedirs(collection_path + relevant_model["name"])
                os.makedirs(collection_path + relevant_model["name"] + "/trained_models")
                os.makedirs(collection_path + relevant_model["name"] + "/logs")
                os.makedirs(collection_path + relevant_model["name"] + "/configs")

            destination_file = os.path.join(collection_path + relevant_model["name"] + "/trained_models", model_file_name)
            destination_log = os.path.join(collection_path + relevant_model["name"] + "/logs", log_file_name)
            destination_config = os.path.join(collection_path + relevant_model["name"] + "/configs", config_file_name)

            try:
                # Copy the file to the destination directory
                shutil.copy2(relevant_model["model_path"], destination_file)
                shutil.copy2(relevant_model["log_path"], destination_log)
                shutil.copy2(relevant_model["config_path"], destination_config)
                print(f"File '{model_file_name}' copied successfully.")
            except Exception as e:
                print(f"An error occurred: {e}")


def main():

    with open(table_path, "rb") as f:
        table_meta = json.load(f)

    collect_models(table_meta)

if __name__ == "__main__":
    # base_path = "/raid/infolab/ashwinr/isonetpp/"
    base_path = "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/"
    # # base_path = "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/"
    paths_to_experiment_dir = [
        # base_path + "experiments/",
        # base_path + "experiments_from_infolab_vaibhav/",
        # base_path + "experiments_updated/",
        # base_path + "experiments/",
        # base_path + "experiments_from_gise/experiments/",
        # "/mnt/home/vaibhavraj/isonetpp_enhanced_code/experiments_archived_march_16/",
        # "/mnt/home/vaibhavraj/isonetpp_enhanced_code/experiments/",
        # "/mnt/nas/vaibhavraj/isonetpp_experiments/",
        # "/mnt/nas/vaibhavraj/isonetpp_experiments_march_16/",
        # "/mnt/nas/vaibhavraj/isonet_experiments_02_april/",
        # "/mnt/home/vaibhavraj/isonetpp_enhanced_code/experiments/",
        # "/mnt/home/vaibhavraj/isonetpp_enhanced_code/experiments_archived_march_16/",
        # "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/experiments_updated/",
        # "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/experiments_updated_may_9_9_am/",
        # "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/experiments_updated_may_10_9_am/",
        # "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/experiments_updated_may_10_9_pm/",
        # "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/experiments_updated_may_9_12_pm/",
        # "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/efficiency_experiments/",
        # "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/rq7_efficiency/",
        # "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/egsc_margin_01/",
        "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/edge_mutag/"
    ]

    table_num = 4
    table_path = base_path + f"paper_artifacts/table_metadata/table_{table_num}.json"

    collection_path = base_path + "paper_artifacts/collection/"
    main()
