import os
import matplotlib.pyplot as plt

datasets = ["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"]
dataset_batch = [1094, 1102, 1078, 1122, 1116, 1086]
for dataset_idx in range(len(datasets)):
    consistency_scores = []
    pair_scores = []
    for lambd in ["0", "01", "1"]:
        model_name = f"./nanl_consistency_45_losstype5_losslambda{lambd}_consistencylambda07"
        for file in os.listdir(model_name + "/logDir/"):
            if datasets[dataset_idx] in file:
                with open(model_name + "/logDir/" + file, "r") as f:
                    logs = f.readlines()
                logs_runs = []
                for line in logs:
                    if "Run" in line:
                        logs_runs.append(line)
                pair_score = []
                consistency_score = []
                for index in range(0, len(logs_runs), 7):
                    pair_score.append(float(logs_runs[index:index+7][1].split(" ")[-1]))
                    consistency_score.append(float(logs_runs[index:index+7][3].split(" ")[-1]))
                consistency_scores.append(consistency_score)
                pair_scores.append(pair_score)
    for lambd in ["10"]:
        model_name = f"./nanl_consistency_45_losstype5_losslambda{lambd}_consistencylambda07"
        for file in os.listdir(model_name + "/logDir/"):
            if datasets[dataset_idx] in file:
                with open(model_name + "/logDir/" + file, "r") as f:
                    logs = f.readlines()
                logs_runs = []
                for line in logs:
                    if "Run" in line:
                        logs_runs.append(line)
                pair_score = []
                consistency_score = []
                for index in range(0, len(logs_runs), 6):
                    consistency_score.append(float(logs_runs[index:index+6][2].split(" ")[-1]))
                    pair_score.append((float(logs_runs[index:index+6][0].split(" ")[-3]) / dataset_batch[0]) - float(logs_runs[index:index+6][2].split(" ")[-1])*10)
                consistency_scores.append(consistency_score)
                pair_scores.append(pair_score)
    lambd_list = ["0", "0.1", "1", '10']
    for plot_idx in range(len(pair_scores)):
        plt.plot(range(len(pair_scores[plot_idx])), pair_scores[plot_idx], label=lambd_list[plot_idx])
    plt.xlabel('Runs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'Plot of Ranking Loss for dataset: {datasets[dataset_idx]}')  
    plt.legend()
    plt.savefig(f"plots_45/pair_scores_{datasets[dataset_idx]}.png")
    plt.clf()
    for plot_idx in range(len(consistency_scores)):
        plt.plot(range(len(consistency_scores[plot_idx])), consistency_scores[plot_idx], label=lambd_list[plot_idx])
    plt.xlabel('Runs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'Plot of Consistency |sinkhorn(K1+K2) - S| Loss for dataset: {datasets[dataset_idx]}')  
    plt.legend()
    plt.savefig(f"plots_45/consistency_scores_{datasets[dataset_idx]}.png")
    plt.clf()