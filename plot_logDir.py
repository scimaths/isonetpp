import os
import re
import glob
import matplotlib.pyplot as plt

dataset_names=["aids", "mutag", "ptc_fr", "ptc_fm", "ptc_mr", "ptc_mm"]
fig, axs = plt.subplots(6, 2)
for idx in range(len(dataset_names)):
    dataset = dataset_names[idx]
    filesT = []
    patterns = [f'edge*02*/logDir/*{dataset}*', f"edge*05*/logDir/*{dataset}*", f"edge*08*/logDir/*{dataset}*", f"edge_early_clean/logDir/*{dataset}*"]
    j = 0
    while j < 4:
        pattern_name = patterns[j] # input("model log dir file name?")
        j += 1
        if pattern_name == "q":
            break
        pattern_name = ".*".join(pattern_name.split("*"))
        pattern = re.compile(pattern_name)
        matching_files = []
        for root, dirs, files in os.walk("./"):
            for file in files:
                file_path = os.path.join(root, file)
                if pattern.search(file_path):
                    matching_files.append(file_path)
        if len(matching_files) != 1:
            exit()
        filesT.append(matching_files[0])
    label = {" vs ".join(filesT)}
    val_map_scores = []
    train_losses = []
    for file in filesT:
        print(file)
        with open(file, "r") as f:
            logs = f.readlines()
        logs_runs = []
        for line in logs:
            if "Run" in line:
                logs_runs.append(line)
        if len(logs_runs) % 2 != 0:
            logs_runs = logs_runs[:-1]
        val_map_score = []
        train_loss = []
        for index in range(0, len(logs_runs), 2):
            train_loss.append(float(logs_runs[index:index+2][0].split(" ")[-3]))
            val_map_score.append(float(logs_runs[index:index+2][1].split(" ")[-3]))
        val_map_scores.append(val_map_score)
        train_losses.append(train_loss)
    labels = ["0.2","0.5","0.8","1"]
    for plot_idx in range(len(train_losses)):
        axs[idx,0].plot(range(len(train_losses[plot_idx])), train_losses[plot_idx], label=labels[plot_idx], linewidth=0.5)
    axs[idx,0].set_xlabel('Runs')
    axs[idx,0].set_ylabel('Loss')
    #axs[idx,0].yscale('log')
    axs[idx,0].set_title(f'Plot of Training Loss for Edge Early Fringe: {dataset}')  
    axs[idx,0].legend()
    #axs[idx,0].savefig(f"train_losses_{dataset}.png")
    #axs[idx,0].clf()
    for plot_idx in range(len(val_map_scores)):
        axs[idx,1].plot(range(len(val_map_scores[plot_idx])), val_map_scores[plot_idx], label=labels[plot_idx], linewidth=0.5)
    axs[idx,1].set_xlabel('Runs')
    axs[idx,1].set_ylabel('Score')
    #axs[idx,1].yscale('log')
    axs[idx,1].set_title(f'Plot of Val MAP Scores for Edge Early Fringe: {dataset}')  
    axs[idx,1].legend()
    #axs[idx,1].savefig(f"val_map_scores_{dataset}.png")
    #axs[idx,1].clf()
plt.savefig("edge_early_fringe.png")
