import os
import re
import glob
import matplotlib.pyplot as plt

filesT = []
while True:
    pattern_name = input("model log dir file name?")
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
for plot_idx in range(len(train_losses)):
    plt.plot(range(len(train_losses[plot_idx])), train_losses[plot_idx], label=filesT[plot_idx], linewidth=0.5)
plt.xlabel('Runs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Plot of Training Loss for files')  
plt.legend()
plt.savefig("train_losses.png")
plt.clf()
for plot_idx in range(len(val_map_scores)):
    plt.plot(range(len(val_map_scores[plot_idx])), val_map_scores[plot_idx], label=filesT[plot_idx], linewidth=0.5)
plt.xlabel('Runs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Plot of Val MAP Scores for files')  
plt.legend()
plt.savefig("val_map_scores.png")
plt.clf()