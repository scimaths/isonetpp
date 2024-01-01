import os
import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

edge_early_models = []
for model_loc in sorted(os.listdir("./histogram_dumps")):
    if "model_edge_early" in model_loc and "time_0" in model_loc:
        edge_early_models.append(model_loc)

isonet_models = []
for model_loc in sorted(os.listdir("./histogram_dumps")):
    if "model_isonet" in model_loc:
        isonet_models.append(model_loc)

for model_idx in range(len(edge_early_models)):
    edge_early_model = edge_early_models[model_idx]
    isonet_model = isonet_models[model_idx]
    edge_early_model_dump = pickle.load(open("histogram_dumps/"+ edge_early_model, "rb"))
    isonet_model_dump = pickle.load(open("histogram_dumps/" + isonet_model, "rb"))
    sns.histplot(edge_early_model_dump, binwidth=0.1, binrange=(0, np.ceil(np.max(edge_early_model_dump))), kde=True, label=f'edge early time = 0', palette='pastel')
    sns.histplot(isonet_model_dump, binwidth=0.1, binrange=(0, np.ceil(np.max(edge_early_model_dump))), kde=True, label=f'isonet', palette='pastel')

    plt.xlabel('NORM(P * P_hat)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram {edge_early_model} vs {isonet_model}')

    plt.legend()
    plt.grid(True)
    plt.savefig(f'histogram_plots/{edge_early_model}_vs_{isonet_model}.png')
    plt.clf()