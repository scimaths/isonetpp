import os
import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

node_early_models = []
for model_loc in sorted(os.listdir("./histogram_dumps")):
    if "model_node_early" in model_loc and "time_0" in model_loc:
        node_early_models.append(model_loc)

nanl_models = []
for model_loc in sorted(os.listdir("./histogram_dumps")):
    if "model_node_align" in model_loc:
        nanl_models.append(model_loc)

for model_idx in range(len(node_early_models)):
    node_early_model = node_early_models[model_idx]
    nanl_model = nanl_models[model_idx]
    node_early_model_dump = pickle.load(open("histogram_dumps/"+ node_early_model, "rb"))
    nanl_model_dump = pickle.load(open("histogram_dumps/" + nanl_model, "rb"))
    sns.histplot(node_early_model_dump, binwidth=0.1, binrange=(0, np.ceil(np.max(node_early_model_dump))), kde=True, label=f'node early time = 0', palette='pastel')
    sns.histplot(nanl_model_dump, binwidth=0.1, binrange=(0, np.ceil(np.max(node_early_model_dump))), kde=True, label=f'nanl', palette='pastel')

    plt.xlabel('NORM(P * P_hat)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram {node_early_model} vs {nanl_model}')

    plt.legend()
    plt.grid(True)
    plt.savefig(f'histogram_plots/{node_early_model}_vs_{nanl_model}.png')
    plt.clf()