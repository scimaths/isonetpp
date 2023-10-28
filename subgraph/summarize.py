import pandas as pd
import numpy as np
import pickle

scores = pickle.load(open('scores.pkl', 'rb'))
seeds_found = []
for model in scores:
    for dataset in scores[model]:
        seeds_found.extend(list(scores[model][dataset].keys()))
seeds_found = set(list(seeds_found))

dics = []
for model in ['isonet', 'node_align_node_loss']:
    for dataset in ['ptc_fr', 'ptc_fm', 'ptc_mr', 'ptc_mm', 'aids', 'mutag']:
        dic = {'Model': model, 'Dataset': dataset}
        for seed in seeds_found:
            if seed in scores[model][dataset]:
                dic[f'{seed}'] = scores[model][dataset][seed]
            else:
                dic[f'{seed}'] = None
        dics.append(dic)

df = pd.DataFrame(dics)
print(df)
df.to_csv("scores.csv", index=False)