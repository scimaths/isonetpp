import os

best_seed = {}
best_map = {}
name = "h2mn"
num = {}

for file in sorted(os.listdir(f"experiments/{name}/logs")):
    if file.endswith(".txt") and "H2MN" in file:
        with open(f"experiments/{name}/logs/"+file, "r") as f:
            lines = f.readlines()
            curr_dataset = ""
            curr_map = 0
            curr_seed = 0
            for line in lines:
                if "--dataset_name" in line:
                    curr_dataset = line.split(" ")[1]
                if "--seed" in line:
                    curr_seed = int(line.split(" ")[1])
                if "VAL ap_score:" in line:
                    curr_map = max(curr_map, float(line.split(" ")[-3]))
                if "Run: 9 VAL" in line:
                    if curr_dataset in num:
                        num[curr_dataset].append(file)
                    else:
                        num[curr_dataset] = [file]
                    if curr_dataset in best_seed:
                        if curr_map > best_map[curr_dataset]:
                            best_seed[curr_dataset] = curr_seed
                            best_map[curr_dataset] = curr_map
                    else:
                        best_seed[curr_dataset] = curr_seed
                        best_map[curr_dataset] = curr_map
                    break
print(dict(sorted(best_seed.items(), key=lambda item: item[0])))
for dataset in num:
    print(dataset, len(num[dataset]))
    # assert len(num[dataset]) >= 5, len(num[dataset])
