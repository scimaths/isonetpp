import os
import pickle
import numpy as np

dataset_base_path = "large_dataset/"

def load_first_split(dataset_name):
    def file_name(mode):
        return os.path.join(
        dataset_base_path,
        "splits",
        mode,
        f"{mode}_{dataset_name}_query_subgraphs.pkl"
    )

    graphs = []
    relations = []
    for mode in ["train", "dev", "test"]:
        data = pickle.load(open(file_name(mode)))

        for index in len(data):
            graphs.append(
                data.get(index)
            )

        relations.extend(
            pickle.load(open(file_name(mode).replace("query_subgraphs", "rel_nx_is_subgraph_iso")))
        )

    return graphs, relations


def save_split(train_index, dev_index, test_index):

    graphs = lambda dataset_name: pickle.load(
        open(dataset_base_path + dataset_name + ".pkl", "rb")
    )

    for dataset_name in ["aids"]:
        
        graph_data = graphs(dataset_name)
        dataset_len = len(graph_data)
        print(dataset_len)


def generate_splits(num=5, k=2):
    total_graphs_num = 300
    train_len = 180
    dev_len = 45
    test_len = 75

    initial_index = np.arange(total_graphs_num)
    indexes_list = [initial_index]

    for _ in range(num-1):
        indexes = initial_index.copy()
        np.random.shuffle(indexes)
        indexes_list.append(indexes)

    overlap_test = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            overlap_test[i, j] = len(
                np.intersect1d(
                    indexes_list[i][-test_len:], indexes_list[j][-test_len:]
                )
            )
    print(overlap_test)

    # choose the top k best split
    curr_splits = [0]
    for _ in range(k):
        overlap_sum = overlap_test[curr_splits].sum(axis=0)
        best_split = np.argmax(overlap_test[])
        best_splits.append(best_split)

generate_splits()