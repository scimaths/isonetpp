import time
import networkx as nx
from utils.parser import Parser
import subgraph_matching.dataset as dataset
from subgraph_matching.dataset import get_datasets


def isomorphic(test_dataset):

    query_graphs = test_dataset.query_graphs
    corpus_graphs = test_dataset.corpus_graphs

    for idx, query_graph in enumerate(query_graphs):
        num_nodes = len(query_graph.nodes)
        edges = query_graph.edges
        mapping = {}
        for edge in edges:
            if edge[0] not in mapping:
                mapping[edge[0]] = []
            if edge[1] not in mapping:
                mapping[edge[1]] = []
            mapping[edge[0]].append(edge[1])
            mapping[edge[1]].append(edge[0])
        with open(f"query_graphs/{idx}.txt", "w") as f:
            f.write(f"{num_nodes}\n")
            for node in mapping:
                other = " ".join([str(node) for node in mapping[node]])
                f.write(f"{len(mapping[node])} {other}\n")
                
    for idx, corpus_graph in enumerate(corpus_graphs):
        num_nodes = len(corpus_graph.nodes)
        edges = corpus_graph.edges
        mapping = {}
        for edge in edges:
            if edge[0] not in mapping:
                mapping[edge[0]] = []
            if edge[1] not in mapping:
                mapping[edge[1]] = []
            mapping[edge[0]].append(edge[1])
            mapping[edge[1]].append(edge[0])
        with open(f"corpus_graphs/{idx}.txt", "w") as f:
            f.write(f"{num_nodes}\n")
            for node in mapping:
                other = " ".join([str(node) for node in mapping[node]])
                f.write(f"{len(mapping[node])} {other}\n")

if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()

    dataset_config = parser.get_dataset_config()
    data_type = dataset.PYG_DATA_TYPE

    datasets = get_datasets(dataset_config, None, data_type, modes=['test'])
    test_dataset = datasets['test']

    isomorphic(test_dataset)