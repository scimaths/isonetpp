import networkx as nx
from utils.parser import Parser
from utils.tooling import read_config
from utils.experiment import Experiment
import subgraph_matching.dataset as dataset
from subgraph_matching.dataset import get_datasets


def isomorphic(train_dataset, val_dataset, test_dataset):

    # Sanity check
    graph_1 = nx.Graph()
    graph_1.add_edge(1, 2)
    graph_1.add_edge(2, 3)
    graph_1.add_edge(3, 4)
    graph_1.add_edge(4, 1)

    graph_2 = nx.Graph()
    graph_2.add_edge(4, 2)
    graph_2.add_edge(2, 3)
    graph_2.add_edge(3, 1)
    graph_2.add_edge(1, 4)

    print("Comparing isomorphic graphs", nx.is_isomorphic(graph_1, graph_2))


    train_query_graphs = train_dataset.query_graphs
    val_query_graphs = val_dataset.query_graphs
    test_query_graphs = test_dataset.query_graphs

    print("Train Query Graphs", len(train_query_graphs))
    print("Val Query Graphs", len(val_query_graphs))
    print("Test Query Graphs", len(test_query_graphs))
    print("Total pairs possible", len(train_query_graphs) * len(val_query_graphs) + len(train_query_graphs) * len(test_query_graphs) + len(val_query_graphs) * len(test_query_graphs))

    total_count = 0
    isomorphic_count = 0

    for qa in train_query_graphs:
        for qb in test_query_graphs:
            total_count += 1
            if nx.is_isomorphic(qa, qb):
                isomorphic_count += 1

    for qa in train_query_graphs:
        for qb in val_query_graphs:
            total_count += 1
            if nx.is_isomorphic(qa, qb):
                isomorphic_count += 1

    for qa in val_query_graphs:
        for qb in test_query_graphs:
            total_count += 1
            if nx.is_isomorphic(qa, qb):
                isomorphic_count += 1

    print("Out of ", total_count, " pairs, ", isomorphic_count, " are isomorphic")


    corpus_graphs = train_dataset.corpus_graphs
    print("Corpus Graphs", len(corpus_graphs))


    total_count = 0
    isomorphic_count = 0

    for qa in train_query_graphs:
        for qb in corpus_graphs:
            total_count += 1
            if nx.is_isomorphic(qa, qb):
                isomorphic_count += 1

    print("Out of ", total_count, " pairs, ", isomorphic_count, " are isomorphic")


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()

    device = 'cuda' if args.use_cuda else 'cpu'

    model_params, config_dict = read_config(args.model_config_path, with_dict=True)

    experiment_config = parser.get_experiment_config(model_params.name)
    experiment = Experiment(config=experiment_config, device=device)

    dataset_config = parser.get_dataset_config()
    data_type = dataset.PYG_DATA_TYPE

    datasets = get_datasets(dataset_config, experiment, data_type)
    train_dataset, val_dataset = datasets['train'], datasets['val']

    datasets = get_datasets(dataset_config, experiment, data_type, modes=['test'])
    test_dataset = datasets['test']

    isomorphic(train_dataset, val_dataset, test_dataset)