import time
import torch
import networkx as nx
from utils.parser import Parser
import subgraph_matching.dataset as dataset
from subgraph_matching.dataset import get_datasets

def isomorphic(test_dataset):

    pos_pairs, neg_pairs = test_dataset.pos_pairs, test_dataset.neg_pairs
    all_pairs = pos_pairs + neg_pairs

    num_wrong = 0
    tot_time = 0
    for query_idx in range(len(all_pairs)):
        pos_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, pos_pairs))
        neg_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, neg_pairs))
        num_pos_pairs_query, num_neg_pairs_query = len(pos_pairs_for_query), len(neg_pairs_for_query)

        all_pairs_for_query = pos_pairs_for_query + neg_pairs_for_query

        predictions = []
        for query_index, corpus_index in all_pairs_for_query:
            query = test_dataset.query_graphs[query_index]
            corpus = test_dataset.corpus_graphs[corpus_index]
            GM = nx.isomorphism.GraphMatcher(corpus, query)
            time_now = time.time()
            pred = GM.subgraph_is_isomorphic()
            tot_time += time.time() - time_now
            predictions.append(pred)
        all_predictions = torch.tensor(predictions)
        all_labels = torch.cat([torch.ones(num_pos_pairs_query), torch.zeros(num_neg_pairs_query)])

        num_wrong += torch.sum(all_predictions != all_labels)
    print(tot_time)


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()

    dataset_config = parser.get_dataset_config()
    data_type = dataset.PYG_DATA_TYPE

    datasets = get_datasets(dataset_config, None, data_type, modes=['test'])
    test_dataset = datasets['test']

    isomorphic(test_dataset)