import os
import sys
import time
import pickle
import argparse
import itertools
import numpy as np
import networkx as nx
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from subgraph.graphs import TUDatasetGraph
from subgraph.sampler import OnTheFlySubgraphSampler

def check_isomorphism(data):
    gc, gq = data
    return nx.algorithms.isomorphism.GraphMatcher(gc, gq).subgraph_is_isomorphic()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--logpath", type=str, default="logDir/logfile", help="/path/to/log")
    parser.add_argument("--DIR_PATH", type=str, help="path/to/datasets")
    # Graph size
    parser.add_argument("--MIN_QUERY_SUBGRAPH_SIZE", type=int)
    parser.add_argument("--MAX_QUERY_SUBGRAPH_SIZE", type=int)
    parser.add_argument("--MIN_CORPUS_SUBGRAPH_SIZE", type=int)
    parser.add_argument("--MAX_CORPUS_SUBGRAPH_SIZE", type=int)
    # Dataset specifics
    parser.add_argument("--DATASET_NAME", type=str)
    parser.add_argument("--TASK", type=str, default="NESGIso", help="PermGnnPointEmbedBON/PermGnnPointEmbedBOE")
    parser.add_argument("--NUM_QUERY_SUBGRAPHS", type=int, default=100)
    parser.add_argument("--NUM_CORPUS_SUBGRAPHS", type=int, default=500)

    args = parser.parse_args()
    os.makedirs(os.path.join(args.DIR_PATH, "Datasets"), exist_ok=True)
    os.makedirs(os.path.join(args.DIR_PATH, "preprocessed"), exist_ok=True)
    splits_dir = os.path.join(args.DIR_PATH, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    args.logpath = f"{args.logpath}_{args.TASK}_{args.DATASET_NAME}"

    logger = open(args.logpath, "w")
    def log(string):
        logger.write(string + "\n")
        logger.flush()

    log("Command line arguments")
    log('\n'.join(sys.argv[:]))

    tu_graph = TUDatasetGraph(args)
    graphs = tu_graph.get_nx_graph()

    no_of_query_subgraphs = args.NUM_QUERY_SUBGRAPHS
    no_of_corpus_subgraphs = args.NUM_CORPUS_SUBGRAPHS
    pair_count_str = f"{(no_of_corpus_subgraphs * no_of_query_subgraphs) // 1000}k"

    preprocessed_corpus_graphs_path = os.path.join(
        args.DIR_PATH, "preprocessed",
        f"{args.DATASET_NAME}{pair_count_str}_corpus_subgraphs_{no_of_corpus_subgraphs}_min_{args.MIN_CORPUS_SUBGRAPH_SIZE}_max_{args.MAX_CORPUS_SUBGRAPH_SIZE}.pkl"
    )
    log("Sampling corpus subgraphs")

    corpus_subgraph_list, corpus_anchor_list, corpus_subgraph_id_list = [], [], []
    args.MIN_SUBGRAPH_SIZE = args.MIN_CORPUS_SUBGRAPH_SIZE
    args.MAX_SUBGRAPH_SIZE = args.MAX_CORPUS_SUBGRAPH_SIZE
    subgraph_sampler = OnTheFlySubgraphSampler(args, graphs)
    selected_corpus_graph_cnt = 0
    while selected_corpus_graph_cnt < no_of_corpus_subgraphs:
        subgraph, anchor, subgraph_id = subgraph_sampler.sample_subgraph()
        if any([nx.is_isomorphic(subgraph, sel_graph) for sel_graph in corpus_subgraph_list]):
            continue
        corpus_subgraph_list.append(subgraph)
        corpus_anchor_list.append(anchor)
        corpus_subgraph_id_list.append(subgraph_id)
        selected_corpus_graph_cnt += 1

    log(f"Corpus Preprocessed Graphs Path - {preprocessed_corpus_graphs_path}")
    with open(preprocessed_corpus_graphs_path, "wb") as f:
        pickle.dump((corpus_subgraph_list, corpus_anchor_list, corpus_subgraph_id_list), f)

    preprocessed_query_graphs_path = os.path.join(
        args.DIR_PATH, "preprocessed",
        f"{args.DATASET_NAME}{pair_count_str}_query_subgraphs_{no_of_query_subgraphs}_min_{args.MIN_QUERY_SUBGRAPH_SIZE}_max_{args.MAX_QUERY_SUBGRAPH_SIZE}.pkl"
    )
    log("Sampling query subgraphs")

    args.MIN_SUBGRAPH_SIZE = args.MIN_QUERY_SUBGRAPH_SIZE
    args.MAX_SUBGRAPH_SIZE = args.MAX_QUERY_SUBGRAPH_SIZE
    subgraph_sampler = OnTheFlySubgraphSampler(args, graphs)
    query_subgraph_list, query_anchor_list, query_subgraph_id_list = [], [], []

    sampling_start_time = time.time()
    rel_dict = {}
    n_queries = 0

    with ProcessPool(max_workers=100) as pool:
        while n_queries < no_of_query_subgraphs:
            start = time.time()

            sgraph, anchor, sgraph_id = subgraph_sampler.sample_subgraph()
            if any([nx.is_isomorphic(sgraph, sel_graph) for sel_graph in query_subgraph_list]):
                log("Discarded due to isomorphism")
                continue
            pos_c, neg_c = [], []
            
            future = pool.map(check_isomorphism, zip(corpus_subgraph_list, itertools.repeat(sgraph)), timeout=30)
            iterator = future.result()

            for c_i in range(no_of_corpus_subgraphs): 
                try:
                    result = next(iterator)
                    #assert(q_i==result[0] and c_i==result[1])
                    if result ==1:
                        pos_c.append(c_i)
                    else:
                        neg_c.append(c_i)
                    
                except StopIteration:
                    break
                except TimeoutError as error:  
                    #rels[c_i][q_i] = False
                    neg_c.append(c_i)
                    log(str(c_i) + " Timeout")            
                
            if len(neg_c) ==0:
                r = 0
            else:  
                r = len(pos_c)/len(neg_c)
            if r>=0.1 and r<=0.4:
                log(f"q: {n_queries} ratio : {r}")
                query_subgraph_list.append(sgraph)
                query_anchor_list.append(anchor)
                query_subgraph_id_list.append(sgraph_id)
                rel_dict[n_queries] = {}
                rel_dict[n_queries]['pos'] = pos_c
                rel_dict[n_queries]['neg'] = neg_c
                n_queries = n_queries + 1
            else: 
                log(f"Discarded due to ratio : {r}")
            log(f"time to decide: {time.time()-start}")

    log(f"Query Preprocessed Graphs Path - {preprocessed_query_graphs_path}")
    with open(preprocessed_query_graphs_path, 'wb') as f:
        pickle.dump((query_subgraph_list, query_anchor_list, query_subgraph_id_list), f)

    subgraph_rel_type = "nx_is_subgraph_iso"

    joint_path_string = (
        f"{args.DATASET_NAME}{pair_count_str}_query_{no_of_query_subgraphs}_corpus_{no_of_corpus_subgraphs}" +
        f"_minq_{args.MIN_QUERY_SUBGRAPH_SIZE}_maxq_{args.MIN_QUERY_SUBGRAPH_SIZE}" +
        f"_minc_{args.MIN_CORPUS_SUBGRAPH_SIZE}_maxc_{args.MAX_CORPUS_SUBGRAPH_SIZE}" +
        f"_rel_{subgraph_rel_type}.pkl"
    )
    combined_preprocessed_path = os.path.join(
        args.DIR_PATH, "preprocessed", joint_path_string
    )
    
    with open(combined_preprocessed_path, 'wb') as f:
        pickle.dump(rel_dict, f)

    log(f"Total time: {time.time() - sampling_start_time}")

    def relabel_graphs(graph_list):
        return list(map(lambda g: nx.relabel.relabel_nodes(g, {node: idx for idx, node in enumerate(g.nodes)}), graph_list))

    with open(
        os.path.join(args.DIR_PATH, "splits", f"{args.DATASET_NAME}{pair_count_str}_corpus_subgraphs.pkl"
    ), "wb") as f:
        pickle.dump(relabel_graphs(corpus_subgraph_list), f)

    relabeled_query_subgraph_list = relabel_graphs(query_subgraph_list)
    reordering = np.random.permutation(len(relabeled_query_subgraph_list))
    train_len, val_len = int(0.6 * len(reordering)), int(0.15 * len(reordering))
    test_len = len(reordering) - train_len - val_len
    train_graph_idxs = reordering[:train_len]
    val_graph_idxs = reordering[train_len : train_len + val_len]
    test_graph_idxs = reordering[train_len + val_len:]

    for indices, name in [
        (train_graph_idxs, "train"),
        (val_graph_idxs, "val"),
        (test_graph_idxs, "test"),
    ]:
        mode_dir = os.path.join(splits_dir, name)
        os.makedirs(mode_dir, exist_ok=True)
        mode_graph_list = []
        mode_rel_dict = {}
        for mode_idx, original_idx in enumerate(indices):
            mode_graph_list.append(relabeled_query_subgraph_list[original_idx])
            mode_rel_dict[mode_idx] = rel_dict[original_idx]

        with open(os.path.join(mode_dir, f"{name}_{args.DATASET_NAME}{pair_count_str}_query_subgraphs.pkl"), "wb") as f:
            pickle.dump(mode_graph_list, f)
        with open(os.path.join(mode_dir, f"{name}_{args.DATASET_NAME}{pair_count_str}_rel_{subgraph_rel_type}.pkl"), "wb") as f:
            pickle.dump(mode_rel_dict, f)