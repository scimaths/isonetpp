import os
import sys
import time
import torch
import argparse
from datetime import datetime
from common import logger, set_log
from GMN.configure import get_default_config
from subgraph.utils import save_initial_model

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from subgraph.earlystopping import EarlyStoppingModule
from subgraph.dataset import OurMatchingModelSubgraphIsoData

# Import models
from subgraph.models.simgnn import SimGNN
from subgraph.models.neuromatch import NeuroMatch
from subgraph.models.temporal_gnn import TemporalGNN
from subgraph.models.isonet import ISONET, ISONET_Sym
from subgraph.models.gmn_match import GMN_match, GMN_match_hinge
from subgraph.models.node_align_node_loss import Node_align_Node_loss
from subgraph.models.node_align_edge_loss import Node_align_Edge_loss
from subgraph.models.hungarian_node_align import Hungarian_Node_align_Node_loss
from subgraph.models.fringed_node_align_node_loss import Fringed_node_align_Node_loss
from subgraph.models.gmn_embed import GMN_embed, GMN_embed_hinge, GMN_embed_with_ColBERT_scores, GMN_embed_with_MLP_and_ColBERT_scores, GMN_embed_maxsim_dot, GMN_embed_maxsim_dot_corrected

from subgraph.eval_utils import evaluate_embeddings_similarity, pairwise_ranking_loss_similarity

def train(av,config):
  device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
  train_data = OurMatchingModelSubgraphIsoData(av,mode="train")
  val_data = OurMatchingModelSubgraphIsoData(av,mode="val")

  if av.TASK.startswith("node_align_node_loss"):
    logger.info("Loading model NodeAlignNodeLoss")  
    logger.info("This uses GMN encoder followed by parameterized sinkhorn with LRL and similarity computation using hinge scoring (H_q, PH_c)")  
    model = Node_align_Node_loss(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("temporal_gnn"):
    logger.info("Loading model Temporal_GMM")
    logger.info("This model implements early interaction with a temporal GNN")
    model = TemporalGNN(av, config, 1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_19_gmn_all"):
    logger.info("Loading model OurMatchingModelVar19_GMN_all")  
    logger.info("This uses GMN embedding model.No regularizer. ")  
    model = GMN_embed(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_27_gmn_edge_perm_sinkhorn_param_big_hinge_score_on_edges"):
    logger.info("Loading model OurMatchingModelVar27_GMN_encoding_EdgePerm_SinkhornParamBig_HingeScoreOnEdges")  
    logger.info("This uses GMN encoder to obtain bad of node embeddings per graph, which is then used to obtain bag of edge embeddings per graph - then parameterized sinkhorn with LRL and similarity computation using hinge scoring (H_q, PH_c) where H consists of edge embeds") 
    #One more hack. 
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = ISONET(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("isonet_with_fringe_nodes"):
    logger.info("Loading model IsoNet_with_Fringe_Nodes")  
    logger.info("This uses IsoNet with fringe node reweighting") 
    #One more hack. 
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = Fringed_node_align_Node_loss(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("hungarian_node_align_node_loss"):
    logger.info("Loading model Hungarian_Node_align_Node_loss")
    logger.info("This model implements two tracks - one uses the asymmetric score, its output passed through the Hungarian algo is subjected to another model and the symmetric score")
    model = Hungarian_Node_align_Node_loss(av, config, 1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_29_gmn_sinkhorn_param_big_hinge_score_on_edge_similarity_ff_adj_mask"):
    logger.info("Loading model OurMatchingModelVar29_GMN_encoding_SinkhornParamBig_HingeScoreOnEdgeSimilarityByFF_AdjMask")  
    logger.info("This uses GMN encoder followed by parameterized sinkhorn with LRL and similarity computation using hinge scoring (A_q, PA_cP^T) . Each entry of A is a single score forward and backward message for the edge obtained using a FF. On top of the similarity matrix, we put a mask corr to adjacency matrix")  
    model = Node_align_Edge_loss(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_graphsim"):
    logger.info("Loading model GraphSim")  
    logger.info("This is GraphSim model")  
    model = GraphSim(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "pyg"
    val_data.data_type = "pyg"
  elif av.TASK.startswith("ir_modified_gotsim"):
    logger.info("Loading IR_modified_GotSim")  
    logger.info("This uses IR_modified_GotSim  model. ")  
    model = GOTSim(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "pyg"
    val_data.data_type = "pyg"
  elif av.TASK.startswith("simgnn_noperm") :
    logger.info("Loading model SimGNN")
    logger.info("This loads the entire SimGNN model. Input feature is [1]. No node permutation is done after nx graph loading")
    model = SimGNN(av,1).to(device)
    train_data.data_type = "pyg"
    val_data.data_type = "pyg"
  elif av.TASK.startswith("gmn_match"):
    logger.info("Loading GMN_match")  
    logger.info("This uses GMN matching model.No regularizer. ")  
    model = GMN_match(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_34_gmn_embed_hinge"):
    logger.info("Loading GMN_embed_hinge")  
    logger.info("This uses GMN embedding model with hinge loss.No regularizer. ")  
    model = GMN_embed_hinge(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_35_gmn_match_hinge"):
    logger.info("Loading GMN_match_hinge")  
    logger.info("This uses GMN matching model with hinge loss.No regularizer. ")  
    model = GMN_match_hinge(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_36_gmn_edge_perm_sinkhorn_param_big_sqeuc_score_on_edges"):
    logger.info("Loading model OurMatchingModelVar36_GMN_encoding_EdgePerm_SinkhornParamBig_SqEucScoreOnEdges")  
    logger.info("This uses GMN encoder to obtain bad of node embeddings per graph, which is then used to obtain bag of edge embeddings per graph - then parameterized sinkhorn with LRL and similarity computation using squared euclidean scoring (H_q, PH_c) where H consists of edge embeds") 
    #One more hack. 
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = ISONET_Sym(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_neuromatch"): 
    logger.info("Loading model neuromatch")   
    logger.info("This is neuromatch model")   
    model = NeuroMatch(1,av.neuromatch_hidden_dim,av).to(device) 
    logger.info(model) 
    train_data.data_type = "pyg" 
    val_data.data_type = "pyg" 
  elif av.TASK.startswith("matching_iso_var_gmn_with_colbert_objective"):
    logger.info("Loading model GMN_with_ColBERT_scores")  
    logger.info("This uses GMN encoder to obtain set of node embeddings per graph, then subjects that to ColBERT loss") 
    #One more hack. 
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = GMN_embed_with_ColBERT_scores(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_gmn_with_mlp_and_colbert_objective"):
    logger.info("Loading model GMN_with_MLP_and_ColBERT_scores")  
    logger.info("This uses GMN encoder to obtain set of node embeddings per graph processed with a gated MLP, then subjects that to ColBERT loss") 
    #One more hack. 
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = GMN_embed_with_MLP_and_ColBERT_scores(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_gmn_with_maxsim_dot_corrected"):
    logger.info("Loading model GMN_embed_maxsim_dot_corrected")  
    logger.info("This uses GMN encoder to obtain set of node embeddings per graph processed with a gated MLP, then subjects that to ColBERT loss (maxsim)") 
    #One more hack. 
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = GMN_embed_maxsim_dot_corrected(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  else:
    logger.info("ALERT!! CHECK FOR ERROR")  

  if os.path.exists(os.path.join("initialModelWeights", av.TASK)):
    model.load_state_dict(torch.load(os.path.join("initialModelWeights", av.TASK)))
  save_initial_model(av,model)
  optimizer = torch.optim.Adam(model.parameters(),
                                    lr=av.LEARNING_RATE,
                                    weight_decay=av.WEIGHT_DECAY)  
  cnt =0
  for param in model.parameters():
        cnt=cnt+torch.numel(param)
  logger.info("no. of params in model: %s",cnt)
  es = EarlyStoppingModule(av,av.ES)

  run = 0
  while av.RUN_TILL_ES or run<av.NUM_RUNS:
    masking_iters = run // 30 + 1
    model.train()
    start_time = time.time()
    n_batches = train_data.create_stratified_batches()
    epoch_loss = 0
    start_time = time.time()
    for i in range(n_batches):
      batch_data,batch_data_sizes,target,batch_adj = train_data.fetch_batched_data_by_id(i)
      optimizer.zero_grad()
      prediction = model(batch_data,batch_data_sizes,batch_adj)
      #Pairwise ranking loss
      predPos = prediction[target>0.5]
      predNeg = prediction[target<0.5]
      losses = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), av.MARGIN)
      #losses = torch.nn.functional.mse_loss(target, prediction,reduction="sum")
      losses.backward()
      optimizer.step()
      epoch_loss = epoch_loss + losses.item()

    logger.info("Run: %d train loss: %f Time: %.2f",run,epoch_loss,time.time()-start_time)
    #TODO:SAVE model - If needed
    #if run%10==0:
    #save_model_at_epoch(av,model, run)
    start_time = time.time()
    ap_score,map_score = evaluate_embeddings_similarity(av,model,val_data)
    logger.info("Run: %d VAL ap_score: %.6f map_score: %.6f Time: %.2f",run,ap_score,map_score, time.time()-start_time)
    if av.RUN_TILL_ES:
      if es.check([map_score],model,run):
        break
    run+=1

def test_indra_our(av, config):
  device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
  train_data = OurMatchingModelSubgraphIsoData(av,mode="train")
  av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
  import copy
  copy_config = copy.deepcopy(config)
  model_our = GMN_embed_with_MLP_and_ColBERT_scores(av,copy_config,1).to(device)
  model_indra = GMN_embed_maxsim_dot_corrected(av,config,1).to(device)

  updated_state_dict = model_our.state_dict().copy()
  updated_state_dict["aggregator.MLP1.0.weight"] = updated_state_dict["node_feature_processor.MLP.0.weight"]
  updated_state_dict["aggregator.MLP2.0.weight"] = model_indra.state_dict()["aggregator.MLP2.0.weight"]
  updated_state_dict["aggregator.MLP1.0.bias"] = updated_state_dict["node_feature_processor.MLP.0.bias"]
  updated_state_dict["aggregator.MLP2.0.bias"] = model_indra.state_dict()["aggregator.MLP2.0.bias"]
  updated_state_dict.pop("node_feature_processor.MLP.0.weight")
  updated_state_dict.pop("node_feature_processor.MLP.0.bias")
  model_indra.load_state_dict(updated_state_dict)

  torch.save(model_our.state_dict(), "initialModelWeights/matching_iso_var_gmn_with_mlp_and_colbert_objective")
  torch.save(model_indra.state_dict(), "initialModelWeights/matching_iso_var_gmn_with_maxsim_dot_corrected")

  exit()

  train_data.data_type = "gmn"
  for batch_idx in range(1000):
    batch_data, batch_data_sizes, _, batch_adj = train_data.fetch_batched_data_by_id(batch_idx)
    scores_our = model_our(batch_data,batch_data_sizes,batch_adj)
    scores_indra = model_indra(batch_data,batch_data_sizes,batch_adj)
    score_delta = torch.sum((scores_our - scores_indra)**2)
    print(scores_our, scores_indra)
    if score_delta > 1e-10:
      print("Mismatch found", score_delta)
      exit(0)

def compare_hungarian_with_normal(av, config):
  device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
  train_data = OurMatchingModelSubgraphIsoData(av,mode="train")
  av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
  import copy
  copy_config = copy.deepcopy(config)
  model_1 = Node_align_Node_loss(av,copy_config,1).to(device)
  model_2 = Hungarian_Node_align_Node_loss(av,config,1).to(device)

  model_1_state_dict_copy = model_1.state_dict().copy()
  model_2_state_dict_copy = model_2.state_dict().copy()
  for key in model_2_state_dict_copy.keys():
    if "asymm_" in key:
      model_2_state_dict_copy[key] = model_1_state_dict_copy[key.replace("asymm_", "")]
  model_2.load_state_dict(model_2_state_dict_copy)

  n_batches = train_data.create_stratified_batches()
  train_data.data_type = "gmn"
  for batch_idx in range(1000):
    model_1.train()
    model_2.train()
    batch_data, batch_data_sizes, _, batch_adj = train_data.fetch_batched_data_by_id(batch_idx)
    scores_1 = model_1(batch_data,batch_data_sizes,batch_adj)
    scores_2 = model_2(batch_data,batch_data_sizes,batch_adj)
    score_delta = torch.sum((scores_1 - scores_2)**2)
    if score_delta > 1e-10:
      # Mismatch because of randomness in sinkhorn_iters
      print("Mismatch found", batch_idx, score_delta)
      exit(0)

def plot_hungarian_graph_pairs(av, config):
  device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
  train_data = OurMatchingModelSubgraphIsoData(av,mode="train")
  av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
  import copy
  copy_config = copy.deepcopy(config)
  hungarian_model = Hungarian_Node_align_Node_loss(av,copy_config,1).to(device)
  hungarian_model.load_state_dict(torch.load("/mnt/home/ashwinr/btp24/grph/ISONET/bestValidationModels/hungarian_node_align_node_loss_truncated_transport_plan_ptc_fm_0")['model_state_dict'])

  n_batches = train_data.create_stratified_batches()
  train_data.data_type = "gmn"
  for batch_idx in range(1000):
    batch_data, batch_data_sizes, _, batch_adj = train_data.fetch_batched_data_by_id(batch_idx)
    hungarian_model.visualize([0, 1, 2, 3],batch_data,batch_data_sizes,batch_adj)
    input()

def seed_everything(seed: int):
    import random, os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--logpath",                        type=str,   default="logDir/logfile",help="/path/to/log")
  ap.add_argument("--want_cuda",                      type=bool,  default=True)
  ap.add_argument("--RUN_TILL_ES",                    type=bool,  default=True)
  ap.add_argument("--has_cuda",                       type=bool,  default=torch.cuda.is_available())
  ap.add_argument("--is_sig",                         type=bool,  default=False)
  ap.add_argument("--ES",                             type=int,   default=50)
  ap.add_argument("--MIN_QUERY_SUBGRAPH_SIZE",        type=int,   default=5)
  ap.add_argument("--MAX_QUERY_SUBGRAPH_SIZE",        type=int,   default=10)
  ap.add_argument("--MIN_CORPUS_SUBGRAPH_SIZE",       type=int,   default=11)
  ap.add_argument("--MAX_CORPUS_SUBGRAPH_SIZE",       type=int,   default=15)
  ap.add_argument("--MAX_GRAPH_SIZE",                 type=int,   default=0)
  ap.add_argument("--n_layers",                       type=int,   default=3)
  ap.add_argument("--conv_type",                      type=str,   default='SAGE')
  ap.add_argument("--method_type",                    type=str,   default='order')
  ap.add_argument("--skip",                           type=str,   default='learnable')
  ap.add_argument("--neuromatch_hidden_dim",          type=int,   default=10)
  ap.add_argument("--post_mp_dim",                    type=int,   default=64)
  ap.add_argument("--filters_1",                      type=int,   default=128)
  ap.add_argument("--filters_2",                      type=int,   default=64)
  ap.add_argument("--filters_3",                      type=int,   default=10)
  ap.add_argument("--dropout",                        type=float, default=0)
  ap.add_argument("--tensor_neurons",                 type=int,   default=16)
  ap.add_argument("--time_updates",                   type=int,   default=3)
  ap.add_argument("--time_update_idx",                type=str,   default="k_t")
  ap.add_argument('--prop_separate_params',           action=argparse.BooleanOptionalAction)
  ap.add_argument("--transform_dim" ,                 type=int,   default=10)
  ap.add_argument("--bottle_neck_neurons",            type=int,   default=16)
  ap.add_argument("--bins",                           type=int,   default=16)
  ap.add_argument("--histogram",                      type=bool,  default=False)
  ap.add_argument("--GMN_NPROPLAYERS",                type=int,   default=5)
  ap.add_argument("--MASKING_FOR_MSG_PASSING_COUNT",  type=int,   default=5)
  ap.add_argument("--MARGIN",                         type=float, default=0.1)
  ap.add_argument("--NOISE_FACTOR",                   type=float, default=1.0)
  ap.add_argument("--NUM_RUNS",                       type=int,   default=2)
  ap.add_argument("--BATCH_SIZE",                     type=int,   default=128)
  ap.add_argument("--LEARNING_RATE",                  type=float, default=0.001)
  ap.add_argument("--WEIGHT_DECAY",                   type=float, default=5*10**-4)
  ap.add_argument("--FEAT_TYPE",                      type=str,   default="Onehot1",help="One/Onehot/Onehot1/Adjrow/Adjrow1/AdjOnehot")
  ap.add_argument("--CONV",                           type=str,   default="GCN",help="GCN/GAT/GIN/SAGE")
  ap.add_argument("--DIR_PATH",                       type=str,   default=".",help="path/to/datasets")
  ap.add_argument("--DATASET_NAME",                   type=str,   default="mutag", help="TODO")
  ap.add_argument("--TASK",                           type=str,   default="OurMatchingSimilarity",help="TODO")
  ap.add_argument("--SEED",                           type=int,   default=0)

  av = ap.parse_args()

  if av.FEAT_TYPE == "Adjrow" or  av.FEAT_TYPE == "Adjrow1" or av.FEAT_TYPE == "AdjOnehot": 
      av.TASK = av.TASK + "_" + av.FEAT_TYPE
  if av.CONV != "GCN": 
      av.TASK = av.TASK + "_" + av.CONV
  av.logpath = av.logpath+"_"+av.TASK+"_"+av.DATASET_NAME+"_"+str(av.SEED)+"_"+str(datetime.now()).replace(" ", "_")
  set_log(av)
  logger.info("Command line")
  logger.info('\n'.join(sys.argv[:]))

  # Print configure
  config = get_default_config()
  config['seed'] = av.SEED
  config['encoder'] ['node_hidden_sizes'] = [10]
  config['encoder'] ['node_feature_dim'] = 1
  config['encoder'] ['edge_feature_dim'] = 1
  config['aggregator'] ['node_hidden_sizes'] = [10]
  config['aggregator'] ['graph_transform_sizes'] = [10]
  config['aggregator'] ['input_size'] = [10]
  config['graph_matching_net'] ['node_state_dim'] = 10
  #config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['edge_hidden_sizes'] = [20]
  config['graph_matching_net'] ['node_hidden_sizes'] = [10]
  config['graph_matching_net'] ['n_prop_layers'] = 5
  config['graph_embedding_net'] ['node_state_dim'] = 10
  #config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_embedding_net'] ['edge_hidden_sizes'] = [20]
  config['graph_embedding_net'] ['node_hidden_sizes'] = [10]
  config['graph_embedding_net'] ['n_prop_layers'] = 5
  config['graph_embedding_net'] ['n_prop_layers'] = 5
  config['temporal_gnn'] = {
    'n_time_updates': av.time_updates,
    'time_update_idx': av.time_update_idx,
    'prop_separate_params': av.prop_separate_params
  }
  config['fringe_isonet'] ['masking_for_msg_passing_count'] = av.MASKING_FOR_MSG_PASSING_COUNT
  
  #logger.info("av gmn_prop_param")
  #logger.info(av.GMN_NPROPLAYERS) 
  #logger.info("config param")
  #logger.info(config['graph_embedding_net'] ['n_prop_layers'] )
  config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  #logger.info("config param")
  #logger.info(config['graph_embedding_net'] ['n_prop_layers'] )

  config['training']['batch_size']  = av.BATCH_SIZE
  config['training']['margin']  = av.MARGIN
  config['evaluation']['batch_size']  = av.BATCH_SIZE
  config['model_type']  = "embedding"
  config['graphsim'] = {} 
  config['graphsim']['conv_kernel_size'] = [10,4,2]
  config['graphsim']['linear_size'] = [24, 16]
  config['graphsim']['gcn_size'] = [10,10,10]
  config['graphsim']['conv_pool_size'] = [3,3,2]
  config['graphsim']['conv_out_channels'] = [2,4,8]
  config['graphsim']['dropout'] = av.dropout 

  for (k, v) in config.items():
      logger.info("%s= %s" % (k, v))  

  # Set random seeds
  seed = config['seed']
  seed_everything(seed)

  av.dataset = av.DATASET_NAME
  train(av, config)
  # plot_hungarian_graph_pairs(av,config)
  # compare_hungarian_with_normal(av, config)

