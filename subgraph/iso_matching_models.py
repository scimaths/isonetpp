import os
import sys
import time
import torch
import argparse

from datetime import datetime
from logger import logger, set_log
from GMN.configure import get_default_config
from subgraph.utils import save_initial_model

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from utils.earlystopping import EarlyStoppingModule
from subgraph.dataset import OurMatchingModelSubgraphIsoData

# Import models
from subgraph.models.gotsim import GOTSim
from subgraph.models.simgnn import SimGNN
from subgraph.models.graphsim import GraphSim
from subgraph.models.neuromatch import NeuroMatch
from subgraph.models.node_early_interaction import NodeEarlyInteraction
from subgraph.models.edge_early_interaction import EdgeEarlyInteraction
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
  elif av.TASK.startswith("node_early_interaction"):
    logger.info("Loading model NodeEarlyInteraction")
    logger.info("This model implements early interaction for nodes")
    model = NodeEarlyInteraction(av, config, 1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("edge_early_interaction"):
    logger.info("Loading model EdgeEarlyInteraction")
    logger.info("This model implements early interaction for edges")
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = EdgeEarlyInteraction(av, config, 1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("isonet"):
    logger.info("Loading model ISONET")  
    logger.info("This uses GMN encoder to obtain bad of node embeddings per graph, which is then used to obtain bag of edge embeddings per graph - then parameterized sinkhorn with LRL and similarity computation using hinge scoring (H_q, PH_c) where H consists of edge embeds") 
    #One more hack. 
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = ISONET(av,config,1).to(device)
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
    model.to(device)
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
    start_time = time.time()
    ap_score, map_score = evaluate_embeddings_similarity(av,model,val_data)
    logger.info("Run: %d VAL ap_score: %.6f map_score: %.6f Time: %.2f", run, ap_score,map_score, time.time()-start_time)
    if av.RUN_TILL_ES:
      if es.check([map_score],model,run):
        break
    run+=1

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
  ap.add_argument("--experiment_group",               type=str)
  ap.add_argument("--TASK",                           type=str)
  ap.add_argument("--logpath",                        type=str,   default="logDir/",help="/path/to/log")
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
  ap.add_argument("--DATASET_NAME",                   type=str,   default="mutag")
  ap.add_argument("--SEED",                           type=int,   default=0)
  ap.add_argument('--EXPLICIT_SEED',                  type=int,   nargs='?')

  av = ap.parse_args()
  seeds = [4586, 7366, 7474, 7762, 4929, 3543, 1704, 356, 4891, 3133]
  av.SEED = seeds[av.SEED]
  if av.EXPLICIT_SEED is not None:
     av.SEED = av.EXPLICIT_SEED
  av.time_key = str(datetime.now()).replace(' ', '_')
  
  exp_name = f"{av.TASK}_{av.DATASET_NAME}_margin_{av.MARGIN}_seed_{av.SEED}_time_{av.time_key}"
  av.logpath = av.experiment_group + "/" + av.logpath + exp_name
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
  config['graph_matching_net'] ['edge_hidden_sizes'] = [20]
  config['graph_matching_net'] ['node_hidden_sizes'] = [10]
  config['graph_matching_net'] ['n_prop_layers'] = 5
  config['graph_embedding_net'] ['node_state_dim'] = 10
  config['graph_embedding_net'] ['edge_hidden_sizes'] = [20]
  config['graph_embedding_net'] ['node_hidden_sizes'] = [10]
  config['graph_embedding_net'] ['n_prop_layers'] = 5
  config['graph_embedding_net'] ['n_prop_layers'] = 5
  config['early_interaction'] = {
    'n_time_updates': av.time_updates,
    'time_update_idx': av.time_update_idx,
    'prop_separate_params': av.prop_separate_params
  }
  config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS

  config['training']['batch_size']  = av.BATCH_SIZE
  config['training']['margin']  = av.MARGIN
  config['evaluation']['batch_size']  = av.BATCH_SIZE
  config['model_type']  = "embedding"

  for (k, v) in config.items():
      logger.info("%s= %s" % (k, v))  

  # Set random seeds
  seed = config['seed']
  seed_everything(seed)

  av.dataset = av.DATASET_NAME
  
  train(av, config)