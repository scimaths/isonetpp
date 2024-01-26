import os
import sys
import time
import torch
import random
import argparse
import numpy as np

from datetime import datetime
from common import logger, set_log
from GMN.configure import get_default_config
from subgraph.utils import save_initial_model

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)

from subgraph.earlystopping import EarlyStoppingModule
from subgraph.dataset import OurMatchingModelSubgraphIsoData

# Import models
from subgraph.models.gotsim import GOTSim
from subgraph.models.simgnn import SimGNN
from subgraph.models.graphsim import GraphSim
from subgraph.models.neuromatch import NeuroMatch
from subgraph.models.node_early_interaction import NodeEarlyInteraction
from subgraph.models.edge_early_interaction import EdgeEarlyInteraction
from subgraph.models.edge_early_interaction_with_delete import EdgeEarlyInteractionDelete
from subgraph.models.edge_early_interaction_baseline import EdgeEarlyInteractionBaseline
from subgraph.models.node_early_interaction_baseline import NodeEarlyInteractionBaseline
from subgraph.models.node_edge_early_interaction_with_consistency import NodeEdgeEarlyInteractionWithConsistency
from subgraph.models.node_edge_early_interaction_with_consistency_and_two_sinkhorns import NodeEdgeEarlyInteractionWithConsistencyAndTwoSinkhorns
from subgraph.models.adding_to_q import AddingToQ
from subgraph.models.velugoti_39 import OurMatchingModelVar39_GMN_encoding_NodePerm_SinkhornParamBig_HingeScore_EdgePermConsistency
from subgraph.models.velugoti_45 import OurMatchingModelVar45_GMN_encoding_NodeAndEdgePerm_SinkhornParamBig_HingeScore
from subgraph.models.isonet import ISONET, ISONET_Sym
from subgraph.models.gmn_match import GMN_match, GMN_match_hinge, GMN_match_hinge_baseline
from subgraph.models.gmn_match_hinge_scoring import GMN_match_hinge_scoring, GMN_match_hinge_scoring_sinkhorn, GMN_match_hinge_colbert, GMN_match_hinge_scoring_injective_attention
from subgraph.models.gmn_match_hinge_lrl import GMN_match_hinge_lrl, GMN_match_hinge_lrl_scoring, GMN_match_hinge_hinge_similarity, GMN_match_hinge_hinge_similarity_scoring, GMN_match_hinge_replicated, GMN_match_hinge_injective_attention
from subgraph.models.gmn_match_hinge_lrl_sinkhorn import GMN_match_hinge_lrl_sinkhorn, GMN_match_hinge_lrl_scoring_sinkhorn, GMN_match_hinge_hinge_similarity_sinkhorn, GMN_match_hinge_hinge_similarity_scoring_sinkhorn, GMN_match_hinge_lrl_scoring_sinkhorn_inter
from subgraph.models.gmn_match_hinge_lrl_injective_attention import GMN_match_hinge_lrl_injective_attention, GMN_match_hinge_lrl_scoring_injective_attention, GMN_match_hinge_hinge_similarity_injective_attention, GMN_match_hinge_hinge_similarity_scoring_injective_attention, GMN_match_hinge_lrl_scoring_injective_attention_inter
from subgraph.models.vaibhav import GMN_match_hinge_vaibhav, GMN_match_hinge_vaibhav_injective_attention
from subgraph.models.node_align_node_loss import Node_align_Node_loss
from subgraph.models.node_align_edge_loss import Node_align_Edge_loss
from subgraph.models.hungarian_node_align import Hungarian_Node_align_Node_loss
from subgraph.models.fringed_node_align_node_loss import Fringed_node_align_Node_loss
from subgraph.models.node_early_interaction_interpretability import NodeEarlyInteractionInterpretability
from subgraph.models.gmn_embed import GMN_embed, GMN_embed_hinge_scoring, GMN_embed_hinge, GMN_embed_with_ColBERT_scores, GMN_embed_with_MLP_and_ColBERT_scores, GMN_embed_maxsim_dot, GMN_embed_maxsim_dot_corrected
from subgraph.models.node_early_interaction_with_consistency import NodeEarlyInteractionWithConsistency

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
  elif av.TASK.startswith("gmn_match_hinge_lrl_scoring_injective_attention_inter"):
    logger.info("Loading model GMN Match Hinge, LRL scoring, Injective Attention, Interaction features")  
    model = GMN_match_hinge_lrl_scoring_injective_attention_inter(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_injective_attention"):
    logger.info("Loading model GMN Match Hinge Injective Attention")  
    model = GMN_match_hinge_injective_attention(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_replicated"):
    logger.info("Loading model gmn_match_hinge_replicated")  
    model = GMN_match_hinge_replicated(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_lrl_scoring_sinkhorn_inter"):
    logger.info("Loading model gmn_match_hinge_lrl_scoring_sinkhorn_inter")  
    model = GMN_match_hinge_lrl_scoring_sinkhorn_inter(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_colbert"):
    logger.info("Loading model GMN_match_hinge_colbert")  
    model = GMN_match_hinge_colbert(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_embed_hinge_scoring"):
    logger.info("Loading model GMN_embed_hinge_scoring")  
    model = GMN_embed_hinge_scoring(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn" 
  elif av.TASK.startswith("gmn_match_hinge_vaibhav_injective_attention"):
    logger.info("Loading model GMN_match_hinge_vaibhav_injective_attention")  
    model = GMN_match_hinge_vaibhav_injective_attention(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_vaibhav"):
    logger.info("Loading model GMN_match_hinge_vaibhav")  
    model = GMN_match_hinge_vaibhav(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_lrl_scoring_injective_attention"):
    logger.info("Loading model GMN Match Hinge lrl_scoring")  
    model = GMN_match_hinge_lrl_scoring_injective_attention(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"  
  elif av.TASK.startswith("gmn_match_hinge_lrl_injective_attention"):
    logger.info("Loading model GMN Match Hinge lrl")  
    model = GMN_match_hinge_lrl_injective_attention(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_hinge_similarity_scoring_injective_attention"):
    logger.info("Loading model GMN Match Hinge hinge_similarity")  
    model = GMN_match_hinge_hinge_similarity_scoring_injective_attention(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_hinge_similarity_injective_attention"):
    logger.info("Loading model GMN Match Hinge hinge_similarity")  
    model = GMN_match_hinge_hinge_similarity_injective_attention(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"  
  elif av.TASK.startswith("gmn_match_hinge_scoring_injective_attention"):
    logger.info("Loading model GMN Match Hinge baseline scoring")  
    model = GMN_match_hinge_scoring_injective_attention(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_lrl_scoring_sinkhorn"):
    logger.info("Loading model GMN Match Hinge lrl_scoring")  
    model = GMN_match_hinge_lrl_scoring_sinkhorn(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"  
  elif av.TASK.startswith("gmn_match_hinge_lrl_sinkhorn"):
    logger.info("Loading model GMN Match Hinge lrl")  
    model = GMN_match_hinge_lrl_sinkhorn(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_hinge_similarity_scoring_sinkhorn"):
    logger.info("Loading model GMN Match Hinge hinge_similarity")  
    model = GMN_match_hinge_hinge_similarity_scoring_sinkhorn(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"  
  elif av.TASK.startswith("gmn_match_hinge_hinge_similarity_sinkhorn"):
    logger.info("Loading model GMN Match Hinge hinge_similarity")  
    model = GMN_match_hinge_hinge_similarity_sinkhorn(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"  
  elif av.TASK.startswith("gmn_match_hinge_scoring_sinkhorn"):
    logger.info("Loading model GMN Match Hinge baseline scoring")  
    model = GMN_match_hinge_scoring_sinkhorn(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_lrl_scoring"):
    logger.info("Loading model GMN Match Hinge lrl_scoring")  
    model = GMN_match_hinge_lrl_scoring(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"  
  elif av.TASK.startswith("gmn_match_hinge_lrl"):
    logger.info("Loading model GMN Match Hinge lrl")  
    model = GMN_match_hinge_lrl(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_hinge_similarity_scoring"):
    logger.info("Loading model GMN Match Hinge hinge_similarity")  
    model = GMN_match_hinge_hinge_similarity_scoring(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"  
  elif av.TASK.startswith("gmn_match_hinge_hinge_similarity"):
    logger.info("Loading model GMN Match Hinge hinge_similarity")  
    model = GMN_match_hinge_hinge_similarity(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"  
  elif av.TASK.startswith("gmn_match_hinge_scoring"):
    logger.info("Loading model GMN Match Hinge baseline scoring")  
    model = GMN_match_hinge_scoring(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge_baseline"):
    logger.info("Loading model GMN Match Hinge baseline")  
    model = GMN_match_hinge_baseline(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("gmn_match_hinge"):
    logger.info("Loading model GMN Match Hinge")  
    model = GMN_match_hinge(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("graphsim"):
    logger.info("Loading model GraphSim")  
    logger.info("This is GraphSim model")  
    model = GraphSim(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "pyg"
    val_data.data_type = "pyg"
  elif av.TASK.startswith("gotsim"):
    logger.info("Loading IR_modified_GotSim")  
    logger.info("This uses IR_modified_GotSim  model. ")  
    model = GOTSim(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "pyg"
    val_data.data_type = "pyg"
  elif av.TASK.startswith("simgnn") :
    logger.info("Loading model SimGNN")
    logger.info("This loads the entire SimGNN model. Input feature is [1]. No node permutation is done after nx graph loading")
    model = SimGNN(av,1).to(device)
    train_data.data_type = "pyg"
    val_data.data_type = "pyg"
  elif av.TASK.startswith("gmn_embed_hinge"):
    logger.info("Loading GMN_embed_hinge")  
    logger.info("This uses GMN embedding model with hinge loss.No regularizer. ")  
    model = GMN_embed_hinge(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("neuromatch"): 
    logger.info("Loading model neuromatch")   
    logger.info("This is neuromatch model")   
    model = NeuroMatch(1,av.neuromatch_hidden_dim,av).to(device) 
    logger.info(model) 
    train_data.data_type = "pyg" 
    val_data.data_type = "pyg" 
  elif av.TASK.startswith("node_early_interaction_with_consistency"):
    logger.info("Loading model node_early_interaction_with_consistency")
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                       max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = NodeEarlyInteractionWithConsistency(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("node_edge_early_interaction_with_consistency_and_two_sinkhorns"):
    logger.info("Loading model node_edge_early_interaction_with_consistency_and_two_sinkhorns")
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                       max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = NodeEdgeEarlyInteractionWithConsistencyAndTwoSinkhorns(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("node_edge_early_interaction_with_consistency"):
    logger.info("Loading model node_edge_early_interaction_with_consistency")
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                       max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = NodeEdgeEarlyInteractionWithConsistency(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("node_early_interaction_interpretability"):
    logger.info("Loading model node_early_interaction_interpretability")  
    model = NodeEarlyInteractionInterpretability(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("node_early_interaction_baseline"):
    logger.info("Loading model NodeEarlyInteractionBaseline")
    logger.info("This model implements early interaction for nodes baseline")
    model = NodeEarlyInteractionBaseline(av, config, 1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("node_early_interaction"):
    logger.info("Loading model NodeEarlyInteraction")
    logger.info("This model implements early interaction for nodes")
    model = NodeEarlyInteraction(av, config, 1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("adding_to_q"):
    logger.info("Loading model AddingTOQ")
    logger.info("This model implements adding to query")
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = AddingToQ(av, config, 1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("edge_early_interaction_with_delete"):
    logger.info("Loading model EdgeEarlyInteractionDelete")
    logger.info("This model implements early interaction for edges delete")
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = EdgeEarlyInteractionDelete(av, config, 1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("edge_early_interaction_baseline"):
    logger.info("Loading model EdgeEarlyInteractionBaseline")
    logger.info("This model implements early interaction for edges baseline")
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = EdgeEarlyInteractionBaseline(av, config, 1).to(device)
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
  elif av.TASK.startswith("nanl_consistency_45"):
    logger.info("Loading model OurMatchingModelVar45_GMN_encoding_NodeAndEdgePerm_SinkhornParamBig_HingeScore")
    logger.info("This uses GMN encoder followed by parameterized sinkhorn with LRL and similarity computation using hinge scoring (H_q, PH_c) .We're taking edge perm from node perm kronecker product and the checking edge consistency with edge embeddings")
    #One more hack.
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = OurMatchingModelVar45_GMN_encoding_NodeAndEdgePerm_SinkhornParamBig_HingeScore(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
    av.store_epoch_info = False
  elif av.TASK.startswith("nanl_consistency"):
    logger.info("Loading model OurMatchingModelVar39_GMN_encoding_NodePerm_SinkhornParamBig_HingeScore_EdgePermConsistency")
    logger.info("This uses GMN encoder followed by parameterized sinkhorn with LRL and similarity computation using hinge scoring (H_q, PH_c) .We're taking edge perm from node perm kronecker product and the checking edge consistency with edge embeddings")
    #One more hack. 
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = OurMatchingModelVar39_GMN_encoding_NodePerm_SinkhornParamBig_HingeScore_EdgePermConsistency(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
    av.store_epoch_info = False
  else:
    logger.info("ALERT!! CHECK FOR ERROR")  
  # for name, param in model.named_parameters():
  #   if param.requires_grad:
  #       print(name, param.data.shape)
  #       if 'fc_combine_interaction' in name and 'bias' in name:
  #         print(param.data)
  # exit(0)
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
    epoch_loss_pair = 0
    epoch_consistency_loss2 = 0
    epoch_scores_node_align = 0
    epoch_scores_kronecker_edge_align = 0
    epoch_consistency_loss3 = 0
    epoch_consistency_loss4 = 0
    epoch_consistency_loss2_pos = 0
    epoch_consistency_loss3_pos = 0
    epoch_consistency_loss4_pos = 0
    epoch_consistency_loss2_neg = 0
    epoch_consistency_loss3_neg = 0
    epoch_consistency_loss4_neg = 0

    start_time = time.time()
    for i in range(n_batches):
      batch_data,batch_data_sizes,target,batch_adj = train_data.fetch_batched_data_by_id(i)
      optimizer.zero_grad()
      if av.output_type == 1:
        prediction = model(batch_data,batch_data_sizes,batch_adj)
      elif av.output_type == 2:
        outputs = model(batch_data,batch_data_sizes,batch_adj)

        prediction = outputs[0]
        consistency_loss2 = torch.sum(outputs[1])/outputs[1].shape[0]
        consistency_loss3 = torch.sum(outputs[2])/outputs[2].shape[0]
        consistency_loss4 = torch.sum(outputs[5])/outputs[5].shape[0]
        epoch_consistency_loss2 += consistency_loss2.item()
        epoch_consistency_loss3 += consistency_loss3.item()
        epoch_consistency_loss4 += consistency_loss4.item()

        scores_node_align = torch.sum(outputs[3])/outputs[3].shape[0]
        scores_kronecker_edge_align = torch.sum(outputs[4])/outputs[4].shape[0]
        epoch_scores_node_align += scores_node_align.item()
        epoch_scores_kronecker_edge_align += scores_kronecker_edge_align.item()

        consistency_loss2_pos = torch.sum(outputs[1][target==1])/outputs[1][target==1].shape[0]
        consistency_loss3_pos = torch.sum(outputs[2][target==1])/outputs[2][target==1].shape[0]
        consistency_loss4_pos = torch.sum(outputs[5][target==1])/outputs[5][target==1].shape[0]
        epoch_consistency_loss2_pos += consistency_loss2_pos.item()
        epoch_consistency_loss3_pos += consistency_loss3_pos.item()
        epoch_consistency_loss4_pos += consistency_loss4_pos.item()
      
        consistency_loss2_neg = torch.sum(outputs[1][target==0])/outputs[1][target==0].shape[0]
        consistency_loss3_neg = torch.sum(outputs[2][target==0])/outputs[2][target==0].shape[0]
        consistency_loss4_neg = torch.sum(outputs[5][target==0])/outputs[5][target==0].shape[0]
        epoch_consistency_loss2_neg += consistency_loss2_neg.item()
        epoch_consistency_loss3_neg += consistency_loss3_neg.item()
        epoch_consistency_loss4_neg += consistency_loss4_neg.item()

      #Pairwise ranking loss
      predPos = prediction[target>0.5]
      predNeg = prediction[target<0.5]
      if av.loss_type == 1:
          losses = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), av.MARGIN)
      elif av.loss_type == 3:
          losses = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), av.MARGIN) + \
                    (av.loss_lambda * consistency_loss2)
      elif av.loss_type == 4:
          losses = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), av.MARGIN) + \
                    (av.loss_lambda * consistency_loss3)
      elif av.loss_type == 5:
          losses = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), av.MARGIN) + \
                    (av.loss_lambda * consistency_loss4)
          epoch_loss_pair += pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), av.MARGIN).item()

      losses.backward()
      optimizer.step()
      epoch_loss = epoch_loss + losses.item()

    logger.info("Run: %d train loss: %f Time: %.2f",run,epoch_loss,time.time()-start_time)
    if av.output_type == 2:
      logger.info("Run: %d train pair similarity score: %f",run, epoch_loss_pair/n_batches)
      logger.info("Run: %d train scores node align: %f, scores kronecker align: %f",run,epoch_scores_node_align/n_batches, epoch_scores_kronecker_edge_align/n_batches)
      logger.info("Run: %d train consistency_loss2: %f, consistency_loss3: %f, consistency_loss4: %f",run,epoch_consistency_loss2/n_batches, epoch_consistency_loss3/n_batches, epoch_consistency_loss4/n_batches)
      logger.info("Run: %d train consistency_loss2_pos: %f, consistency_loss3_pos: %f, consistency_loss4_pos: %f",run,epoch_consistency_loss2_pos/n_batches, epoch_consistency_loss3_pos/n_batches, epoch_consistency_loss4_pos/n_batches)
      logger.info("Run: %d train consistency_loss2_neg: %f, consistency_loss3_neg: %f, consistency_loss4_neg: %f",run,epoch_consistency_loss2_neg/n_batches, epoch_consistency_loss3_neg/n_batches, epoch_consistency_loss4_neg/n_batches)

    start_time = time.time()
    ap_score, map_score = evaluate_embeddings_similarity(av,model,val_data)
    logger.info("Run: %d VAL ap_score: %.6f map_score: %.6f Time: %.2f", run, ap_score,map_score, time.time()-start_time)
    if av.RUN_TILL_ES:
      if es.check([map_score],model,run):
        break
    run+=1

def seed_everything(seed: int):    
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
  ap.add_argument('--lambd',                          type=float, default=1.0, nargs='?')
  ap.add_argument('--consistency_lambda',             type=float, default=1.0, nargs='?')
  ap.add_argument('--interpretability_lambda',        type=float, default=1.0, nargs='?')
  ap.add_argument("--IPLUS_LAMBDA",                   type=float, default=1)
  ap.add_argument("--no_of_query_subgraphs",          type=int,   default=100)
  ap.add_argument("--no_of_corpus_subgraphs",         type=int,   default=800)
  ap.add_argument("--scores",                         nargs="+", default=["node_align", "kronecker_edge_align"])
  ap.add_argument("--loss_type",                      type=int, default=1)
  ap.add_argument("--output_type",                      type=int, default=1)
  ap.add_argument("--loss_lambda",                    type=float, default=1)
  ap.add_argument("--transport_node_type",            type=str,   default="soft")
  ap.add_argument("--transport_edge_type",            type=str,   default="sinkhorn")
  ap.add_argument("--temp_gmn_scoring",               type=float,   default="1")

  av = ap.parse_args()
  seeds = [4586, 7366, 7474, 7762, 4929, 3543, 1704, 356, 4891, 3133]
  best_seed_dict = {
    'node_early_interaction': {'aids': 7474, 'mutag': 7474, 'ptc_fm': 4929, 'ptc_fr': 7366, 'ptc_mm': 7762, 'ptc_mr': 7366},
    'edge_early_interaction': {'aids': 7474, 'mutag': 7474, 'ptc_fm': 4929, 'ptc_fr': 7366, 'ptc_mm': 7762, 'ptc_mr': 7366},
    'isonet': {'aids': 7762, 'mutag': 4586, 'ptc_fm': 7366, 'ptc_fr': 7474, 'ptc_mm': 7366, 'ptc_mr': 7366},
    'nanl_consistency': {'aids': 7762, 'mutag': 4586, 'ptc_fm': 7366, 'ptc_fr': 7474, 'ptc_mm': 7366, 'ptc_mr': 7366},
    'nanl_consistency_45': {'aids': 7762, 'mutag': 4586, 'ptc_fm': 7366, 'ptc_fr': 7474, 'ptc_mm': 7366, 'ptc_mr': 7366},
    'node_align_node_loss': {'aids': 7762, 'mutag': 4586, 'ptc_fm': 4586, 'ptc_fr': 4929, 'ptc_mm': 7762, 'ptc_mr': 4929},
  }
  seed_accessor = av.TASK if av.TASK in best_seed_dict else 'node_early_interaction'
  av.SEED = best_seed_dict[seed_accessor][av.DATASET_NAME]
  av.time_key = str(datetime.now()).replace(' ', '_')

  exp_name = f"{av.TASK}_{av.DATASET_NAME}_margin_{av.MARGIN}_seed_{av.SEED}_time_{av.time_key}_lambd_{av.lambd}"
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
  config['early_interaction'] = {
    'n_time_updates': av.time_updates,
    'time_update_idx': av.time_update_idx,
    'prop_separate_params': av.prop_separate_params
  }
  config['node_early_interaction_interpretability'] = {
    'lambd' : av.lambd
  }
  config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS

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