import os
import torch
import argparse
import numpy as np
import pandas as pd
from subgraph.utils import cudavar
from subgraph import iso_matching_models as im
from GMN.configure import get_default_config
from subgraph.earlystopping import EarlyStoppingModule
from sklearn.metrics import average_precision_score, ndcg_score

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_config(av): 
  config = get_default_config()

  config['seed'] = av.SEED

  config['encoder'] ['node_hidden_sizes'] = [10]
  config['encoder'] ['node_feature_dim'] = 1
  config['encoder'] ['edge_feature_dim'] = 1
    
  config['aggregator'] ['node_hidden_sizes'] = [10]
  config['aggregator'] ['graph_transform_sizes'] = [10]
  config['aggregator'] ['input_size'] = [10]

  config['graph_matching_net'] ['node_state_dim'] = 10
  config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['edge_hidden_sizes'] = [20]
  config['graph_matching_net'] ['node_hidden_sizes'] = [10]
    
  config['graph_embedding_net'] ['node_state_dim'] = 10
  config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_embedding_net'] ['edge_hidden_sizes'] = [20]
  config['graph_embedding_net'] ['node_hidden_sizes'] = [10]
  config['fringe_isonet'] ['masking_for_msg_passing_count'] = av.MASKING_FOR_MSG_PASSING_COUNT

  config['temporal_gnn'] = {
    'n_time_updates': av.time_updates,
    'time_update_idx': av.time_update_idx,
    'prop_separate_params': av.prop_separate_params
  }
  
  config['graphsim']= {}
  config['graphsim']['conv_kernel_size'] = [10,4,2]
  config['graphsim']['linear_size'] = [24, 16]
  config['graphsim']['gcn_size'] = [10,10,10]
  config['graphsim']['conv_pool_size'] = [3,3,2]
  config['graphsim']['conv_out_channels'] = [2,4,8]
  config['graphsim']['dropout'] = av.dropout

  config['training']['batch_size']  = av.BATCH_SIZE
  config['training']['margin']  = av.MARGIN
  config['evaluation']['batch_size']  = av.BATCH_SIZE
  config['model_type']  = "embedding"
    
  return config

def evaluate_embeddings_similarity_map_mrr_mndcg(av,model,sampler):
  model.eval()
  d_pos = sampler.list_pos
  d_neg = sampler.list_neg

  d = d_pos + d_neg
  npos = len(d_pos)
  nneg = len(d_neg)

  pred = []

  n_batches = sampler.create_batches(d)
  for i in range(n_batches):
    #ignoring target values here since not needed for AP ranking score 
    batch_data,batch_data_sizes,_,batch_adj = sampler.fetch_batched_data_by_id(i)
    pred.append( model(batch_data,batch_data_sizes,batch_adj).data)

  all_pred = torch.cat(pred,dim=0) 
  labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
  ap_score   = average_precision_score(labels.cpu(), all_pred.cpu())
  so = np.argsort(all_pred.cpu()).tolist()[::-1]
  labels_rearranged = labels.cpu()[so]
  rr = 1/(labels_rearranged.tolist().index(1)+1)
  ndcg = ndcg_score([labels.cpu().tolist()],[all_pred.cpu().tolist()])

  q_graphs = list(range(len(sampler.query_graphs)))   
    
  all_ap, all_rr, all_ndcg = [], [], []

  for q_id in q_graphs:
    dpos = list(filter(lambda x:x[0][0]==q_id,d_pos))
    dneg = list(filter(lambda x:x[0][0]==q_id,d_neg))
    npos = len(dpos)
    nneg = len(dneg)
    d = dpos+dneg
    if npos>0 and nneg>0:    
      #Damn
      n_batches = sampler.create_batches(d) 
      pred = []  
      for i in range(n_batches):
        #ignoring known ged values here since not needed for AP ranking score 
        batch_data,batch_data_sizes,_,batch_adj = sampler.fetch_batched_data_by_id(i)
        pred.append( model(batch_data,batch_data_sizes,batch_adj).data)
      all_pred = torch.cat(pred,dim=0) 
      labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
      ap   = average_precision_score(labels.cpu(), all_pred.cpu()) 
      all_ap.append(ap)
      so = np.argsort(all_pred.cpu()).tolist()[::-1]
      labels_rearranged = labels.cpu()[so]
      all_rr.append(1/(labels_rearranged.tolist().index(1)+1))
      all_ndcg.append(ndcg_score([labels.cpu().tolist()],[all_pred.cpu().tolist()]))
  return ap_score, np.mean(all_ap), np.std(all_ap), rr, np.mean(all_rr), np.std(all_rr), ndcg, np.mean(all_ndcg), np.std(all_ndcg), all_ap, all_rr

def fetch_gmn_data(av):
    data_mode = "test" if av.test_size==25 else "Extra_test_300"
    test_data = im.OurMatchingModelSubgraphIsoData(av,mode=data_mode)
    val_data = im.OurMatchingModelSubgraphIsoData(av,mode="val")
    test_data.data_type = "pyg"
    val_data.data_type = "pyg"
    return val_data, test_data

def get_result(av,model_loc,state_dict):
    val_data, test_data = fetch_gmn_data(av)
    if model_loc.startswith("node_align_node_loss"):
      config = load_config(av)
      model = im.Node_align_Node_loss(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("temporal_gnn"):
      config = load_config(av)
      model = im.TemporalGNN(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    else:
      print("ALERT!! CHECK FOR ERROR")  
    model.eval()
    model.load_state_dict(state_dict)
    model.load_state_dict(state_dict)
    val_result = evaluate_embeddings_similarity_map_mrr_mndcg(av,model,val_data)
    test_result = evaluate_embeddings_similarity_map_mrr_mndcg(av,model,test_data)

    return val_result, test_result

ap = argparse.ArgumentParser()
ap.add_argument("--MASKING_FOR_MSG_PASSING_COUNT", type=int, default=5)
ap.add_argument("--seeds", type=int, nargs='+')
ap.add_argument("--datasets", type=str, nargs='+')
ap.add_argument("--tasks", type=str, nargs='+')
ad = ap.parse_args()

av = Namespace(   want_cuda                    = True,
                  has_cuda                   = torch.cuda.is_available(),
                  use_pairnorm               = False,
                  is_sig                     = False,
                  n_layers                   = 3,
                  conv_type                  = 'SAGE',
                  method_type                = 'order',
                  skip                       = 'learnable',
                  MIN_QUERY_SUBGRAPH_SIZE    = 5,
                  MAX_QUERY_SUBGRAPH_SIZE    = 10,
                  MIN_CORPUS_SUBGRAPH_SIZE   = 11,
                  MAX_CORPUS_SUBGRAPH_SIZE   = 15,
                  DIR_PATH                   =".",
                  DATASET_NAME               = "ptc_fr",
                  RUN_TILL_ES                = True,
                  ES                         = 50,
                  transform_dim              = 16,
                  GMN_NPROPLAYERS            = 5,
                  FEAT_TYPE                  = "One",
                  filters_1                  = 10,
                  filters_2                  = 10,
                  filters_3                  = 10,
                  time_updates               = 3,
                  time_update_idx            = "k_t",
                  neuromatch_hidden_dim      = 10,
                  post_mp_dim                = 64,
                  bottle_neck_neurons        = 10,
                  tensor_neurons             = 10,               
                  dropout                    = 0,
                  bins                       = 16,
                  histogram                  = False,
                  WEIGHT_DECAY               =5*10**-4,
                  BATCH_SIZE                 =128,
                  LEARNING_RATE              =0.001,
                  CONV                       = "GCN",
                  MARGIN                     = 0.1,
                  NOISE_FACTOR               = 0,
                  NUM_RUNS                   = 2,
                  TASK                       = "",
                  test_size                  = 300,
                  SEED                       = 0,
                  MASKING_FOR_MSG_PASSING_COUNT = ad.MASKING_FOR_MSG_PASSING_COUNT,
              )


task_dict = {} 

task_dict['node_align_node_loss'] = "Node Align Node Loss"
for idx in range(1, 6):
  task_dict[f'temporal_gnn_kt_{idx}_iters_shared_params'] = f"TemporalGNN ({idx} iters)"
datasets = ["aids", "mutag", "ptc_fr", "ptc_fm", "ptc_mr", "ptc_mm"]
test_models = "experiments/temporal_gnn_shared_params/models"

scores = {}
for model_loc in os.listdir(test_models):
    print(model_loc)
    found = False
    model = None
    for task in task_dict.keys():
       if model_loc.startswith(task):
          model = task
    if model not in scores.keys():
       scores[model] = {}
    saved = torch.load(os.path.join(test_models, model_loc))
    av = saved['av']
    av.test_size = 25
    av.prop_separate_params = False
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    model_state_dict = saved['model_state_dict']
    dataset = av.DATASET_NAME
    print("dataset", dataset)
    scores[model][dataset] = get_result(av,model_loc,model_state_dict)[1][1]
    print(scores[model][dataset])
    import pickle
    pickle.dump(scores, open('scores.pkl', 'wb'))

print(scores)