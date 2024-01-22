import os
import torch
import pickle
import argparse
import numpy as np
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

  config['early_interaction'] = {
    'n_time_updates': av.time_updates,
    'time_update_idx': av.time_update_idx,
    'prop_separate_params': av.prop_separate_params
  }

  config['node_early_interaction_interpretability'] = {
    'lambd' : av.lambd
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
    if av.output_type == 1:
        pred.append( model(batch_data,batch_data_sizes,batch_adj).data)
    elif av.output_type == 2:
        pred.append( model(batch_data,batch_data_sizes,batch_adj)[0].data)

  all_pred = torch.cat(pred,dim=0) 
  labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
  ap_score = average_precision_score(labels.cpu(), all_pred.cpu())
  so = np.argsort(all_pred.cpu()).tolist()[::-1]
  labels_rearranged = labels.cpu()[so]
  rr = 1/(labels_rearranged.tolist().index(1)+1)
  ndcg = ndcg_score([labels.cpu().tolist()],[all_pred.cpu().tolist()])

  q_graphs = list(range(len(sampler.query_graphs)))   
    
  all_ap, all_rr, all_ndcg, all_hits_20 = [], [], [], []

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
        if av.output_type == 1:
            pred.append( model(batch_data,batch_data_sizes,batch_adj).data)
        elif av.output_type == 2:
            pred.append( model(batch_data,batch_data_sizes,batch_adj)[0].data)         
      all_pred = torch.cat(pred,dim=0) 
      labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
      ap   = average_precision_score(labels.cpu(), all_pred.cpu()) 
      all_ap.append(ap)
      so = np.argsort(all_pred.cpu()).tolist()[::-1]
      labels_rearranged = labels.cpu()[so]
      all_rr.append(1/(labels_rearranged.tolist().index(1)+1))
      all_ndcg.append(ndcg_score([labels.cpu().tolist()],[all_pred.cpu().tolist()]))
      ranking = np.argsort(-all_pred.cpu()).tolist()
      labels_ranked = labels.cpu()[ranking]
      neg_20 = np.where(labels_ranked == 0)[0][min(19, len(labels_ranked == 0) - 1)]
      hits_20 = torch.sum(labels_ranked[:neg_20]) / (torch.sum(labels_ranked))
      all_hits_20.append(hits_20)
  return ap_score, np.mean(all_ap), np.std(all_ap), rr, np.mean(all_rr), np.std(all_rr), ndcg, np.mean(all_ndcg), np.std(all_ndcg), all_ap, all_rr, np.mean(all_hits_20)

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
    elif model_loc.startswith("gmn_match_hinge"):
      config = load_config(av)
      model = im.GMN_match_hinge(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif av.TASK.startswith("graphsim"):
      config = load_config(av)
      model = im.GraphSim(av,config,1).to(device)
      test_data.data_type = "pyg"
      val_data.data_type = "pyg"
    elif av.TASK.startswith("gotsim"):
      config = load_config(av)
      model = im.GOTSim(av,config,1).to(device)
      test_data.data_type = "pyg"
      val_data.data_type = "pyg"
    elif av.TASK.startswith("simgnn") :
      config = load_config(av)
      model = im.SimGNN(av,1).to(device)
      test_data.data_type = "pyg"
      val_data.data_type = "pyg"
    elif av.TASK.startswith("gmn_embed_hinge"):
      config = load_config(av)
      model = im.GMN_embed_hinge(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif av.TASK.startswith("neuromatch"): 
      config = load_config(av)
      model = im.NeuroMatch(1,av.neuromatch_hidden_dim,av).to(device)
      test_data.data_type = "pyg"
      val_data.data_type = "pyg"
    elif model_loc.startswith("node_early_interaction_with_consistency"):
      config = load_config(av)
      model = im.NodeEarlyInteractionWithConsistency(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("node_early_interaction_baseline"):
      config = load_config(av)
      model = im.NodeEarlyInteractionBaseline(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("node_edge_early_interaction_with_consistency_and_two_sinkhorns"):
      config = load_config(av)
      model = im.NodeEdgeEarlyInteractionWithConsistencyAndTwoSinkhorns(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("edge_early_interaction_with_delete"):
      config = load_config(av)
      model = im.EdgeEarlyInteractionDelete(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn" 
    elif model_loc.startswith("edge_early_interaction"):
      config = load_config(av)
      model = im.EdgeEarlyInteraction(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("node_early_interaction_interpretability"):
      config = load_config(av)
      model = im.NodeEarlyInteractionInterpretability(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("isonet"):
      config = load_config(av)
      model = im.ISONET(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("nanl_consistency_45"):
      config = load_config(av)
      model = im.OurMatchingModelVar45_GMN_encoding_NodeAndEdgePerm_SinkhornParamBig_HingeScore(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("nanl_consistency"):
      config = load_config(av)
      model = im.OurMatchingModelVar39_GMN_encoding_NodePerm_SinkhornParamBig_HingeScore_EdgePermConsistency(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("node_early_interaction"):
      config = load_config(av)
      model = im.NodeEarlyInteraction(av,config,1).to(device)
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
ap.add_argument("--model_dir", type=str)
ap.add_argument("--save_loc", type=str)
# ap.add_argument("--seeds", type=int, nargs='+')
# ap.add_argument("--datasets", type=str, nargs='+')
# ap.add_argument("--tasks", type=str, nargs='+')
ad = ap.parse_args()
task_dict = {} 

task_dict['node_edge_early_interaction'] = "Node Edge Early Interaction"
task_dict['edge_early_interaction'] = "Edge Early Interaction"
task_dict['edge_early_interaction_baseline'] = "Delete Edge early"
task_dict['node_early_interaction_baseline'] = "Node Early Interaction"
task_dict['node_align_node_loss'] = "Node Align Node Loss"
task_dict['isonet'] = "ISONET"
task_dict['nanl_consistency'] = "NANL+Consistency"
task_dict['nanl_consistency_45'] = "NANL+Consistency"
task_dict['gmn_match_hinge'] = "GMN Match Hinge"
task_dict['simgnn'] = "simgnn"
task_dict['neuromatch'] = "neuromatch"
task_dict['gotsim'] = "gotsim"
datasets = ["aids", "mutag", "ptc_fr", "ptc_fm", "ptc_mr", "ptc_mm"]
test_model_dir = ad.model_dir

scores = {}
for model_loc in sorted(os.listdir(test_model_dir)):
    found = False
    model = None
    for task in task_dict.keys():
       if model_loc.startswith(task):
          model = task
          break
    if model is None:
       continue
    seed = int(model_loc.split("_")[-3])
    print(f"{model} ({seed}) - {model_loc}")
    if model not in scores.keys():
       scores[model] = {}
    saved = torch.load(os.path.join(test_model_dir, model_loc))
    av = saved['av']
    av.test_size = 25
    av.prop_separate_params = False
    av.want_cuda = True
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    model_state_dict = saved['model_state_dict']
    dataset = av.DATASET_NAME
    if dataset not in scores[model].keys():
       scores[model][dataset] = {}
    print("dataset", dataset)
    t = get_result(av,model_loc,model_state_dict)
    scores[model][dataset][seed] = t[1][1]
    print("val", t[0][1])
    print("test", t[1][1])
    # print(scores[model][dataset][seed])
    pickle.dump(scores, open(f'{ad.save_loc}.pkl', 'wb'))

