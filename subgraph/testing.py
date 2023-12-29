import os
import torch
import pickle
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
        pred.append( model(batch_data,batch_data_sizes,batch_adj).data)
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

def evaluate_improvement(av,model,sampler,lambd=1):
  import networkx as nx
  import networkx.algorithms.isomorphism as iso
  
  model.eval()

  d_pos = sampler.list_pos
  d_neg = sampler.list_neg
  q_graphs = list(range(len(sampler.query_graphs)))   

  norms_tot = []

  for q_id in q_graphs:
    dpos = list(filter(lambda x:x[0][0]==q_id,d_pos))
    dneg = list(filter(lambda x:x[0][0]==q_id,d_neg))
    npos = len(dpos)
    nneg = len(dneg)
    d = dpos+dneg

    if npos>0 and nneg>0:

      n_batches = sampler.create_batches(d) 
      norms = []

      for i in range(n_batches): 

        batch_data,batch_data_sizes,_,batch_adj = sampler.fetch_batched_data_by_id(i)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]
        bef = 0
        af = batch_data_sizes_flat[0]
        
        transport_plans = model(batch_data,batch_data_sizes,batch_adj).data

        for j in range(0, transport_plans.shape[0]):

          Query = nx.Graph()

          index = np.logical_and(batch_data.from_idx >= bef, batch_data.from_idx < af)
          query_from = batch_data.from_idx[index]

          index = np.logical_and(batch_data.to_idx >= bef, batch_data.to_idx < af)
          query_to = batch_data.to_idx[index]

          query_edges = [(query_from[k] - bef, query_to[k] - bef) for k in range(len(query_from))]
          Query.add_edges_from(query_edges)
          

          bef += batch_data_sizes_flat[j*2]
          af += batch_data_sizes_flat[j*2+1]

          
          Corpus = nx.Graph()

          corpus_from = batch_data.from_idx[np.logical_and(batch_data.from_idx >= bef, batch_data.from_idx < af)]
          corpus_to = batch_data.to_idx[np.logical_and(batch_data.to_idx >= bef, batch_data.to_idx < af)]
          
          corpus_edges = [(corpus_from[k] - bef, corpus_to[k] - bef) for k in range(len(corpus_from))]
          Corpus.add_edges_from(corpus_edges)
          
          
          bef += batch_data_sizes_flat[j*2 + 1]
          if j*2 + 2 < len(batch_data_sizes_flat):
            af += batch_data_sizes_flat[j*2+2]


          norm = 0
          GM = iso.GraphMatcher(Corpus,Query)

          norms_per = []
          for k in range(0, transport_plans.shape[1]):
            for mapping in GM.subgraph_isomorphisms_iter():
              p_hat = torch.zeros(batch_data_sizes_flat[j*2], batch_data_sizes_flat[j*2+1]).to(transport_plans.device)
              for key in mapping.keys():
                p_hat[mapping[key]][key] = 1
              norm = max(norm, torch.sum(transport_plans[j][k][:batch_data_sizes_flat[j*2], :batch_data_sizes_flat[j*2+1]] * p_hat))
            norms_per.append(norm)
          norms.append(norms_per)
      norms_tot.extend(norms[:npos])
  values_np = np.array([[y.cpu() for y in x] for x in norms_tot])

  color_aray = ['skyblue', 'salmon', 'peach']
  for time in range(values_np.shape[1]):
    pickle.dump(values_np[:,time], open(f'lambda_{lambd}_time_{time}_edge_delete', 'wb'))
    sns.histplot(values_np[:,time], bins=40, kde=True, label=f'time = {time}', palette='pastel')

  # Adding labels and title
  plt.xlabel('NORM(P * P_hat)')
  plt.ylabel('Frequency')
  plt.title(f'Histogram : PTC_FM')
  
  # Show legend
  plt.legend()
  plt.grid(True)
  plt.savefig(f'ptc_fm.png')
  
  return norms_tot

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
    elif model_loc.startswith("isonet"):
      config = load_config(av)
      model = im.ISONET(av,config,1).to(device)
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
    # val_result = evaluate_embeddings_similarity_map_mrr_mndcg(av,model,val_data)
    # test_result = evaluate_embeddings_similarity_map_mrr_mndcg(av,model,test_data)
    test_result = evaluate_improvement(av,model,test_data)

    return val_result, test_result

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", type=str)
# ap.add_argument("--seeds", type=int, nargs='+')
# ap.add_argument("--datasets", type=str, nargs='+')
# ap.add_argument("--tasks", type=str, nargs='+')
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
              )


task_dict = {} 

task_dict['node_early_interaction'] = "Early Interaction"
# task_dict['node_align_node_loss'] = "Node Align Node Loss"
# task_dict['isonet'] = "ISONET"
datasets = ["ptc_mr"]
# datasets = ["aids", "mutag", "ptc_fr", "ptc_fm", "ptc_mr", "ptc_mm"]
test_model_dir = ad.model_dir

scores = {}
for model_loc in os.listdir(test_model_dir):
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
    scores[model][dataset][seed] = get_result(av,model_loc,model_state_dict)[1][1]
    print(scores[model][dataset][seed])
    pickle.dump(scores, open('scores.pkl', 'wb'))

print(scores)