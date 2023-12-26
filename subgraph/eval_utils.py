import time
import scipy
import torch
import numpy as np
from subgraph.utils import cudavar
from sklearn.metrics import average_precision_score

def evaluate_embeddings_similarity(av,model,sampler=True):
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
        pred.append( model(batch_data,batch_data_sizes,batch_adj)[2].data)

  all_pred = torch.cat(pred,dim=0) 
  labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
  ap_score = average_precision_score(labels.cpu(), all_pred.cpu())  

  q_graphs = list(range(len(sampler.query_graphs)))    
    
  all_ap = []

  for q_id in q_graphs:
    dpos = list(filter(lambda x:x[0][0]==q_id,d_pos))
    dneg = list(filter(lambda x:x[0][0]==q_id,d_neg))
    npos = len(dpos)
    nneg = len(dneg)
    d = dpos+dneg
    if npos>0 and nneg>0:    
      n_batches = sampler.create_batches(d) 
      pred = []  
      for i in range(n_batches):
        batch_data,batch_data_sizes,_,batch_adj = sampler.fetch_batched_data_by_id(i)
        if av.output_type == 1:
            pred.append( model(batch_data,batch_data_sizes,batch_adj).data)
        elif av.output_type == 2:
            pred.append( model(batch_data,batch_data_sizes,batch_adj)[2].data)
      all_pred = torch.cat(pred,dim=0) 
      labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
      ap   = average_precision_score(labels.cpu(), all_pred.cpu()) 
      all_ap.append(ap)  
  return ap_score, np.mean(all_ap)

def eval_edge_alignment(av,bestM, sampler):
    """
      bestM: usually best validation model is to be passed here for diagnostics
      sampler: evaluation data to be passed here. 
    """
    s = time.time()
    bestM.diagnostic_mode = True
    all_ap1 = []
    all_ap3 = []
    all_pos_hs1 = []
    all_neg_hs1 = []
    all_pos_hs3 = []
    all_neg_hs3 = []
    maskDiag = torch.ones(av.MAX_EDGES,av.MAX_EDGES).fill_diagonal_(0)
    q_graphs = list(range(len(sampler.query_graphs)))    
    for q_id in q_graphs:
        dpos = list(filter(lambda x:x[0][0]==q_id,sampler.list_pos))
        dneg = list(filter(lambda x:x[0][0]==q_id,sampler.list_neg))
        npos = len(dpos)
        nneg = len(dneg)
        d = dpos+dneg
        #if npos>0 and nneg>0:
        n_batches = sampler.create_batches(d)
        plans = []  
        qesim = []
        cesim = []
        qnormvals = [] 
        cnormvals = []
        for i in range(n_batches):
            batch_data,batch_data_sizes,_,batch_adj = sampler.fetch_batched_data_by_id(i)
            plans.append( bestM(batch_data,batch_data_sizes,batch_adj))
            node_features, edge_features, from_idx, to_idx, graph_idx = bestM.get_graph(batch_data)
            edge_counts  = bestM.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
            from_idx_split = torch.split(from_idx, edge_counts, dim=0)
            to_idx_split = torch.split(to_idx, edge_counts, dim=0)
            batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
            prefix_sum_node_counts = [sum(batch_data_sizes_flat[:k])for k in range(len(batch_data_sizes_flat))]
            to_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(to_idx_split, prefix_sum_node_counts)]
            from_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(from_idx_split, prefix_sum_node_counts)]
            incidence_mat = [torch.zeros(av.MAX_EDGES,x) for x in batch_data_sizes_flat]
            from_idx_enum = [(torch.arange(a),b) for (a,b) in zip(edge_counts,from_idx_split_relabeled)]
            to_idx_enum = [(torch.arange(a),b) for (a,b) in zip(edge_counts,to_idx_split_relabeled)]
            for a,b in zip(incidence_mat,from_idx_enum):
                a[b] = 1
            for a,b in zip(incidence_mat,to_idx_enum):
                a[b] = 1
            edge_sim_mat = [torch.mul(maskDiag,(x@x.T)) for x in incidence_mat]
            qesim.append(cudavar(av,torch.stack(edge_sim_mat[0::2])))
            cesim.append(cudavar(av,torch.stack(edge_sim_mat[1::2]))) 
            normvals = [torch.sum(x) for x in incidence_mat]
            qnormvals.append(cudavar(av,torch.stack(normvals[0::2])))
            cnormvals.append(cudavar(av,torch.stack(normvals[1::2])))

        all_plans = torch.cat(plans,dim=0) 
        all_qesim  = torch.cat(qesim)
        all_cesim  = torch.cat(cesim) 
        all_qnormvals  =torch.cat(qnormvals)
        all_cnormvals  =torch.cat(cnormvals)
        all_hard_plans = []
        for p in all_plans:
            soft_plan = p.detach().cpu()
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-soft_plan)
            all_hard_plans.append(torch.eye(av.MAX_EDGES)[col_ind])
        all_hard_plans = cudavar(av,torch.stack(all_hard_plans))


        hingeScore1 = -torch.sum(torch.nn.ReLU()(all_qesim - all_hard_plans@all_cesim@all_hard_plans.permute(0,2,1)),\
                      dim=(1,2))
        hingeScore3 = torch.div(hingeScore1,all_qnormvals)/2
        labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
        all_ap1.append(average_precision_score(labels.cpu(), hingeScore1.cpu()))
        all_ap3.append(average_precision_score(labels.cpu(), hingeScore3.cpu()))
        all_pos_hs1= all_pos_hs1 + hingeScore1[labels>0.5].tolist()
        all_neg_hs1= all_neg_hs1 + hingeScore1[labels<0.5].tolist()
        all_pos_hs3= all_pos_hs3 + hingeScore3[labels>0.5].tolist()
        all_neg_hs3= all_neg_hs3 + hingeScore3[labels<0.5].tolist()
    
    # Reverting back to original state
    bestM.diagnostic_mode = False
    return all_ap1, all_ap3 , all_pos_hs1, all_neg_hs1,all_pos_hs3, all_neg_hs3

def eval_node_alignment(av,bestM, sampler):
    """
      bestM: usually best validation model is to be passed here for diagnostics
      sampler: evaluation data to be passed here. 
    """
    bestM.diagnostic_mode = True
    all_ap1 = []
    all_ap2 = []
    all_ap3 = []
    all_pos_hs1 = []
    all_neg_hs1 = []
    all_pos_hs2 = []
    all_neg_hs2 = []
    all_pos_hs3 = []
    all_neg_hs3 = []
    q_graphs = list(range(len(sampler.query_graphs)))    
    for q_id in q_graphs:
        dpos = list(filter(lambda x:x[0][0]==q_id,sampler.list_pos))
        dneg = list(filter(lambda x:x[0][0]==q_id,sampler.list_neg))
        npos = len(dpos)
        nneg = len(dneg)
        d = dpos+dneg
        if npos>0 and nneg>0:
          n_batches = sampler.create_batches(d)
          plans = []  
          aq = []
          ac = []
          qsz = [] 
          csz = []
          for i in range(n_batches):
            batch_data,batch_data_sizes,_,batch_adj = sampler.fetch_batched_data_by_id(i)
            plans.append( bestM(batch_data,batch_data_sizes,batch_adj))
            a, b = zip(*batch_adj)
            aq.append(torch.stack(a))
            ac.append(torch.stack(b))
            a,b = zip(*batch_data_sizes)
            qsz.append(cudavar(av,torch.tensor(a)))
            csz.append(cudavar(av,torch.tensor(b)))
          all_plans = torch.cat(plans,dim=0) 
          all_aq  = torch.cat(aq)
          all_ac  = torch.cat(ac)  
          all_qsz = torch.cat(qsz)
          all_csz = torch.cat(csz)
          all_hard_plans = []
          for p in all_plans:
            soft_plan = p.detach().cpu()
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-soft_plan)
            all_hard_plans.append(torch.eye(15)[col_ind])
          all_hard_plans = cudavar(av,torch.stack(all_hard_plans))
          all_qedge_sz = torch.sum(all_aq,dim=(1,2))/2
          all_cedge_sz = torch.sum(all_ac,dim=(1,2))/2

        hingeScore1 = -torch.sum(torch.nn.ReLU()(all_aq - all_hard_plans@all_ac@all_hard_plans.permute(0,2,1)),\
                      dim=(1,2))
        hingeScore2 = torch.div(hingeScore1,torch.mul(all_qsz,all_csz))
        hingeScore3 = torch.div(hingeScore1,all_qedge_sz)/2
        labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
        all_ap1.append(average_precision_score(labels.cpu(), hingeScore1.cpu()))
        all_ap2.append(average_precision_score(labels.cpu(), hingeScore2.cpu()))
        all_ap3.append(average_precision_score(labels.cpu(), hingeScore3.cpu()))
        all_pos_hs1= all_pos_hs1 + hingeScore1[labels>0.5].tolist()
        all_neg_hs1= all_neg_hs1 + hingeScore1[labels<0.5].tolist()
        all_pos_hs2= all_pos_hs2 + hingeScore2[labels>0.5].tolist()
        all_neg_hs2= all_neg_hs2 + hingeScore2[labels<0.5].tolist()
        all_pos_hs3= all_pos_hs3 + hingeScore3[labels>0.5].tolist()
        all_neg_hs3= all_neg_hs3 + hingeScore3[labels<0.5].tolist()

    # Reverting back to original state     
    bestM.diagnostic_mode = False
    return all_ap1, all_ap2 , all_ap3 , all_pos_hs1, all_neg_hs1,\
           all_pos_hs2 , all_neg_hs2, all_pos_hs3, all_neg_hs3

def pairwise_ranking_loss_similarity(predPos, predNeg, margin):
    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    ell = margin + expanded_2 - expanded_1
    hinge = torch.nn.ReLU()
    loss = hinge(ell)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss/(n_1*n_2)
