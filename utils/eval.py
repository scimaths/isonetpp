import time
import torch
from sklearn.metrics import average_precision_score

def pairwise_ranking_loss(pred_pos, pred_neg, margin):
    num_pos, dim = pred_pos.shape
    num_neg, _ = pred_neg.shape

    expanded_pred_pos = pred_pos.unsqueeze(1)
    expanded_pred_neg = pred_neg.unsqueeze(0)
    relu = torch.nn.ReLU()
    loss = relu(margin + expanded_pred_neg - expanded_pred_pos)
    mean_loss = torch.mean(loss, dim=(0, 1))

    return mean_loss

def compute_average_precision(model, pos_pairs, neg_pairs, dataset, return_pred_and_labels=False, return_running_time=False):
    assert not(return_running_time and return_pred_and_labels)
    all_pairs = pos_pairs + neg_pairs
    num_pos_pairs, num_neg_pairs = len(pos_pairs), len(neg_pairs)

    if return_running_time:
        total_running_time = 0
        total_batches = 0
    predictions = []
    num_batches = dataset.create_custom_batches(all_pairs)
    for batch_idx in range(num_batches):
        batch_graphs, batch_graph_sizes, _, batch_adj_matrices = dataset.fetch_batch_by_id(batch_idx)

        if return_running_time:
            start_time = time.time()

        model_output = model(batch_graphs, batch_graph_sizes, batch_adj_matrices)

        if return_running_time:
            end_time = time.time()
            total_running_time += end_time - start_time
            total_batches += 1

        predictions.append(model_output.data)
    all_predictions = torch.cat(predictions, dim=0)
    all_labels = torch.cat([torch.ones(num_pos_pairs), torch.zeros(num_neg_pairs)])

    average_precision = average_precision_score(all_labels, all_predictions.cpu())
    if return_pred_and_labels:
        return average_precision, all_labels, all_predictions
    elif return_running_time:
        return average_precision, total_running_time, total_batches
    else:
        return average_precision