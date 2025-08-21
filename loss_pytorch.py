import torch
import torch.nn as nn

def custom_loss(y_true, y_pred):
    '''
    Customized loss function to improve the recoil response,
    by balancing the response above one and below one
    '''
    
    px_truth = y_true[:, 0].flatten()
    py_truth = y_true[:, 1].flatten()
    px_pred = y_pred[:, 0].flatten()
    py_pred = y_pred[:, 1].flatten()

    pt_truth = torch.sqrt(px_truth * px_truth + py_truth * py_truth)

    px_truth1 = px_truth / pt_truth
    py_truth1 = py_truth / pt_truth

    # Using absolute response
    upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred) - pt_truth
    pt_cut = pt_truth > 0.0 / 50.0
    upar_pred = upar_pred[pt_cut]
    pt_truth_filtered = pt_truth[pt_cut]

    filter_bin0 = pt_truth_filtered < 5.0 / 50.0
    filter_bin1 = torch.logical_and(pt_truth_filtered > 5.0 / 50.0, pt_truth_filtered < 10.0 / 50.0)
    filter_bin2 = pt_truth_filtered > 10.0 / 50.0

    upar_pred_pos_bin0 = upar_pred[torch.logical_and(filter_bin0, upar_pred > 0.0)]
    upar_pred_neg_bin0 = upar_pred[torch.logical_and(filter_bin0, upar_pred < 0.0)]
    upar_pred_pos_bin1 = upar_pred[torch.logical_and(filter_bin1, upar_pred > 0.0)]
    upar_pred_neg_bin1 = upar_pred[torch.logical_and(filter_bin1, upar_pred < 0.0)]
    upar_pred_pos_bin2 = upar_pred[torch.logical_and(filter_bin2, upar_pred > 0.0)]
    upar_pred_neg_bin2 = upar_pred[torch.logical_and(filter_bin2, upar_pred < 0.0)]
    
    norm = torch.sum(pt_truth_filtered)
    dev = torch.abs(torch.sum(upar_pred_pos_bin0) + torch.sum(upar_pred_neg_bin0))
    dev += torch.abs(torch.sum(upar_pred_pos_bin1) + torch.sum(upar_pred_neg_bin1))
    dev += torch.abs(torch.sum(upar_pred_pos_bin2) + torch.sum(upar_pred_neg_bin2))
    dev /= norm

    # Base MSE loss
    loss = 0.5 * torch.mean((px_pred - px_truth) ** 2 + (py_pred - py_truth) ** 2)

    # Add response balancing term
    loss += 10.0 * dev
    
    return loss
