import torch


# Output: S_ij = 1 if hetero_score_i > hetero_score_j;
#         S_ij = 0 otherwise
def compute_S(hetero_score):
    # score_expand = [[1, 2, 3],
    #          [1, 2, 3],
    #          [1, 2, 3]]
    score_expand = hetero_score.unsqueeze(0).expand(hetero_score.shape[0], -1)
    S = ((score_expand - score_expand.T) > 0).to(torch.float32).detach()
    return S


def compute_s(hetero_logits):
    logits_expand = hetero_logits.unsqueeze(0).expand(hetero_logits.shape[0], -1)
    s = logits_expand - logits_expand.T
    return s


def rank_loss(pred, GT, param_sigmoid):
    s = compute_s(pred)
    loss = param_sigmoid * (1 - GT) * s + (1 + (-param_sigmoid * s).exp()).log()
    return loss.mean()


def rank_reg_loss(pred, neg, param_sigmoid):
    s = pred - neg
    loss = param_sigmoid * s + (1 + (-param_sigmoid * s).exp()).log()
    return loss.mean()
