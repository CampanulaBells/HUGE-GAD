import torch


def compute_sim(feature, graph, batch_size, compute_neg=False, batch_size_neg=-1, ignore_diagnoal=False):
    edge_v, edge_u = graph.edges()
    ignore_diagnoal = float(ignore_diagnoal)
    similarity = torch.zeros((graph.num_nodes())).to(graph.device)

    for i in range(len(edge_u) // batch_size + 1):
        i_beg = i * batch_size
        i_end = min((i + 1) * batch_size, len(edge_u))
        u = edge_u[i_beg:i_end]
        v = edge_v[i_beg:i_end]
        h = torch.nn.functional.cosine_similarity(feature[u], feature[v])
        similarity.index_add_(0, u, h)

    similarity_pos = (similarity - ignore_diagnoal) / (graph.in_degrees() - ignore_diagnoal)
    if not compute_neg:
        return similarity_pos
    else:
        feature_norm = torch.nn.functional.normalize(feature)
        sim_negs = []
        for i in range(graph.num_nodes() // batch_size_neg + 1):
            i_beg = i * batch_size_neg
            i_end = min((i + 1) * batch_size_neg, len(edge_u))
            pairwise_similarity = torch.mm(feature_norm[i_beg: i_end], feature_norm.T)
            sim_negs.append(pairwise_similarity.sum(axis=1))
        similarity_neg = torch.concatenate(sim_negs, axis=0)
        similarity_neg = similarity_neg - similarity
        similarity_neg = similarity_neg / (graph.num_nodes() - graph.in_degrees())
        return similarity_pos, similarity_neg


def neighbor_KLD(input, target, graph, batch_size):
    target = target.detach()
    edge_v, edge_u = graph.edges()
    kld_sum = 0
    for i in range(len(edge_u) // batch_size + 1):
        i_beg = i * batch_size
        i_end = min((i + 1) * batch_size, len(edge_u))
        u = edge_u[i_beg:i_end]
        v = edge_v[i_beg:i_end]

        input_sim = (1 + torch.nn.functional.cosine_similarity(input[u], input[v])) / 2
        input_sim_log = input_sim.log()
        target_sim = (1 + torch.nn.functional.cosine_similarity(target[u], target[v])) / 2
        target_sim_log = target_sim.log()

        kld_edgesim = target_sim * (target_sim_log - input_sim_log)
        kld_edgesim_normalize = kld_edgesim / (graph.in_degrees()[u] - 1)
        kld_sum += kld_edgesim_normalize.sum()
    kld = kld_sum / graph.num_nodes()
    return kld

def neighbor_KLD_batch(input, target, block):
    target = target.detach()
    src, dst = block.edges()
    input_sim = (1 + torch.nn.functional.cosine_similarity(input[src], input[dst])) / 2

    input_sim_log = input_sim.log()
    target_sim = (1 + torch.nn.functional.cosine_similarity(target[src], target[dst])) / 2
    target_sim_log = target_sim.log()

    kld_edgesim = target_sim * (target_sim_log - input_sim_log)
    idx = (src != dst).nonzero()
    kld_edgesim_normalize = kld_edgesim[idx] / (block.in_degrees()[dst[idx]] - 1)
    kld_sum = kld_edgesim_normalize.sum()
    kld = kld_sum / block.num_dst_nodes()
    return kld

def compute_sim_batch(feature, block, compute_neg=False, ignore_diagnoal=False):
    src, dst = block.edges()
    similarity = torch.zeros((block.num_dst_nodes())).to(block.device)
    h = torch.nn.functional.cosine_similarity(feature[dst], feature[src])
    similarity.index_add_(0, dst, h)
    if ignore_diagnoal:
        similarity_pos = (similarity - 1) / (block.in_degrees() - 1)
    else:
        similarity_pos = similarity / block.in_degrees()
    if not compute_neg:
        return similarity_pos
    else:
        feature_norm = torch.nn.functional.normalize(feature)
        similarity_neg = torch.mm(feature_norm[block.dstnodes()], feature_norm.T).sum(axis=1)
        similarity_neg = similarity_neg - similarity
        similarity_neg = similarity_neg / (feature.shape[0] - block.in_degrees())
        return similarity_pos, similarity_neg
