import torch
from .utils import cos_dist


def heterophily_baselines(g, features, batch_size, dist="l2"):
    edge_v, edge_u = g.edges()
    H = []
    for i in range(len(edge_u) // batch_size + 1):
        i_beg = i * batch_size
        i_end = min((i + 1) * batch_size, len(edge_u))
        u = edge_u[i_beg:i_end]
        v = edge_v[i_beg:i_end]
        if dist == "l2":
            h = (features[u] - features[v]).pow(2).sum(1).sqrt()
        elif dist == "cos":
            h = -1 * torch.nn.functional.cosine_similarity(features[u], features[v])
        elif dist == "attr":
            h = (features[u] != features[v]).sum(axis=1) / features.shape[1]

        H.append(h)
    edge_heterophily = torch.concatenate(H, axis=0)
    node_heterophily = torch.zeros((g.num_nodes())).to(g.device)
    node_heterophily_mean = -1 * torch.ones((g.num_nodes())).to(g.device)

    edge_u, edge_v = g.edges()
    node_heterophily.index_add_(0, edge_u, edge_heterophily)

    index = torch.nonzero(g.in_degrees())
    node_heterophily_mean[index] = node_heterophily[index] / g.in_degrees()[index]

    return node_heterophily_mean


def edge_heterophily(g, features, batch_size, dist=cos_dist, eps=1e-9):
    edge_v, edge_u = g.edges()
    H = []
    for i in range(len(edge_u) // batch_size + 1):
        i_beg = i * batch_size
        i_end = min((i + 1) * batch_size, len(edge_u))
        u = edge_u[i_beg:i_end]
        v = edge_v[i_beg:i_end]
        coeff = torch.abs(features[u] - features[v]) + eps
        h = dist(coeff * features[u], coeff * features[v])
        H.append(h)
    return torch.concatenate(H, axis=0)


def node_heterophily(g, H, normalize=True):
    node_H = torch.zeros((g.num_nodes())).to(g.device)
    edge_v, edge_u = g.edges()
    # INCORRECT!
    # node_H[edge_u] += H
    node_H.index_add_(0, edge_u, H)
    if normalize:
        index = torch.nonzero(g.in_degrees())
        node_H[index] = node_H[index] / g.in_degrees()[index]
        node_H[torch.nonzero(g.in_degrees() == 0)] = -1
    return node_H


def node_heterophily_normalize(g, H, normalize=True):
    node_H = torch.zeros((g.num_nodes())).to(g.device)
    edge_v, edge_u = g.edges()
    node_H.index_add_(0, edge_u, H)
    if normalize:
        index = torch.nonzero(g.in_degrees())
        node_H[index] = node_H[index] / g.in_degrees()[index]
        node_H[torch.nonzero(g.in_degrees() == 0)] = -2
        factor = node_H[index].sum() / (g.in_degrees() > 0).sum() * (
                    g.in_degrees()[index].to(torch.float32) - g.in_degrees()[index].to(torch.float32).mean())
        factor = factor / torch.std(g.in_degrees()[index].to(torch.float32))
        node_H[index] = node_H[index] - factor
    return node_H