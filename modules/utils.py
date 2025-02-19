import os
import argparse
from tqdm import tqdm
import torch
import time
import os
import torchmetrics
import networkx as nx
import scipy.sparse as sp
import dgl
from dgl.data.utils import load_graphs
import random
import json
import numpy as np
import scipy.io as sio

from sklearn.metrics import auc, precision_recall_curve


cos_dist = lambda *args, **kwargs: -1 * torch.nn.functional.cosine_similarity(*args, **kwargs)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def prc_auc_score(gt, pred):
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    auc_precision_recall = auc(recall, precision)
    return auc_precision_recall

def halo(x_u, x_v, eps=1e-6):
    dist = (x_u - x_v).pow(2).sum(1).sqrt()/((x_u.pow(2).sum(1) + x_v.pow(2).sum(1)+eps).sqrt())
    return dist


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def load_mat(dataset):
    """Load .mat dataset."""

    data = sio.loadmat("./datasets/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label, dtype=np.int64))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    return adj, feat, ano_labels, str_ano_labels, attr_ano_labels

def load_dataset(dataset_name, normalize=True, to_bidirected=True):
    raw_name = dataset_name
    if dataset_name in ['Amazon','Facebook','Reddit','YelpChi']:
        adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(dataset_name)
        features = features.todense()
        nx_graph = nx.from_scipy_sparse_array(adj)
        graph = dgl.from_networkx(nx_graph)
    elif dataset_name in ['AmazonFull', 'YelpChiFull']:
        graph = load_graphs(f"datasets/{dataset_name}")[0][0]
        ano_label = graph.ndata["label"].cpu().detach().numpy()
        features = graph.ndata["feature"]
    else:
        raise Exception(f"Unimplemented dataset: {dataset_name}")

    graph = graph.remove_self_loop()
    if normalize:
        features = torch.tensor(preprocess_features(features))
    else:
        features = torch.tensor(features)
    if to_bidirected:
        graph = dgl.to_bidirected(graph)
    features = features.to(torch.float32)
    return graph.long(), features, ano_label

def set_random_seeds(seed):
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False