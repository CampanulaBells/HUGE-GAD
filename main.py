import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
import argparse
from tqdm import tqdm
import torch
import os
import dgl
import time
import json
import numpy as np
from modules import utils
from modules.utils import prc_auc_score
from modules import model
from modules import ranking
from modules.loss import neighbor_KLD_batch, compute_sim_batch
from modules.node_heterophily import edge_heterophily, node_heterophily, heterophily_baselines
from pathlib import Path
import datetime

# Settings the warnings to be ignored
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
device_ids = 0

# %%
parser = argparse.ArgumentParser(description='HDGAD')
# Logs info
parser.add_argument('--print_results', type=utils.str2bool, default=True)
parser.add_argument('--expr_name', type=str, default="None")  # default: None

# Parameters
parser.add_argument('--dataset', type=str, default='Facebook')
# 'BlogCatalog', 'Amazon', 'Facebook', Reddit', 'YelpChi', 'AmazonFull', 'YelpChiFull'
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size_heterophily', type=int, default=8192)
parser.add_argument('--batch_size_sampling', type=int, default=8192)
parser.add_argument('--heterophily', type=str, default='ours')  # cos, l2, attr, ours
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--kd_param', type=float, default=0.5)
parser.add_argument('--mlp_only', type=bool, default=False)

args = parser.parse_args()
# Set the random seed
utils.set_random_seeds(args.seed)
graph, features, ano_label = utils.load_dataset(args.dataset, normalize=True, to_bidirected=True)
if args.print_results:
    print(args.dataset, "\n  num_nodes =", graph.num_nodes(), "num_edges =", graph.num_edges(),
          "\n  # isolated nodes =", torch.sum(graph.in_degrees() == 0).item(),
          "\n  feature shape:", tuple(features.shape))
graph = graph.to(device_ids)
features = features.to(device_ids)
# obtain heterophily
if args.heterophily == "l2":
    H = heterophily_baselines(graph, features, args.batch_size_heterophily, dist="l2")
elif args.heterophily == "cos":
    H = heterophily_baselines(graph, features, args.batch_size_heterophily, dist="cos")
elif args.heterophily == "attr":
    H = heterophily_baselines(graph, features, args.batch_size_heterophily, dist="attr")
elif args.heterophily == "ours":
    # HALO
    H = edge_heterophily(graph, features, args.batch_size_heterophily, utils.halo)
    H = node_heterophily(graph, H, normalize=True)

# train GAD model
S = ranking.compute_S(H)
graph = graph.remove_self_loop()
graph = graph.add_self_loop()
graph.ndata["features"] = features

# Batch Training
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
dataloader = dgl.dataloading.DataLoader(
    graph, torch.tensor(list(range(graph.num_nodes()))).to(device_ids), sampler,
    batch_size=args.batch_size_sampling,
    shuffle=True,
    drop_last=False,
    num_workers=0)
# Init MLP and GNN parameters
model_GAD = model.HUGE(features.shape[1], args.hidden).cuda()
optimiser = torch.optim.Adam(model_GAD.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model_GAD.train()

epochs = args.epoch
results_loss = []
results_loss_align = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pbar = tqdm(total=epochs, desc=args.dataset)
    train_total_sec = 0
    for epoch in range(epochs):
        time_train_start = datetime.datetime.now()
        batch_loss = 0
        batch_loss_align = 0
        model_GAD.train()
        embs = np.zeros((features.shape[0], args.hidden), dtype=np.float32)
        # Load batch and neighbor subgraph
        for input_nodes, output_nodes, blocks in dataloader:
            optimiser.zero_grad()
            block = blocks[0]
            batch_features = block.srcdata['features']
            # We want to get MLP and GNN embedding for 1 hops neighbors of nodes
            mlp_emb, gnn_emb = model_GAD.forward(blocks[0], batch_features)

            if args.mlp_only:
                score_pred_mlp, score_neg_mlp = compute_sim_batch(mlp_emb, blocks[1], compute_neg=True, ignore_diagnoal=False)
                rank_loss_mlp = ranking.rank_loss(-1 * score_pred_mlp,
                                                  S[output_nodes][:, output_nodes].to(graph.device), 1)
                rank_loss_mlp_reg = ranking.rank_reg_loss(-1 * score_pred_mlp, -1 * score_neg_mlp, 1)
                loss = (rank_loss_mlp + rank_loss_mlp_reg) / 2
                loss_kd = torch.tensor(0.0)
            else:
                score_pred_gnn, score_neg_gnn = compute_sim_batch(gnn_emb, blocks[1], compute_neg=True, ignore_diagnoal=False)
                score_pred_mlp, score_neg_mlp = compute_sim_batch(mlp_emb, blocks[1], compute_neg=True, ignore_diagnoal=False)
                # Compute the rank loss
                rank_loss_gnn = ranking.rank_loss(-1 * score_pred_gnn,
                                                  S[output_nodes][:, output_nodes].to(graph.device), 1)
                rank_loss_mlp = ranking.rank_loss(-1 * score_pred_mlp,
                                                  S[output_nodes][:, output_nodes].to(graph.device), 1)

                rank_loss_gnn_reg = ranking.rank_reg_loss(-1 * score_pred_gnn, -1 * score_neg_gnn, 1)
                rank_loss_mlp_reg = ranking.rank_reg_loss(-1 * score_pred_mlp, -1 * score_neg_mlp, 1)
                rank_loss_gnn = (rank_loss_gnn + rank_loss_gnn_reg) / 2
                rank_loss_mlp = (rank_loss_mlp + rank_loss_mlp_reg) / 2
                # compute the alignment loss
                loss_align = neighbor_KLD_batch(mlp_emb, gnn_emb, blocks[1])
                loss = rank_loss_gnn + rank_loss_mlp + args.kd_param * loss_align
            loss.backward()
            optimiser.step()
            batch_loss += loss.detach().cpu().item() * len(output_nodes)
            batch_loss_align += loss_align.detach().cpu().item() * len(output_nodes)
        batch_loss /= graph.num_nodes()
        batch_loss_align /= graph.num_nodes()
        results_loss.append(batch_loss)
        results_loss_align.append(batch_loss_align)

        time_train_end = datetime.datetime.now()
        timedelta_train = time_train_end - time_train_start
        train_total_sec += timedelta_train.total_seconds()

        pbar.update(1)
    time_eval_start = datetime.datetime.now()
    with torch.no_grad():
        model_GAD.eval()
        scores = torch.zeros(graph.num_nodes(), dtype=torch.float32).to(graph.device)
        for input_nodes, output_nodes, blocks in dataloader:
            block = blocks[0]
            batch_features = block.srcdata['features']
            mlp_emb, gnn_emb = model_GAD.forward(blocks[0], batch_features)
            scores_batch = compute_sim_batch(mlp_emb, blocks[1], compute_neg=False, ignore_diagnoal=False)
            scores[output_nodes] = scores_batch.detach()
            if epoch == epochs - 1:
                embs[output_nodes.detach().cpu().numpy()] = mlp_emb[blocks[1].dstnodes()].detach().cpu().numpy()
        # embs = embs.detach()
        scores = -scores.detach().cpu().numpy()
        auc_roc = roc_auc_score(ano_label, scores)
        auc_prc = prc_auc_score(ano_label, scores)
    time_eval_end = datetime.datetime.now()
    timedelta_eval = time_eval_end - time_eval_start
if args.print_results:
    print()
    print(f"auc_roc:", auc_roc, "auc_prc:", auc_prc, flush=True)
    print("Train time (sec):", train_total_sec)
    print("Test time (sec):", timedelta_eval.total_seconds())
