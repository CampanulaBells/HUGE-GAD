import torch
import torch.nn as nn
import dgl


class LinearEncoder(nn.Module):
    def __init__(self, n_in, n_h, n_out, n_layers):
        super(LinearEncoder, self).__init__()
        self.mlp = nn.Sequential()
        assert n_layers > 0
        if n_layers == 1:
            self.mlp.add_module("dense1", nn.Linear(n_in, n_out, bias=True))
            self.mlp.add_module("act1", nn.PReLU())
        elif n_layers == 2:
            self.mlp.add_module("dense1", nn.Linear(n_in, n_h, bias=True))
            self.mlp.add_module("act1", nn.PReLU())
            self.mlp.add_module("dense2", nn.Linear(n_h, n_out, bias=True))
            self.mlp.add_module("act2", nn.PReLU())
        else:
            self.mlp.add_module("dense1", nn.Linear(n_in, n_h, bias=True))
            self.mlp.add_module("act1", nn.PReLU())
            for i in range(n_layers - 2):
                self.mlp.add_module(f"dense{i + 2}", nn.Linear(n_h, n_h, bias=True))
                self.mlp.add_module(f"act{i + 2}", nn.PReLU())
            self.mlp.add_module(f"dense{n_layers}", nn.Linear(n_h, n_out, bias=True))
            self.mlp.add_module(f"act{n_layers}", nn.PReLU())

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat):
        emb = self.mlp(feat)
        return emb


class HUGE(nn.Module):
    def __init__(self, n_in, n_h):
        super(HUGE, self).__init__()
        self.mlp = LinearEncoder(n_in, n_h, n_h, 2)
        self.gcn = dgl.nn.GraphConv(n_h, n_h)
        self.gnn_act = nn.PReLU()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, g, feat):
        emb_mlp = self.mlp(feat)
        emb_gnn = self.gnn_act(self.gcn(g, emb_mlp))
        return emb_mlp, emb_gnn
