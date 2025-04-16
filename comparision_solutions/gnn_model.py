import dgl
import torch
from wireless_config.WirelessNetwork import WirelessNetwork, cal_net_metrics
from tqdm import tqdm
import warnings
# Filter or ignore the DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import random
from datetime import datetime
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# Model definition
def MLP(channels):
    return nn.Sequential(*[
        nn.Sequential(nn.BatchNorm1d(channels[i - 1]),
                      nn.Linear(channels[i - 1], channels[i]),
                      nn.ReLU())
        for i in range(1, len(channels))
    ])


class EdgeConv(nn.Module):
    def __init__(self, mlp1, mlp2):
        super(EdgeConv, self).__init__()
        self.mlp1 = mlp1
        self.mlp2 = mlp2


    def concat_message_function(self, edges):
        b = edges.data['feature']
        c = edges.src['hid']

        cat = torch.cat((b, c), axis=1)
        return {'out': self.mlp1(cat)}

    def apply_func(self, nodes):
        a = nodes.data['reduced_vector']
        b = nodes.data['hid']

        cat = torch.cat((a, b), axis=1)

        return {"hid": self.mlp2(cat)}

    def forward(self, g):
        g.apply_edges(self.concat_message_function)

        g.update_all(fn.copy_e('out', 'msg'),
                     fn.mean('msg', 'reduced_vector'),
                     self.apply_func)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = EdgeConv(MLP([3, 9]), MLP([10, 10]))

        self.conv2 = EdgeConv(MLP([12, 10]), MLP([11 + 9, 10]))
        self.conv3 = EdgeConv(MLP([12, 10]), MLP([11 + 9, 10]))
        self.conv4 = EdgeConv(MLP([12, 10]), MLP([11 + 9, 10]))

    def forward(self, g):
        g.ndata['hid'] = g.ndata['feature']  # initialization of GNN
        self.conv1(g)
        self.conv2(g)
        self.conv3(g)
        self.conv4(g)

        return g.ndata['hid']


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.gcn1 = GCN()
        self.gcn2 = GCN()
        self.gcn3 = GCN()
        self.gcn4 = GCN()
        self.gcn5 = GCN()

        self.embedding_layers = [self.gcn1,
                                 self.gcn2,
                                 self.gcn3,
                                 self.gcn4,
                                 self.gcn5]

        self.cnn = nn.Sequential(
            nn.Conv2d(10, 5, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(5, 2, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(2, 1, kernel_size=(3, 3), stride=1, padding=1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g_list, training=False):

        X = []
        for k in range(len(g_list)):
            X.append(self.embedding_layers[k](g_list[k]))

        A = self.cnn(torch.stack(X, dim=0).permute(2, 1, 0)).squeeze(0)

        if training:
            p_train = torch.sum(self.sigmoid(A), dim=1) / 5
            rb_train = self.softmax(A)
            return p_train, rb_train
        else:
            p_test, rb_test = torch.sum(self.sigmoid(A), dim=1) / 5, torch.eye(5)[
                torch.max(self.softmax(A), dim=1)[1]]
            return p_test, rb_test


# Compute regulated loss using the min_rate and rates of all users
def regulated_loss(min_rate, rates):
    # minimum rate per user calculated by maximum interference

    return torch.mean(F.relu(rates - min_rate))


def binarization_loss(rb_hat):
    return - 1 * torch.mean(torch.sqrt(((rb_hat / 0.5) - 1) ** 2 + 0.5))


def build_graph(csi, net_par):
    adj = []
    for i in range(0, net_par["N"]):
        for j in range(0, net_par["N"]):
            if not (i == j):
                adj.append([i, j])

    # build a graph for each Resource Block
    g_list = []
    log_x = torch.log10(csi)
    mean_log_x = torch.mean(log_x)
    variance_log_x = torch.var(log_x)

    # Calculate the psi values for each element in the tensor using the formula
    csi = (log_x - mean_log_x) / torch.sqrt(variance_log_x)

    for k in range(net_par["K"]):

        # create a graph
        graph = dgl.graph(adj, num_nodes=net_par["N"])

        # Node feature of the i-th node is the direct link channel of i-th pair
        node_features = torch.diag(csi[k, :, :]).unsqueeze(1)

        # Edge feature between node e[0] and e[1] is the interference channel between e[0]-th pair and e[1]-th pair
        edge_features = []
        for e in adj:
            # print(e, csi[k, e[0], e[1]])
            edge_features.append([csi[k, e[0], e[1]], csi[k, e[1], e[0]]])

        graph.ndata['feature'] = node_features
        graph.edata['feature'] = torch.tensor(edge_features, dtype=torch.float)

        g_list.append(graph)

    return g_list


def inference(model, csi, net_par, device, c_min, iterations=300):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    g_list = tuple([g.to(device) for g in build_graph(csi, net_par)])
    csi = csi.to(device)

    rates_res = torch.zeros(csi.shape[1])

    # Create the tqdm progress bar
    progress_bar = tqdm(total=iterations, desc="Iterations")

    for i in range(iterations):
        # Zero gradient for every batch
        optimizer.zero_grad()

        # Model output
        p_hat, rb_hat = model.forward(g_list, training=True)

        if i < 0.3 * iterations:
            alpha = 0.3
        else:
            alpha = 0

        _, rates = cal_net_metrics(csi, rb_hat, p_hat, net_par, device)
        net_rate = rates.mean()

        reg_loss = regulated_loss(c_min, rates)

        loss = - net_rate + alpha * reg_loss 
        loss.backward()

        # weights update
        optimizer.step()

        # Explicitly set the device for intermediate tensors
        indices = torch.max(rb_hat, dim=1)[1].to(device)
        eye_tensor = torch.eye(5).to(device)
        eye_indexed = eye_tensor[indices]

        # Now, perform the operation with tensors on the same device
        _, rates_res = cal_net_metrics(csi, eye_indexed, p_hat, net_par, device)


        # Update the progress bar
        progress_bar.update(1)
        progress_bar.set_postfix( {"Mean Rate": rates_res.mean().item(), "Violations": torch.sum(rates_res < c_min).item()})

    # Close the progress bar when done
    progress_bar.close()

    return p_hat, eye_indexed 


if __name__ == '__main__':
    pass



