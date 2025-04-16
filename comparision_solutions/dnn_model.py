
import torch
import torch.nn as nn
from wireless_config.WirelessNetwork import WirelessNetwork, cal_net_metrics
from Geometric_embedding import build_graph
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import MLP

class PCDataset(torch.utils.data.Dataset):

    def process(self):
        pass

    def __init__(self, data_root, net_par, data_type=None):

        # Data type
        self.data_type = data_type

        # Network parameters
        self.net_par = net_par

        # extract paths
        self.csi_path = os.path.join(data_root, "csi")
        self.p_path = os.path.join(data_root, "p")
        self.rb_path = os.path.join(data_root, "rb")
        self.samples_list_path = os.path.join(data_root, "samples_list.csv")

        # get samples_list
        self.samples_list = np.genfromtxt(self.samples_list_path, delimiter=',')

        super().__init__()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.samples_list) - 1

    def __getitem__(self, index):
        # index = 0
        """Generates one sample of data"""
        # CSI
        csi = torch.from_numpy(np.load(os.path.join(self.csi_path, "csi_" + str(index) + ".npy"))).float()
        # Power and Bandwidth allocation
        p = torch.from_numpy(np.load(os.path.join(self.p_path, "p_" + str(index) + ".npy"))).float()
        rb = torch.from_numpy(np.load(os.path.join(self.rb_path, "rb_" + str(index) + ".npy"))).type(torch.int64)

        return csi, rb, p



class DNN_model(nn.Module):
    def __init__(self, net_par):
        super(DNN_model, self).__init__()

        self.shared = MLP([net_par["N"] * net_par["K"], 1000, 1000, 32])

        self.p = MLP([32, 4, 4, 1])
        self.rb = MLP([32, 4, 4, net_par["K"]])

    def forward(self, csi):
        csi_flat = csi.view(csi.shape[1], -1)

        x = self.shared(csi_flat)

        p = torch.sigmoid(self.p(x))
        rb = F.softmax(self.rb(x), dim=1)
        return p, rb




if __name__ == '__main__':
   pass