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



class CNN_model(nn.Module):
    def __init__(self, net_par):
        super(CNN_model, self).__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(net_par["K"], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.fcl = MLP([net_par["N"] * 256, 100, 50])
        self.p = MLP([50, 1])
        self.rb = MLP([50, net_par["K"]])


    def forward(self, csi):

        x = self.shared(csi)
        x = x.view(x.shape[-1], -1)

        x = self.fcl(x)

        p = torch.sigmoid(self.p(x))
        rb = F.softmax(self.rb(x), dim=1)
        return p, rb



if __name__ == '__main__':
    pass