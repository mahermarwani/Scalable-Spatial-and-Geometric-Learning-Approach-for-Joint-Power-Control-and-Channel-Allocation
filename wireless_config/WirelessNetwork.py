import random
import numpy as np
from sklearn.datasets import make_blobs
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix

from .Channel import Channel
from tqdm import tqdm
import warnings
import os


class WirelessNetwork():
    def __init__(self, net_par):
        # Object for modeling a Wireless Network
        # Network parameters
        self.wx = net_par["wx"]  # width of the network land
        self.wy = net_par["wy"]  # length of the network land
        self.wc = net_par['wc']  # maximum distance between users
        self.wd = net_par['wd']  # minimum distance between users
        self.links_numb = net_par['N']  # number of links

        # Determines transmitter and receiver positions
        self.t_pos, self.r_pos = self.determine_positions()

        # Calculate distance matrix using scipy.spatial method
        self.dist_mat = distance_matrix(self.t_pos, self.r_pos)
        # Channel gain parameters
        # self.d0 = net_par['d0']
        # self.gamma = net_par['gamma']
        self.htx = net_par['htx']
        self.hrx = net_par['hrx']
        self.G = net_par['antenna_gain_decibel']
        self.N0mdb = net_par['noise_density_milli_decibel']
        self.N0 = np.power(10, ((self.N0mdb - 30) / 10))
        self.carrier_f = net_par['carrier_f']
        self.shadow_std = net_par['shadow_std']
        # Creates a channel with the given parameters
        self.channel = Channel(self.htx, self.hrx, self.G, self.N0mdb, self.carrier_f, self.shadow_std)

        #  resource blocks Parameters
        self.rb_bandwidth = net_par['rb_bandwidth']
        self.rb_numb = net_par['K']

        #  Generate channel gain tensor [number of blocks, number of Tx, number of Rx]
        self.csi, self.fast_fading = self.generate_csi()

       
    def determine_positions(self):
        # Calculate transmitter positions
        t_x_pos = np.random.uniform(0, self.wx, (self.links_numb, 1))
        t_y_pos = np.random.uniform(0, self.wy, (self.links_numb, 1))
        t_pos = np.hstack((t_x_pos, t_y_pos))
        # Calculate receiver positions
        r_distance = np.random.uniform(self.wd, self.wc, (self.links_numb, 1))
        r_angle = np.random.uniform(0, 2 * np.pi, (self.links_numb, 1))
        r_rel_pos = r_distance * np.hstack((np.cos(r_angle), np.sin(r_angle)))
        r_pos = t_pos + r_rel_pos
        return t_pos, r_pos 

    def plot_network(self):
        # Plot transmitter and receiver positions
        plt.figure()
        plt.scatter(self.t_pos[:, 0], self.t_pos[:, 1], marker='o', color='b')
        plt.scatter(self.r_pos[:, 0], self.r_pos[:, 1], marker='x', color='r')

        plt.xlabel('x')
        plt.ylabel('y')
        # legends
        plt.legend(['Transmitter', 'Receiver'])
        plt.grid(True)
        plt.show()

    def generate_csi(self):
        channel_gain = np.array([self.channel.path_loss(self.dist_mat) for _ in range(self.rb_numb)]) 

        channel_gain = self.channel.add_shadowing(channel_gain)
        channel_losses, fast_fadings = self.channel.add_fast_fading(channel_gain)

        return torch.from_numpy(channel_losses).float(), torch.from_numpy(fast_fadings)


def cal_net_metrics(csi, rb, p, net_par, device="cpu"):
    K = net_par["K"]
    N = net_par["N"]

    N0 = np.power(10, ((net_par["noise_density_milli_decibel"] - 30) / 10))
    bandwidth = net_par["rb_bandwidth"]

    rates = torch.zeros(N, device=device)
    for k in range(K):
        received_power = torch.diag(csi[k, :, :]) * rb[:, k] * p
        interference = torch.matmul(csi[k, :, :], rb[:, k] * p) - received_power

        noise_power = N0 * bandwidth

        snr = received_power / (interference + noise_power)
        snr = torch.clamp(snr, min=0.0)

        rates = torch.add(rates, bandwidth * rb[:, k] * torch.log2(1 + snr))

    EEs = torch.div(rates, p) 

    # SEs = torch.zeros_like(rates, device=device)
    # for i in range(N):
    #     if torch.sum(rb[i, :]) > 0:
    #         SEs[i] = rates[i] / net_par['rb_bandwidth']
    
    # torch.div(rates, net_par['rb_bandwidth'] * torch.sum(rb, dim=1))

    # EEs = torch.sum(rates) / torch.sum(p)
    # SEs = torch.sum(rates) / torch.sum(net_par['rb_bandwidth'] * torch.sum(rb, dim=1))

    return EEs, rates





if __name__ == '__main__':
    net_par = {"d0": 1,
               'htx': 1.5,
               'hrx': 1.5,
               'antenna_gain_decibel': 2.5,
               'noise_density_milli_decibel': -169,
               'carrier_f': 2.4e9,
               'shadow_std': 8,
               "rb_bandwidth": 5e2,
               "wc": 50,
               "wd": 20,
               "wx": 500,
               "wy": 500,
               "N": 10,
               "K": 5}

    N = net_par['N']  # Number of links
    K = net_par['K']  # Number of Rbs

    network = WirelessNetwork(net_par)
    # network.plot_network()
    csi = network.csi

    p = torch.zeros(N)
    rb = torch.zeros((N, K))

    for n in range(N):
        rb[n, random.randint(0, K - 1)] = 1
        p[n] = random.random()

    print("input parameters shapes")
    print("csi: ", csi.shape)
    print("rb: ", rb.shape)
    print("p: ", p.shape)

    EEs, rates = cal_net_metrics(csi, rb, p, net_par)

    print("output parameters shapes")
    print(rates)
    print(EEs)

