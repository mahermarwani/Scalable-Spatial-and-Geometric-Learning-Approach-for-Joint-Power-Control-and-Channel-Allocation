import matplotlib.pyplot as plt
import numpy as np


class Channel(object):
    def __init__(self, htx, hrx, G, N0mdb, carrier_f, shadow_std=8):

        self.htx = htx
        self.hrx = hrx
        self.G = G
        self.N0mdb = N0mdb
        self.N0 = np.power(10, ((self.N0mdb - 30) / 10))
        self.carrier_f = carrier_f
        self.shadow_std = shadow_std

    # Path loss using an empirical model
    def path_loss(self, d):
        N = d.shape[0]
        signal_lambda = 2.998e8 / self.carrier_f
        # compute relevant quantity
        Rbp = 4 * self.hrx * self.htx / signal_lambda
        Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * self.htx * self.htx)))
        # compute coefficient matrix for each Tx/Rx pair
        sum_term = 20 * np.log10(d / Rbp)
        Tx_over_Rx = Lbp + 6 + sum_term + ((d > Rbp).astype(int)) * sum_term  # adjust for longer path loss
        # for a vector of distances
        if d.shape[1] == 1:
            pathloss = -Tx_over_Rx + np.ones(N).reshape(d.shape) * self.G  # only add antenna gain for direct channel
        else:
            # for a matrix of distances
            pathloss = -Tx_over_Rx + np.eye(N) * self.G  # only add antenna gain for direct channel
        pathloss = np.power(10, (pathloss / 10))  # convert from decibel to absolute
        return pathloss

    # Add in shadowing into channel losses
    def add_shadowing(self, channel_losses):
        shadow_coefficients = np.random.normal(loc=0, scale=self.shadow_std, size=np.shape(channel_losses))
        channel_losses = channel_losses * np.power(10.0, shadow_coefficients / 10)
        return channel_losses

    # Add in fast fading into channel losses
    def add_fast_fading(self, channel_losses):

        I = np.random.normal(loc=0, scale=1, size=np.shape(channel_losses))
        R = np.random.normal(loc=0, scale=1, size=np.shape(channel_losses))

        fastfadings = R + I * 1j

        channel_losses = channel_losses * (np.abs(fastfadings) ** 2) / 2
        return channel_losses, fastfadings

    # Calculate capacity of a single user
    def build_fading_capacity_channel(self, h, p):
        return np.log2(1 + h * p / self.N0 * 5e2)


if __name__ == '__main__':
    pass
