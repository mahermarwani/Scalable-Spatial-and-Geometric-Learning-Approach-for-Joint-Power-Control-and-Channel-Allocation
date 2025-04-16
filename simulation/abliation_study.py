import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wireless_config.WirelessNetwork import WirelessNetwork
from final_model import model_S
import numpy as np
import torch.nn as nn


# Apply smoothing function to the loss data
def smooth_curve(points, factor=0.5):  # Adjust the factor as needed
    smoothed_points = np.empty_like(points)
    smoothed_points[0] = points[0]
    for i in range(1, len(points)):
        smoothed_points[i] = smoothed_points[i - 1] * factor + points[i] * (1 - factor)
    return smoothed_points


# Plot results
def plot_results(results, metric, ylabel):
    for key in results.keys():
        plt.plot(smooth_curve(results[key][metric]), label=key)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Epoch")
    plt.show()


def initialize_model_weights(model):
    for name, module in model.named_modules():
        if "VGAE" in name or "VSAE" in name:
            continue
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

def start_expeirment(index=0):
    # Network parameters
    net_par = {
        "d0": 1,
        'htx': 1.5,
        'hrx': 1.5,
        'antenna_gain_decibel': 2.5,
        'noise_density_milli_decibel': -169,
        'carrier_f': 2.4e9,
        'shadow_std': 8,
        "rb_bandwidth": 5e2,
        "wc": 50,
        "wd": 5,
        "wx": 200,
        "wy": 100,
        "N": 30,
        "K": 5
    }

    K = net_par["K"]

    latent_dim_node = 100
    latent_dim_edge = 100
    latent_dim_csi = 100
    latent_dim_shared = 300

    min_rate = 800
    num_epochs = 300
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize network and CSI
    network = WirelessNetwork(net_par)
    csi = network.csi.to(device)

    results = {}

    # Define model configurations
    model_configs = [
        ("3LRGAT_3CNN_3SH_wSamp", 3, [16, 24, 32], 3, True, "VGSAE", "./VGAE_100_100_3_30000.pt", "./VSAE_[16, 24, 32].pt"),
        ("3LRGAT_3CNN_1SH_wSamp", 3, [16, 24, 32], 1, True, "VGSAE", "./VGAE_100_100_3_30000.pt", "./VSAE_[16, 24, 32].pt"),
        ("3LRGAT_3CNN_5SH_wSamp", 3, [16, 24, 32], 5, True, "VGSAE", "./VGAE_100_100_3_30000.pt", "./VSAE_[16, 24, 32].pt"),
        
        ("6LRGAT_6CNN_3SH_wSamp", 6, [16, 24, 32, 48, 64, 80], 3, True, "VGSAE", "./VGAE_100_100_6_30000.pt", "./VSAE_[16, 24, 32, 48, 64, 80].pt"),
        ("9LRGAT_9CNN_3SH_wSamp", 9, [16, 24, 32, 48, 64, 80, 96, 112, 128], 3, True, "VGSAE", "./VGAE_100_100_9_30000.pt", "./VSAE_[16, 24, 32, 48, 64, 80, 96, 112, 128].pt"),

        ("VSAE_3CNN_wSamp", 3, [16, 24, 32], 3, True, "VSAE", None, "./VSAE_[16, 24, 32].pt"),
        ("VGAE_3LRGAT_wSamp", 3, [16, 24, 32], 3, False, "VGAE", "./VGAE_100_100_3_30000.pt", None),
        ("3LRGAT_3CNN_3SH_woSamp", 3, [16, 24, 32], 3, False, "VGSAE", "./VGAE_100_100_3_30000.pt", "./VSAE_[16, 24, 32].pt"),
        ("3LRGAT_3CNN_3SH_woPretrain", 3, [16, 24, 32], 3, False, "VGSAE", None, None)
    ]





    # Run model configurations
    for config in model_configs:
        model_name, num_layers, intermediate_channels, num_layers_shared, with_sampling, model_version, pretrained_VGAE_path, pretrained_VSAE_path = config

        model =  model_S(K, 
                    latent_dim_node, 
                    latent_dim_edge, 
                    num_layers, 
                    latent_dim_csi, 
                    intermediate_channels, 
                    latent_dim_shared, 
                    num_layers_shared,
                    model_version,
                    VGAE_path=pretrained_VGAE_path,
                    VSAE_path=pretrained_VSAE_path).to(device)
        
        # Same initialization 
        initialize_model_weights(model)


        _, rate_history, violation_history = model.model_execution( csi, 
                                                                    net_par,
                                                                    device, 
                                                                    with_sampling, 
                                                                    min_rate, 
                                                                    num_epochs,
                                                                    learning_rate,
                                                                    return_history=True
                                                                    )


        results[model_name] = (rate_history, violation_history)






    # plot_results(results, 0, "Rate")
    # plot_results(results, 1, "Violation")

    # save results as numpy
    np.save("results_{}.npy".format(index), results)


if __name__ == '__main__':

    for index in range(50):
        start_expeirment(index)
    exit()
    
    # get the results
    results = np.load("results.npy", allow_pickle=True).item()
    # plot the results


    for key in results.keys():
        plt.plot(smooth_curve(results[key][0]), label=key)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Rate")
    plt.title("Rate vs Epoch")
    plt.show()






    # plot_results(results, 0, "Rate")
    # plot_results(results, 1, "Violation")
