import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wireless_config.WirelessNetwork import WirelessNetwork
from final_model import model_S

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

def initialize_network_and_model(net_par, device, model_version, latent_dims, intermediate_channels, paths):
    model = model_S(
        net_par["K"], 
        latent_dims["node"], 
        latent_dims["edge"], 
        latent_dims["num_layers"], 
        latent_dims["csi"], 
        intermediate_channels, 
        latent_dims["shared"], 
        3,
        model_version,
        VGAE_path=paths["VGAE"],
        VSAE_path=paths["VSAE"]
    ).to(device)
    
    network = WirelessNetwork(net_par)
    csi = network.csi.to(device)
    
    return model, csi

def execute_model_with_permutations(net_par, device, model_version, latent_dims, intermediate_channels, paths, csi, with_sampling, min_rate, num_epochs, learning_rate, num_permutations=10):
    p_histories = []
    rb_histories = []
    permutations = []

    for _ in range(num_permutations):
        model, _ = initialize_network_and_model(net_par, device, model_version, latent_dims, intermediate_channels, paths)
        
        p_l_1 = random.sample(range(csi.shape[0]), csi.shape[0])
        p_l_2 = random.sample(range(csi.shape[1]), csi.shape[1])

        permutations.append(p_l_2)

        permuted_csi = csi[p_l_1, :, :][:, p_l_2, :][:, :, p_l_2]

        p_history, rb_history = model.model_execution(
            permuted_csi, 
            net_par,
            device, 
            with_sampling, 
            min_rate, 
            num_epochs,
            learning_rate,
            return_history=False,
            return_history_sol=True
        )

        p_histories.append(p_history)
        rb_histories.append(rb_history)
    
    return p_histories, rb_histories, permutations

def apply_permutations(tensors, permutations):
    permuted_tensors = []
    for tensor, p_l_2 in zip(tensors, permutations):
        permuted_tensor = tensor[p_l_2]
        permuted_tensors.append(permuted_tensor)
    return permuted_tensors

def compute_similarity_matrices(histories, num_permutations, permutations, num_epochs):
    avg_similarity_matrix = np.zeros((num_permutations, num_permutations))
    


    """
    for epoch in range(num_epochs):
        vectors = [histories[i][epoch] for i in range(num_permutations)]
        vectors = apply_permutations(vectors, permutations)
        vectors_np = [tensor.detach().cpu().numpy() for tensor in vectors]
        vectors_np = np.stack(vectors_np)
        
        similarity_matrix = cosine_similarity(vectors_np)
        avg_similarity_matrix += similarity_matrix
    
    avg_similarity_matrix /= num_epochs
    """


    epoch = -1
    vectors = [histories[i][epoch] for i in range(num_permutations)]
    vectors = apply_permutations(vectors, permutations)
    vectors_np = [tensor.detach().cpu().numpy() for tensor in vectors]
    vectors_np = np.stack(vectors_np)
    
    similarity_matrix = cosine_similarity(vectors_np)
    avg_similarity_matrix += similarity_matrix

    return avg_similarity_matrix




def experiment():
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
    
    latent_dims = {
        "node": 100,
        "edge": 100,
        "csi": 100,
        "shared": 300,
        "num_layers": 3
    }
    
    intermediate_channels = [16, 24, 32]
    paths = {
        "VGAE": "./VGAE_100_100_3_30000.pt",
        "VSAE": "./VSAE_[16, 24, 32].pt"
    }
    
    model_version = "VGSAE"
    with_sampling = True
    min_rate = 300
    num_epochs = 1000
    learning_rate = 0.001
    num_permutations = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, csi = initialize_network_and_model(net_par, device, model_version, latent_dims, intermediate_channels, paths)
    
    p_histories, rb_histories, permutations = execute_model_with_permutations(
        net_par, device, model_version, latent_dims, intermediate_channels, paths, csi, 
        with_sampling, min_rate, num_epochs, learning_rate, num_permutations
    )

    p_avg_similarity_matrix = compute_similarity_matrices(p_histories, num_permutations, permutations, num_epochs)
    rb_avg_similarity_matrix = compute_similarity_matrices(rb_histories, num_permutations, permutations, num_epochs)
    
    plt.figure()
    plt.imshow(p_avg_similarity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Average Cosine Similarity Matrix for P Histories')
    plt.show()
    
    plt.figure()
    plt.imshow(rb_avg_similarity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Average Cosine Similarity Matrix for RB Histories')
    plt.show()


    # save the results
    np.save("p_avg_similarity_matrix.npy", p_avg_similarity_matrix)
    np.save("rb_avg_similarity_matrix.npy", rb_avg_similarity_matrix)






if __name__ == '__main__':
    pass
    # Run the experiment
    # experiment()

    # exit()
