import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from wireless_config.WirelessNetwork import WirelessNetwork
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader

class VSAE(nn.Module):
    def __init__(self, K, intermediate_channels=[16, 24, 32]):
        super(VSAE, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(K, intermediate_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels[0], intermediate_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels[1], intermediate_channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.mean_conv = nn.Conv2d(intermediate_channels[2], 1, kernel_size=3, stride=1, padding=1)
        self.log_var_conv = nn.Conv2d(intermediate_channels[2], 1, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(1, intermediate_channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(intermediate_channels[2], intermediate_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(intermediate_channels[1], intermediate_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(intermediate_channels[0], K, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )


    def encode(self, csi):

        x = self.encoder_conv(csi)

        z_mean = self.mean_conv(x)
        z_log_var = self.log_var_conv(x)

        z = z_mean + torch.randn_like(z_log_var) * torch.exp(0.5 * z_log_var)

        return z, z_mean.squeeze(0), z_log_var.squeeze(0)


    def decode(self, z):
        rec = self.decoder_conv(z)
        return rec

    def kl_loss(self, z_mean, z_log_var):
        kl_divergence = -0.5 * torch.mean(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var))
        return kl_divergence
    
    def rec_loss(self, rec, csi):
        reconstruction_loss = F.mse_loss(rec, csi, reduction='mean')
        return reconstruction_loss


class WirelessNetworkDataset(Dataset):
    def __init__(self, net_par, size):
        self.csi_data = []
        for _ in tqdm(range(size), desc="Generating CSI data"):
            # net_par["N"] = random.randint(10, 50)
            # net_par["wc"] = random.randint(50, 80)
            # net_par["shadow_std"] = random.randint(4, 12)
            network = WirelessNetwork(net_par)  # this function generates the CSI data
            self.csi_data.append(network.csi.unsqueeze(0))

        # Convert to a single tensor and min max normalize
        self.csi_data = torch.cat(self.csi_data, dim=0)
        # self.csi_data =  (self.csi_data - self.csi_data.min()) / (self.csi_data.max() - self.csi_data.min())

    def __len__(self):
        return len(self.csi_data)

    def __getitem__(self, idx):
        return self.csi_data[idx]
    


# Apply smoothing function to the loss data
def smooth_curve(points, factor=0.8):  # Adjust the factor as needed
    smoothed_points = np.empty_like(points)
    smoothed_points[0] = points[0]
    for i in range(1, len(points)):
        smoothed_points[i] = smoothed_points[i - 1] * factor + points[i] * (1 - factor)
    return smoothed_points




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
               "wd": 5,
               "wx": 200,
               "wy": 100,
               "N": 30,
               "K": 5}
    
    intermediate_channels = [16, 24, 32]


    # Create an instance of the VAE
    model = VSAE(net_par['K'], intermediate_channels)
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize lists to store loss values for each epoch
    reconstruction_losses = []
    kl_losses = []

    # Training loop
    num_epochs = 100



    # Define the batch size
    batch_size = 4048
    # Create the dataset and dataloader
    dataset_size = 10000
    print("creating dataset of size: ", dataset_size)
    wireless_dataset = WirelessNetworkDataset(net_par, dataset_size)
    dataloader = DataLoader(wireless_dataset, batch_size=batch_size, shuffle=True)

    
    # Create a progress bar object with tqdm
    pbar = tqdm(range(num_epochs), desc='Training Progress')
    for epoch in pbar:
        
        # epoch rec and kl loss
        epoch_rec_loss = 0
        epoch_kl_loss = 0

        for index, csi in enumerate(dataloader):

            # Move the data to the device GPU 
            csi = csi.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Encode CSI
            z, z_mean, z_log_var = model.encode(csi)

            # Decode the latent vector
            rec = model.decode(z)

            # Compute the loss
            rec_loss = model.rec_loss(rec, csi)
            kl_loss = model.kl_loss(z_mean, z_log_var)

            # Backward pass and optimization
            loss = rec_loss +  0.1 * kl_loss
            loss.backward()
            optimizer.step()

            print("batch index: ", index, "rec_loss: ", rec_loss.item(), "kl_loss: ", kl_loss.item())

            # Store the loss values
            reconstruction_losses.append(rec_loss.item())
            kl_losses.append(kl_loss.item())

            # epoch rec and kl loss
            epoch_rec_loss += rec_loss.item()
            epoch_kl_loss += kl_loss.item()
        
        # epoch rec and kl loss
        epoch_rec_loss /= len(dataloader)
        epoch_kl_loss /= len(dataloader)

        # Update the progress bar with the latest loss information
        pbar.set_postfix({'rec Loss': epoch_rec_loss, 'KL Loss': epoch_kl_loss})



    # save hisroty data
    np.save("results/VSAE_convergence/VSAE_reconstruction_losses.npy", reconstruction_losses)
    np.save("results/VSAE_convergence/VSAE_kl_losses.npy", kl_losses)

    # Save the model
    model_path = "./VSAE_{}.pt".format(intermediate_channels)
    torch.save(model.state_dict(), model_path)



    plt.plot(smooth_curve(reconstruction_losses), label='Reconstruction Loss')  # Adjust the line width as needed
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')  # Apply a logarithmic scale to the y-axis
    plt.legend()
    plt.tight_layout()  # Adjust the layout to fit everything nicely


    plt.plot(smooth_curve(kl_losses), label='KL Loss')  # Adjust the line width as needed
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')  # Apply a logarithmic scale to the y-axis
    plt.legend()
    plt.tight_layout()  # Adjust the layout to fit everything nicely


    plt.savefig("results/VSAE_convergence/VSAE_smoothed_reconstruction_loss.png")  # Save the figure
    # plt.show()  # Display the plot

 

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt


    network = WirelessNetwork(net_par)  
    csi = network.csi.unsqueeze(0).to(device)
    z, z_mean, z_log_var = model.encode(csi)


    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(z_mean.squeeze().cpu().detach().numpy())

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(' Embeddings Visualized with PCA')
    plt.savefig("results/VSAE_convergence/VSAE_PCA.png")  # Save the figure
    # plt.show()  # Display the plot
