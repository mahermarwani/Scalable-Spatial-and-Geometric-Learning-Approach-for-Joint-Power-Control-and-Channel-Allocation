import torch
from Spacial_embedding import VSAE
from Geometric_embedding import VGAE, build_graph
from torch_geometric.nn import MLP
from tqdm import tqdm
from wireless_config.WirelessNetwork import WirelessNetwork, cal_net_metrics
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import random
import time
from comparision_solutions.GA_solver import GA_solver
import numpy as np
import seaborn as sns


class ResidualBlockFC(nn.Module):
    def __init__(self, input_features, output_features, activation_fn="ReLU", dropout_prob=0):
        super(ResidualBlockFC, self).__init__()
        self.fc1 = nn.Linear(input_features, output_features)
        self.bn1 = nn.BatchNorm1d(output_features)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(output_features, output_features)
        self.bn2 = nn.BatchNorm1d(output_features)
        self.dropout2 = nn.Dropout(dropout_prob)

        if activation_fn == "ReLU":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "Sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation_fn == "Softmax":
            self.activation_fn = nn.Softmax(dim=1)

        if input_features != output_features:
            self.adjust_dims = nn.Sequential(
                nn.Linear(input_features, output_features),
                nn.BatchNorm1d(output_features),
                nn.Dropout(dropout_prob)
            )
        else:
            self.adjust_dims = None

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.activation_fn(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        if self.adjust_dims is not None:
            identity = self.adjust_dims(x)

        out += identity
        out = self.activation_fn(out)

        return out

class ResidualNetwork(nn.Module):
    def __init__(self, layer_sizes, activation_fn="ReLU", final_activation_fn="ReLU"):
        super(ResidualNetwork, self).__init__()
        blocks = []
        for i in range(len(layer_sizes) - 2):
            blocks.append(ResidualBlockFC(layer_sizes[i], layer_sizes[i+1], activation_fn))

        blocks.append(ResidualBlockFC(layer_sizes[-2], layer_sizes[-1], final_activation_fn))
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.res_blocks(x)

class ResidualBlockConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation_fn="ReLU", dropout_prob=0):
        super(ResidualBlockConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

        if activation_fn == "ReLU":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "Sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation_fn == "Softmax":
            self.activation_fn = nn.Softmax(dim=1)

        if in_channels != out_channels:
            self.adjust_dims = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),  # 1x1 convolution
            )
        else:
            self.adjust_dims = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.activation_fn(out)

        out = self.conv2(out)

        if self.adjust_dims is not None:
            identity = self.adjust_dims(x)

        out += identity
        out = self.activation_fn(out)

        return out

class ResidualNetworkConv(nn.Module):
    def __init__(self, channels, kernel_sizes, strides, paddings, activation_fn="ReLU", final_activation_fn="ReLU"):
        super(ResidualNetworkConv, self).__init__()
        blocks = []
        for i in range(len(channels) - 2):
            blocks.append(ResidualBlockConv(channels[i], channels[i+1], kernel_sizes[i], strides[i], paddings[i], activation_fn))

        blocks.append(ResidualBlockConv(channels[-2], channels[-1], kernel_sizes[-1], strides[-1], paddings[-1], final_activation_fn))
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.res_blocks(x)


class model_S(torch.nn.Module):
    def __init__(self, 
                    K,
                    latent_dim_node, latent_dim_edge, num_layers,
                    latent_dim_csi, intermediate_channels,
                    latent_dim_shared, num_layers_shared,
                    model_version = "VGSAE",
                    VGAE_path = None,
                    VSAE_path = None,
                ):
        
        super().__init__()

        self.K = K

        self.latent_dim_node = latent_dim_node
        self.latent_dim_edge = latent_dim_edge
        self.num_layers = num_layers

        self.intermediate_channels = intermediate_channels
        self.latent_dim_csi = latent_dim_csi

        self.latent_dim_shared = latent_dim_shared
        self.num_layers_shared = num_layers_shared

        self.model_version = model_version
        
        

        self.VSAE = VSAE(self.K, intermediate_channels=intermediate_channels)
        if VSAE_path is not None:
            self.VSAE.load_state_dict(torch.load(VSAE_path, weights_only=True))
        
        self.VGAE = VGAE(   n_node_features=self.K,
                            n_edge_features=2 * self.K,
                            n_node_hidden=latent_dim_node,
                            n_edge_hidden=latent_dim_edge,
                            num_layers=num_layers)
        if VGAE_path is not None:
            self.VGAE.load_state_dict(torch.load(VGAE_path, weights_only=True))


        if self.model_version == "VGAE":
            self.MLP_shared = ResidualNetwork([latent_dim_node + latent_dim_edge] + [latent_dim_shared] * self.num_layers_shared,
                                                activation_fn="ReLU",
                                                final_activation_fn="ReLU")
        elif self.model_version == "VSAE":
            self.MLP_shared = ResidualNetwork([latent_dim_csi] + [latent_dim_shared] * self.num_layers_shared,
                                                activation_fn="ReLU",
                                                final_activation_fn="ReLU")
        elif self.model_version == "VGSAE":
            self.MLP_shared = ResidualNetwork([latent_dim_node + latent_dim_edge + latent_dim_csi] + [latent_dim_shared] * self.num_layers_shared,
                                                activation_fn="ReLU",
                                                final_activation_fn="ReLU")
            


        self.MLP_p = ResidualNetwork([latent_dim_shared, latent_dim_shared, latent_dim_shared, latent_dim_shared, 1],
                                         activation_fn="ReLU",
                                         final_activation_fn="Sigmoid")

        self.MLP_rb = ResidualNetwork([latent_dim_shared, latent_dim_shared, latent_dim_shared, latent_dim_shared, self.K],
                                         activation_fn="ReLU",
                                         final_activation_fn="Softmax")
        
    

        self.adapt = ResidualNetworkConv(       [1, latent_dim_csi],
                                                [3, 3],
                                                [1, 1],
                                                [1, 1],
                                                activation_fn="ReLU",
                                                final_activation_fn="ReLU")


    def encode(self, csi, g):
        z_csi, mu_csi, logvar_csi= self.VSAE.encode(csi)
        z_node, z_edge, mu_node, mu_edge, logvar_node, logvar_edge = self.VGAE.encode(g.x, g.edge_index, edge_attr=g.edge_attr)

        return z_csi, z_node, z_edge, mu_csi, mu_node, mu_edge, logvar_csi, logvar_node, logvar_edge


    def decode(self, z_csi, z_node, z_edge):
        rec_csi = self.VSAE.decode(z_csi)
        rec_node, rec_edge = self.VGAE.decode(z_node, z_edge)

        return rec_csi, rec_node, rec_edge


    def forward(self, z_csi, z_node, z_edge):

        z_s = self.adapt(z_csi.unsqueeze(0)).mean(2).reshape(-1, self.latent_dim_csi)

        z_edge = nn.AdaptiveAvgPool2d((z_node.size(0), self.latent_dim_edge))(z_edge.unsqueeze(0)).squeeze(0)

        if self.model_version == "VGAE":
            z_shared = self.MLP_shared(torch.cat([z_node,z_edge], dim=1))
        elif self.model_version == "VSAE":
            z_shared = self.MLP_shared(z_s)
        elif self.model_version == "VGSAE":
            z_shared = self.MLP_shared(torch.cat([z_node, z_s, z_edge], dim=1))

        p = self.MLP_p(z_shared)
        rb = self.MLP_rb(z_shared)

        return z_shared, p, rb
        
        
    def model_execution(self, 
                        csi,
                        net_par,
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        with_sampling=False,
                        min_rate=300,
                        num_epochs=900,
                        learning_rate=0.01,
                        return_history=False,
                        return_history_sol=False,
                        return_all_user_rates=False  
                        ):

        model = self.to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        EE_history, RATE_history, violation_history = [], [], []
        p_history, rb_history = [], []
        all_user_rates_history = []  # <--- COLLECT USER RATES HERE

        g = build_graph(csi, device).to(device)

        for epoch in range(num_epochs):

            optimizer.zero_grad()

            z_csi, z_node, z_edge, mu_csi, mu_node, mu_edge, logvar_csi, logvar_node, logvar_edge = model.encode(csi, g)
            rec_csi, rec_node, rec_edge = model.decode(z_csi, z_node, z_edge)

            if with_sampling:
                z_shared, p, rb = model.forward(z_csi, z_node, z_edge)
            else:
                z_shared, p, rb = model.forward(mu_csi, mu_node, mu_edge)

            # Compute the loss
            loss_EEs, loss_rates = cal_net_metrics(csi, rb, p.squeeze(1), net_par, device=device)

            if self.model_version == "VGAE":
                kl_VGAE = model.VGAE.kl_loss(mu_node, logvar_node) + model.VGAE.kl_loss(mu_edge, logvar_edge)
                kl_VSAE = 0
                rec_VGAE = model.VGAE.rec_node_loss(rec_node, g.x) + model.VGAE.rec_edge_loss(rec_edge, g.edge_attr)
                rec_VSAE = 0
            elif self.model_version == "VSAE":
                kl_VGAE = 0
                kl_VSAE = model.VSAE.kl_loss(mu_csi, logvar_csi)
                rec_VGAE = 0
                rec_VSAE = model.VSAE.rec_loss(rec_csi, csi)
            elif self.model_version == "VGSAE":
                kl_VGAE = model.VGAE.kl_loss(mu_node, logvar_node) + model.VGAE.kl_loss(mu_edge, logvar_edge)
                kl_VSAE = model.VSAE.kl_loss(mu_csi, logvar_csi)
                rec_VGAE = model.VGAE.rec_node_loss(rec_node, g.x) + model.VGAE.rec_edge_loss(rec_edge, g.edge_attr)
                rec_VSAE = model.VSAE.rec_loss(rec_csi, csi)

            const_loss = F.relu(3 * min_rate - loss_rates).sum()
            total_loss = -1 * loss_rates.mean() + const_loss - loss_rates.min() + kl_VGAE + kl_VSAE + rec_VGAE + rec_VSAE 

            total_loss.backward()
            optimizer.step()

            p = p.squeeze(1)
            rb = torch.eye(self.K, device=device)[rb.argmax(dim=1)]

            EEs, rates = cal_net_metrics(csi, rb, p, net_par, device=device)

            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, EE: {EEs.mean().item():.3f}, MEAN RATE: {rates.mean().item():.3f}, MIN RATE:{rates.min().item():.3f}, Violations RATE: {torch.sum(rates < min_rate).item()}")

            if return_history:
                EE_history.append(EEs.mean().item())
                RATE_history.append(rates.mean().item())
                violation_history.append(torch.sum(rates < min_rate).item())

            if return_history_sol:
                p_history.append(p)
                rb_history.append(rb.argmax(dim=1))

            if return_all_user_rates:
                all_user_rates_history.append(rates.detach().cpu().numpy())  # <--- ADDITION

        # Assemble return values
        return_vals = []
        if return_history:
            return_vals.extend([EE_history, RATE_history, violation_history])
        if return_history_sol:
            return_vals.extend([p_history, rb_history])
        if return_all_user_rates:
            return_vals.append(all_user_rates_history)

        if return_vals:
            return tuple(return_vals)
        else:
            return p, rb




# Apply smoothing function to the loss data
def smooth_curve(points, factor=0.97):  # Adjust the factor as needed
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
               "N": 20,
               "K": 5}
    
    
    K = net_par["K"]
    model_version = "VGSAE"

    latent_dim_node = 64
    latent_dim_edge = 64
    latent_dim_csi = 128
    latent_dim_shared = 256
    num_layers_shared = 3
    num_layers = 3
    intermediate_channels = [16, 24, 32]
    VGAE_path= "checkpoints/vgsae_model/VGAE_node64_edge64_layers3_epochs100.pt"
    VSAE_path="checkpoints/vgsae_model/VSAE_[16, 24, 32].pt"
    
    with_sampling = True
    min_rate = 500
    num_epochs = 500
    learning_rate = 0.001
    return_history = True
    return_history_sol = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model =  model_S(K, 
                     latent_dim_node, 
                     latent_dim_edge, 
                     num_layers, 
                     latent_dim_csi, 
                     intermediate_channels, 
                     latent_dim_shared, 
                     num_layers_shared,
                     model_version,
                     VGAE_path,
                     VSAE_path
                     ).to(device)
    
    network = WirelessNetwork(net_par)
    csi = network.csi.to(device)

    EE_history, RATE_history, violation_history = model.model_execution(    csi, 
                                                                            net_par,
                                                                            device, 
                                                                            with_sampling, 
                                                                            min_rate, 
                                                                            num_epochs,
                                                                            learning_rate,
                                                                            return_history,
                                                                            return_history_sol)
    

    print("MEAN RATE: ", RATE_history[-1])
    print("Violations RATE: ", violation_history[-1])



    # Enhanced visualization for metrics over epochs
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    import datetime  # Add direct import of datetime module

    # Set the style for a professional look
    plt.style.use('seaborn-v0_8-whitegrid')

    # Set font to be more professional
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Create a figure with three subplots sharing the x-axis
    fig, axs = plt.subplots(3, 1, figsize=(14, 16), sharex=True, constrained_layout=True)

    # Define consistent and professional colors
    colors = {
        'EE': '#1f77b4',          # Strong blue
        'Mean Rate': '#2ca02c',   # Strong green
        'Violations': '#d62728'   # Strong red
    }

    # Add a slight shadow effect to lines for depth
    shadow_offset = 0.5
    shadow_alpha = 0.2

    # First subplot: Energy Efficiency (EE)
    # Add shadow effect
    axs[0].plot(EE_history, color=colors['EE'] + '50', linewidth=3, zorder=1)
    # Main line
    axs[0].plot(EE_history, color=colors['EE'], linewidth=2.5, label="EE", zorder=2)
    axs[0].set_title("Energy Efficiency (EE)", fontweight='bold', pad=15)
    axs[0].set_ylabel("Energy Efficiency (EE)")
    axs[0].legend(loc='upper left', frameon=True, facecolor='white', framealpha=1, edgecolor='#dddddd')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    # Add subtle background gradient for better readability
    axs[0].set_facecolor('#f8f9fa')

    # Second subplot: Mean Rate
    # Add shadow effect
    axs[1].plot(RATE_history, color=colors['Mean Rate'] + '50', linewidth=3, zorder=1)
    # Main line
    axs[1].plot(RATE_history, color=colors['Mean Rate'], linewidth=2.5, label="Mean Rate", zorder=2)
    axs[1].set_title("Mean Rate", fontweight='bold', pad=15)
    axs[1].set_ylabel("Mean Rate")
    axs[1].legend(loc='upper right', frameon=True, facecolor='white', framealpha=1, edgecolor='#dddddd')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_facecolor('#f8f9fa')

    # Third subplot: Violations Rate
    # Add shadow effect
    axs[2].plot(violation_history, color=colors['Violations'] + '50', linewidth=3, zorder=1)
    # Main line
    axs[2].plot(violation_history, color=colors['Violations'], linewidth=2.5, label="Violations Rate", zorder=2)
    axs[2].set_title("Violations Rate", fontweight='bold', pad=15)
    axs[2].set_xlabel("Epochs", fontweight='bold')
    axs[2].set_ylabel("violation count")
    axs[2].legend(loc='upper right', frameon=True, facecolor='white', framealpha=1, edgecolor='#dddddd')
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].set_facecolor('#f8f9fa')

    # Make spines (the box around the plot) slightly darker for definition
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(1.5)

    # Add tick marks pointing outward for better readability
    for ax in axs:
        ax.tick_params(direction='out', length=6, width=1.5, colors='#555555')
        
    # Add subtle horizontal lines at key value points for reference
    axs[0].axhline(y=np.mean(EE_history), color='#1f77b4', linestyle=':', alpha=0.6)
    axs[1].axhline(y=np.mean(RATE_history), color='#2ca02c', linestyle=':', alpha=0.6)
    axs[2].axhline(y=np.mean(violation_history), color='#d62728', linestyle=':', alpha=0.6)

    # Annotate important phases in training
    # Early phase
    axs[0].annotate('Rapid Growth', xy=(25, EE_history[25]), xytext=(50, EE_history[25]-500),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7))

    # Plateau phase for violations
    axs[2].annotate('Stabilization', xy=(200, violation_history[200]), xytext=(220, violation_history[200]+5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7))

    # Add main title with subtle background
    fig.suptitle("Metrics over epochs", fontsize=20, fontweight='bold', y=0.98)

    # Add subtle box around the entire figure for a polished look
    fig.patch.set_edgecolor('#dddddd')
    fig.patch.set_linewidth(2)

    # Add timestamp and version info for professional documentation
    # Fixed the datetime reference
    fig.text(0.02, 0.015, f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d")}', 
            fontsize=10, color='#555555')
    fig.text(0.98, 0.015, 'v1.2', fontsize=10, color='#555555', ha='right')

    # Save with high DPI for print quality
    plt.savefig("Metrics_over_epochs.png", dpi=300, bbox_inches='tight')
    plt.show()