from torch.nn.functional import leaky_relu
from wireless_config.WirelessNetwork import WirelessNetwork
from torch_geometric.utils import scatter, softmax
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import random
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from tqdm import tqdm
import torch.nn.functional as F
from wireless_config.WirelessNetwork import WirelessNetwork
from torch_geometric.nn import MLP, GCNConv, GATConv, BatchNorm
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax
from torch.nn.functional import leaky_relu
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA


class LRGATConv(torch.nn.Module):
    def __init__(self, n_node_features, n_edge_features, n_node_hidden, n_edge_hidden):
        super(LRGATConv, self).__init__()
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_node_hidden = n_node_hidden
        self.n_edge_hidden = n_edge_hidden

        # parameters for node feature transformation
        self.W_h = MLP([self.n_node_features, n_node_hidden], act = 'relu')
        self.a = MLP([2 * n_node_hidden, 1], act = 'relu') # Adjusted input size for node attention

        # parameters for edge feature transformation
        self.W_e = MLP([self.n_edge_features, n_edge_hidden], act = 'relu')

        self.b = MLP([n_edge_hidden + n_node_hidden, 1], act = 'relu')



    def update_nodes_opt(self, x, edge_index, edge_attr, batch, include_self_loops=True):
        num_nodes = x.size(0)
        src_index, dst_index = edge_index

        if include_self_loops:
            self_loop_edge = torch.arange(num_nodes, dtype=edge_index.dtype, device=edge_index.device).unsqueeze(0)
            self_loop_edge = torch.cat([self_loop_edge, self_loop_edge], dim=0)
            edge_index = torch.cat([edge_index, self_loop_edge], dim=1)
            src_index, dst_index = edge_index

        h_transformed = self.W_h(x)
        attn_input = torch.cat([h_transformed[src_index], h_transformed[dst_index]], dim=-1)
        attn_scores = F.leaky_relu(self.a(attn_input)).squeeze(-1)

        # Normalize attention scores using softmax per source node
        att_coeff = softmax(attn_scores, index=src_index, num_nodes=num_nodes)

        # Update node features
        out = scatter(att_coeff.unsqueeze(-1) * h_transformed[dst_index], src_index, dim=0, dim_size=num_nodes)

        return out


    def update_edges_opt(self, x, edge_index, edge_attr, batch):
        src_index, dst_index = edge_index
        edge_features = self.W_e(edge_attr)
        node_features = self.W_h(x)
        num_edges = edge_index.size(1)

        updated_edge_features = torch.zeros_like(edge_features)

        unique_batches = torch.unique(batch)

        for current_batch in unique_batches:
            batch_mask_nodes = (batch == current_batch)
            batch_node_indices = torch.nonzero(batch_mask_nodes).squeeze(1)
            if len(batch_node_indices) == 0:
                continue

            batch_mask_edges = (batch[src_index] == current_batch)
            batch_edge_indices = torch.nonzero(batch_mask_edges).squeeze(1)
            if len(batch_edge_indices) == 0:
                continue

            batch_src_local = src_index[batch_edge_indices]
            batch_dst_local = dst_index[batch_edge_indices]
            batch_edge_features_local = edge_features[batch_edge_indices]
            num_local_edges = batch_edge_indices.size(0)

            # ---------- LEFT NEIGHBORS ATTENTION ----------
            target_src = batch_src_local.unsqueeze(0) # shape [1, num_local_edges]
            neighbor_dst = batch_dst_local.unsqueeze(1) # shape [num_local_edges, 1]
            target_dst_expanded = batch_dst_local.unsqueeze(0) # shape [1, num_local_edges]
            neighbor_src_expanded = batch_src_local.unsqueeze(1) # shape [num_local_edges, 1]

            left_neighbor_mask = (neighbor_dst == target_src) & (neighbor_src_expanded != target_dst_expanded)
            left_neighbor_indices = torch.nonzero(left_neighbor_mask, as_tuple=True)
            if left_neighbor_indices[0].numel() > 0:
                left_target_indices_local = left_neighbor_indices[0]
                left_neighbor_indices_local = left_neighbor_indices[1]
                left_target_edges = batch_edge_indices[left_target_indices_local]
                left_neighbor_edges = batch_edge_indices[left_neighbor_indices_local]
                left_neighbors_features = edge_features[left_neighbor_edges]
                left_neighbors_nodes = src_index[left_neighbor_edges]
                left_attn_inputs = torch.cat([left_neighbors_features, node_features[left_neighbors_nodes]], dim=-1)
                left_attn_scores = F.leaky_relu(self.b(left_attn_inputs)).squeeze(-1)
                left_att_coeffs = softmax(left_attn_scores, left_target_edges, dim=0)
                left_weighted_features = left_att_coeffs.unsqueeze(-1) * left_neighbors_features
                updated_edge_features = updated_edge_features.scatter_add(0, left_target_edges.unsqueeze(-1).expand(-1, edge_features.size(-1)), left_weighted_features)

            # ---------- RIGHT NEIGHBORS ATTENTION ----------
            target_dst = batch_dst_local.unsqueeze(0) # shape [1, num_local_edges]
            neighbor_src = batch_src_local.unsqueeze(1) # shape [num_local_edges, 1]
            target_src_expanded = batch_src_local.unsqueeze(0) # shape [1, num_local_edges]
            neighbor_dst_expanded = batch_dst_local.unsqueeze(1) # shape [num_local_edges, 1]

            right_neighbor_mask = (neighbor_src == target_dst) & (neighbor_dst_expanded != target_src_expanded)
            right_neighbor_indices = torch.nonzero(right_neighbor_mask, as_tuple=True)
            if right_neighbor_indices[0].numel() > 0:
                right_target_indices_local = right_neighbor_indices[0]
                right_neighbor_indices_local = right_neighbor_indices[1]
                right_target_edges = batch_edge_indices[right_target_indices_local]
                right_neighbor_edges = batch_edge_indices[right_neighbor_indices_local]
                right_neighbors_features = edge_features[right_neighbor_edges]
                right_neighbors_nodes = dst_index[right_neighbor_edges]
                right_attn_inputs = torch.cat([right_neighbors_features, node_features[right_neighbors_nodes]], dim=-1)
                right_attn_scores = F.leaky_relu(self.b(right_attn_inputs)).squeeze(-1)
                right_att_coeffs = softmax(right_attn_scores, right_target_edges, dim=0)
                right_weighted_features = right_att_coeffs.unsqueeze(-1) * right_neighbors_features
                updated_edge_features = updated_edge_features.scatter_add(0, right_target_edges.unsqueeze(-1).expand(-1, edge_features.size(-1)), right_weighted_features)

        return updated_edge_features


    def forward(self, x, edge_index, edge_attr, batch):


        updated_node_features = self.update_nodes_opt(x, edge_index, edge_attr, batch, include_self_loops=True)

        updated_edge_features = self.update_edges_opt(x, edge_index, edge_attr, batch)

        return updated_node_features, updated_edge_features



class VGAE(torch.nn.Module):
    def __init__(self, n_node_features, n_edge_features, n_node_hidden, n_edge_hidden, num_layers):
        super(VGAE, self).__init__()

        self.num_layers = num_layers

        self.shared_convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.shared_convs.append(LRGATConv(n_node_features, n_edge_features, n_node_hidden, n_edge_hidden))
            else:
                self.shared_convs.append(LRGATConv(n_node_hidden, n_edge_hidden, n_node_hidden, n_edge_hidden))

        self.mu_conv = LRGATConv(n_node_hidden, n_edge_hidden, n_node_hidden, n_edge_hidden)
        self.logstd_conv = LRGATConv(n_node_hidden, n_edge_hidden, n_node_hidden, n_edge_hidden)

        self.edge_decoder = MLP([n_edge_hidden, n_edge_hidden, n_edge_hidden, n_edge_features], act='relu')
        self.node_decoder = MLP([n_node_hidden, n_node_hidden, n_node_hidden, n_node_features], act='relu')

    def encode(self, x, edge_index, edge_attr, batch= None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i in range(self.num_layers):
            if i == 0:
                x, edge_attr = self.shared_convs[i](x, edge_index, edge_attr, batch)
            else:
                x, edge_attr = self.shared_convs[i](F.relu(x), edge_index, F.relu(edge_attr), batch)

        mu_node, mu_edge = self.mu_conv(x, edge_index, edge_attr, batch)
        logvar_node, logvar_edge = self.logstd_conv(x, edge_index, edge_attr, batch)

        z_node = mu_node + torch.randn_like(logvar_node) * torch.exp(0.5 * logvar_node)
        z_edge = mu_edge + torch.randn_like(logvar_edge) * torch.exp(0.5 * logvar_edge)

        return z_node, z_edge, mu_node, mu_edge, logvar_node, logvar_edge

    def decode(self, z_node, z_edge):
        edge_attr = self.edge_decoder(z_edge)
        x = self.node_decoder(z_node)
        return x, edge_attr

    def kl_loss(self, mu, logvar):
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu ** 2 - torch.exp(logvar))
        return kl_divergence

    def rec_node_loss(self, x_rec, x):
        reconstruction_loss = F.mse_loss(x, x_rec, reduction='mean')
        return reconstruction_loss

    def rec_edge_loss(self, edge_rec, edge_attr):
        reconstruction_loss = F.mse_loss(edge_attr, edge_rec, reduction='mean')
        return reconstruction_loss






def build_graph(csi, device="cpu"):

    # csi normalization
    # log_x = torch.log10(csi)
    # mean_log_x = torch.mean(log_x)
    # variance_log_x = torch.var(log_x)
    # csi = (log_x - mean_log_x) / torch.sqrt(variance_log_x)


    # create the adjacency list of the interference graph
    adj = []

    for i in range(0, csi.shape[1]):
        for j in range(i + 1, csi.shape[1]):  # Start from i + 1 to avoid duplicates
            adj.append([i, j])


    """
    for i in range(0, csi.shape[1]):
        for j in range(0, csi.shape[1]):
            if i != j:
                adj.append([i, j])
                adj.append([j, i])
    """
    
    # construct edge features
    edge_features = [torch.cat([csi[:, e[0], e[1]], csi[:, e[1], e[0]]]) for e in adj]
    # edge_features = [csi[:, e[1], e[0]] for e in adj]
    # construct node features
    node_features = [torch.diag(csi[k, :, :]) for k in range(csi.shape[0])]

    # construct the graph
    graph = Data(x=torch.stack(node_features, dim=0).t(),
                 edge_index=torch.tensor(adj).t().contiguous(),
                 edge_attr=torch.stack(edge_features, dim=0)
                 )
    

    graph.to(device)
    # Normalize the graph
    # transform = T.Compose([ T.ToUndirected()])    
    # graph = transform(graph)

    return graph




# Apply smoothing function to the loss data
def smooth_curve(points, factor=0.97):  # Adjust the factor as needed
    smoothed_points = np.empty_like(points)
    smoothed_points[0] = points[0]
    for i in range(1, len(points)):
        smoothed_points[i] = smoothed_points[i - 1] * factor + points[i] * (1 - factor)
    return smoothed_points

       







if __name__ == '__main__':
    # Network parameters
    network_parameters = {
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

    # Model hyperparameters
    latent_dim_node = 64
    latent_dim_edge = 64
    num_layers = 3
    num_epochs = 100  # Set the number of epochs
    learning_rate = 0.001
    batch_size = 128
    dataset_length = 10000  # Number of graphs to generate for training/testing
    num_pca_graphs = 10  # Number of graphs to visualize with PCA

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu" # Uncomment this line to force CPU usage

    # Initialize the model
    model = VGAE(n_node_features=network_parameters["K"],
                 n_edge_features=2 * network_parameters["K"],
                 n_node_hidden=latent_dim_node,
                 n_edge_hidden=latent_dim_edge,
                 num_layers=num_layers)
                 
    model.to(device)
    print(model)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists to store loss values for each epoch
    node_rec_losses = []
    edge_rec_losses = []
    node_kl_losses = []
    edge_kl_losses = []

    # Store the last epoch's average training losses
    last_train_node_rec_loss = 0
    last_train_edge_rec_loss = 0
    last_train_node_kl_loss = 0
    last_train_edge_kl_loss = 0

    # Store the average testing losses
    test_node_rec_loss = 0
    test_edge_rec_loss = 0
    test_node_kl_loss = 0
    test_edge_kl_loss = 0
    num_test_batches = 0

    # --- Training Phase ---
    print("\n--- Training Phase ---")

    # Create the training dataset
    train_dataset = []
    for i in tqdm(range(dataset_length), desc="Creating Training Dataset"):
        network = WirelessNetwork(network_parameters)
        train_dataset.append(build_graph(network.csi, device))

    # Create the training dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        epoch_node_rec_loss = 0
        epoch_edge_rec_loss = 0
        epoch_node_kl_loss = 0
        epoch_edge_kl_loss = 0

        for index, graph in enumerate(train_dataloader):
            graph = graph.to(device)

            optimizer.zero_grad()

            # Encode the graph
            z_node, z_edge, mu_node, mu_edge, logstd_node, logstd_edge = model.encode(
                graph.x, graph.edge_index, edge_attr=graph.edge_attr, batch=graph.batch
            )

            # Decode the latent variables
            x_rec, edge_attr_rec = model.decode(z_node, z_edge)

            # Compute the losses
            kl_node_loss = model.kl_loss(mu_node, logstd_node)
            kl_edge_loss = model.kl_loss(mu_edge, logstd_edge)
            rec_node_loss = model.rec_node_loss(x_rec, graph.x)
            rec_edge_loss = model.rec_edge_loss(edge_attr_rec, graph.edge_attr)

            # Total loss
            total_loss = kl_node_loss + kl_edge_loss + rec_node_loss + rec_edge_loss

            # Perform backpropagation
            total_loss.backward()
            optimizer.step()

            # Print batch losses
            tqdm.write(
                f"Epoch: {epoch}, Batch: {index+1}/{len(train_dataloader)}, "
                f"KL Node Loss: {kl_node_loss.item():.7f}, KL Edge Loss: {kl_edge_loss.item():.7f}, "
                f"Node Rec Loss: {rec_node_loss.item():.7f}, Edge Rec Loss: {rec_edge_loss.item():.7f}"
            )

            # Save batch losses
            node_rec_losses.append(rec_node_loss.item())
            edge_rec_losses.append(rec_edge_loss.item())
            node_kl_losses.append(kl_node_loss.item())
            edge_kl_losses.append(kl_edge_loss.item())

            # Accumulate epoch losses
            epoch_node_rec_loss += rec_node_loss.item()
            epoch_edge_rec_loss += rec_edge_loss.item()
            epoch_node_kl_loss += kl_node_loss.item()
            epoch_edge_kl_loss += kl_edge_loss.item()

        # Calculate average epoch losses
        epoch_node_rec_loss /= len(train_dataloader)
        epoch_edge_rec_loss /= len(train_dataloader)
        epoch_node_kl_loss /= len(train_dataloader)
        epoch_edge_kl_loss /= len(train_dataloader)

        tqdm.write(
            f"Epoch: {epoch}, "
            f"Avg. Node KL Loss: {epoch_node_kl_loss:.7f}, Avg. Edge KL Loss: {epoch_edge_kl_loss:.7f}"
            f"Avg. Node Rec Loss: {epoch_node_rec_loss:.7f}, Avg. Edge Rec Loss: {epoch_edge_rec_loss:.7f}, "
        )

        # Store the last epoch's average training losses
        if epoch == num_epochs - 1:
            last_train_node_rec_loss = epoch_node_rec_loss
            last_train_edge_rec_loss = epoch_edge_rec_loss
            last_train_node_kl_loss = epoch_node_kl_loss
            last_train_edge_kl_loss = epoch_edge_kl_loss

    # --- Testing Phase ---
    print("\n--- Testing Phase ---")

    # Create the test dataset (consider using a separate dataset for actual testing)
    test_dataset = []
    for i in tqdm(range(dataset_length), desc="Creating Testing Dataset"):
        network = WirelessNetwork(network_parameters)
        test_dataset.append(build_graph(network.csi, device))

    # Create the test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluation loop (no gradient calculation needed)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for index, graph in enumerate(test_dataloader):
            graph = graph.to(device)
            # Encode the graph
            z_node, z_edge, mu_node, mu_edge, logstd_node, logstd_edge = model.encode(
                graph.x, graph.edge_index, edge_attr=graph.edge_attr, batch=graph.batch
            )

            # Decode the latent variables
            x_rec, edge_attr_rec = model.decode(z_node, z_edge)

            # Compute the losses
            kl_node_loss = model.kl_loss(mu_node, logstd_node)
            kl_edge_loss = model.kl_loss(mu_edge, logstd_edge)
            rec_node_loss = model.rec_node_loss(x_rec, graph.x)
            rec_edge_loss = model.rec_edge_loss(edge_attr_rec, graph.edge_attr)

            # Accumulate test losses
            test_node_rec_loss += rec_node_loss.item()
            test_edge_rec_loss += rec_edge_loss.item()
            test_node_kl_loss += kl_node_loss.item()
            test_edge_kl_loss += kl_edge_loss.item()
            num_test_batches += 1

            # Print test losses
            tqdm.write(
                f"Test Batch: {index+1}/{len(test_dataloader)}, "
                f"KL Node Loss: {kl_node_loss.item():.7f}, KL Edge Loss: {kl_edge_loss.item():.7f}, "
                f"Node Rec Loss: {rec_node_loss.item():.7f}, Edge Rec Loss: {rec_edge_loss.item():.7f}"
            )

    # Calculate average test losses
    if num_test_batches > 0:
        test_node_rec_loss /= num_test_batches
        test_edge_rec_loss /= num_test_batches
        test_node_kl_loss /= num_test_batches
        test_edge_kl_loss /= num_test_batches

    # --- Saving the Model and Results ---
    print("\n--- Saving Model and Results ---")

    model_filename = f"VGAE_node{latent_dim_node}_edge{latent_dim_edge}_layers{num_layers}_epochs{num_epochs}.pt"
    model_path = os.path.join("./", model_filename)  # Save in the current directory
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    results_dir = "results/VGAE_convergence"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    np.save(os.path.join(results_dir, "VGAE_node_rec_losses.npy"), node_rec_losses)
    np.save(os.path.join(results_dir, "VGAE_edge_rec_losses.npy"), edge_rec_losses)
    np.save(os.path.join(results_dir, "VGAE_node_kl_losses.npy"), node_kl_losses)
    np.save(os.path.join(results_dir, "VGAE_edge_kl_losses.npy"), edge_kl_losses)
    print(f"Loss data saved to: {results_dir}")

    # --- Plotting Loss Curves ---
    print("\n--- Plotting Loss Curves ---")

    loss_data = {
        'Feature Node Reconstruction Loss': node_rec_losses,
        'Feature Edge Reconstruction Loss': edge_rec_losses,
        'Feature Node KL Loss': node_kl_losses,
        'Feature Edge KL Loss': edge_kl_losses,
    }

    plt.figure(figsize=(12, 8))
    for i, (label, losses) in enumerate(loss_data.items()):
        plt.subplot(2, 2, i + 1)
        plt.plot(losses, label=label)
        plt.grid(True)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title(label)
        plt.legend()
        plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "VGAE_loss_curves.png"))
    print(f"Loss curves plot saved to: {results_dir}/VGAE_loss_curves.png")
    # plt.show() # Uncomment to display the plot

    # --- Visualizing Embeddings with PCA per Graph ---
    print("\n--- Visualizing Embeddings with PCA per Graph ---")

    num_rows = num_pca_graphs
    num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))

    test_dataset = []
    for i in tqdm(range(num_pca_graphs), desc="Creating Testing Dataset for PCA"):
        network = WirelessNetwork(network_parameters)
        test_dataset.append(build_graph(network.csi, device))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for i, graph in enumerate(test_dataloader):
            graph = graph.to(device)

            # Encode the graph
            z_node, z_edge, mu_node, mu_edge, logstd_node, logstd_edge = model.encode(
                graph.x, graph.edge_index, edge_attr=graph.edge_attr, batch=graph.batch
            )

            # Visualize node embeddings
            pca_node = PCA(n_components=2)
            node_embeddings_2d = pca_node.fit_transform(mu_node.cpu().detach().numpy())

            ax_node = axes[i, 0] if num_rows > 1 else axes[0]
            ax_node.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1])
            ax_node.set_xlabel('PCA Component 1')
            ax_node.set_ylabel('PCA Component 2')
            ax_node.set_title(f'Graph {i+1} - Node Embeddings')
            ax_node.grid(True)

            # Visualize edge embeddings
            ax_edge = axes[i, 1] if num_rows > 1 else axes[1]
            if mu_edge is not None and mu_edge.numel() > 0:
                pca_edge = PCA(n_components=2)
                edge_embeddings_2d = pca_edge.fit_transform(mu_edge.cpu().detach().numpy())
                ax_edge.scatter(edge_embeddings_2d[:, 0], edge_embeddings_2d[:, 1])
                ax_edge.set_xlabel('PCA Component 1')
                ax_edge.set_ylabel('PCA Component 2')
                ax_edge.set_title(f'Graph {i+1} - Edge Embeddings')
                ax_edge.grid(True)
            else:
                ax_edge.text(0.5, 0.5, "No Edges", ha='center', va='center')
                ax_edge.set_title(f'Graph {i+1} - Edge Embeddings')
                ax_edge.axis('off')

    plt.tight_layout()
    filename_combined_pca = os.path.join(results_dir, "VGAE_combined_embeddings_pca.png")
    plt.savefig(filename_combined_pca)
    print(f"Combined PCA plots saved to: {filename_combined_pca}")


    # --- Print Test vs Train Performance ---
    print("\n--- Test vs Train Performance ---")
    print(f"Last Epoch Training Performance:")
    print(f"  Avg. Node Rec Loss: {last_train_node_rec_loss:.7f}")
    print(f"  Avg. Edge Rec Loss: {last_train_edge_rec_loss:.7f}")
    print(f"  Avg. Node KL Loss: {last_train_node_kl_loss:.7f}")
    print(f"  Avg. Edge KL Loss: {last_train_edge_kl_loss:.7f}")
    print(f"\nAverage Testing Performance:")
    print(f"  Avg. Node Rec Loss: {test_node_rec_loss:.7f}")
    print(f"  Avg. Edge Rec Loss: {test_edge_rec_loss:.7f}")
    print(f"  Avg. Node KL Loss: {test_node_kl_loss:.7f}")
    print(f"  Avg. Edge KL Loss: {test_edge_kl_loss:.7f}")

    print("\n--- End of Script ---")