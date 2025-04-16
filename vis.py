from final_model import model_S
from wireless_config.WirelessNetwork import WirelessNetwork
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

if __name__ == '__main__':

    # ----------------- Network and Model Setup ------------------
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
        "N": 20,
        "K": 5
    }

    K = net_par["K"]
    model_version = "VGSAE"

    latent_dim_node = 64
    latent_dim_edge = 64
    latent_dim_csi = 128
    latent_dim_shared = 256
    num_layers_shared = 3
    num_layers = 3
    intermediate_channels = [16, 24, 32]

    VGAE_path = "checkpoints/vgsae_model/VGAE_node64_edge64_layers3_epochs100.pt"
    VSAE_path = "checkpoints/vgsae_model/VSAE_[16, 24, 32].pt"

    with_sampling = False
    min_rate = 500
    num_epochs = 500
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # ----------------- Initialize Model ------------------
    model = model_S(K, 
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

    # ----------------- Generate Data and Run Model ------------------
    network = WirelessNetwork(net_par)
    csi = network.csi.to(device)

    # Obtain user rates for each epoch; assume model_execution returns a tuple or list
    rates_over_epochs = model.model_execution(
        csi, 
        net_par,
        device, 
        with_sampling, 
        min_rate, 
        num_epochs,
        learning_rate,
        return_all_user_rates=True
    )

    # If the returned object is a tuple or list, extract the needed rates array
    rates_over_epochs = np.array(rates_over_epochs)[0]  # shape: (Number_of_epochs, Number_of_users)
    print("rates_over_epochs shape:", rates_over_epochs.shape)

    Number_of_epochs = rates_over_epochs.shape[0]
    Number_of_users = rates_over_epochs.shape[1]

    print("Number_of_epochs:", Number_of_epochs)
    print("Number_of_users:", Number_of_users)

    # ----------------- Interpolate Rates for Smoother Animation ------------------
    def interpolate_rates(rates, num_interpolated_frames=5):
        epochs, users = rates.shape
        x = np.arange(epochs)
        x_interpolated = np.linspace(0, epochs - 1, epochs * num_interpolated_frames)
        interpolated_rates = np.zeros((len(x_interpolated), users))
        for user in range(users):
            f = interp1d(x, rates[:, user], kind='linear')
            interpolated_rates[:, user] = f(x_interpolated)
        return interpolated_rates

    interpolated_rates_over_epochs = interpolate_rates(rates_over_epochs)
    Number_of_interpolated_epochs = interpolated_rates_over_epochs.shape[0]

    # ----------------- Create Bar Chart Animation ------------------
    def create_bar_chart_animation():
        # QoS threshold value
        QoS_threshold = min_rate

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare positions for each user along the Y-axis
        y_positions = np.arange(Number_of_users)
        
        # Fix the x-axis limits across all frames for smooth transitions
        overall_min = np.min(interpolated_rates_over_epochs) * 0.9
        overall_max = np.max(interpolated_rates_over_epochs) * 0.7

        # Normalize rates for color mapping
        norm = Normalize(vmin=np.min(interpolated_rates_over_epochs), vmax=np.max(interpolated_rates_over_epochs))
        cmap = plt.cm.viridis  # Choose a colormap
        sm = ScalarMappable(cmap=cmap, norm=norm)

        # Add a colorbar to the plot
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label("Rate", fontsize=12)

        # Create a placeholder for statistics text
        stats_text = ax.text(
            0.95, 0.95, "", 
            transform=ax.transAxes, 
            fontsize=12, 
            verticalalignment='top', 
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )

        def init():
            """Initialize the bars and stats for the first epoch."""
            ax.clear()
            
            # Re-create stats_text after clearing
            stats_text_local = ax.text(
                0.95, 0.95, "", 
                transform=ax.transAxes, 
                fontsize=12, 
                verticalalignment='top', 
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
            )
            
            # Make a horizontal bar chart for the first epoch
            bar_container = ax.barh(
                y_positions,
                interpolated_rates_over_epochs[0], 
                color=[sm.to_rgba(rate) for rate in interpolated_rates_over_epochs[0]], 
                edgecolor='black'
            )
            
            # Add a vertical QoS threshold line
            threshold_line = ax.axvline(
                QoS_threshold,
                color='red',
                linestyle='--',
                linewidth=2,
                label='QoS threshold'
            )
            
            # Set y-ticks to label each user
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f"User {i+1}" for i in range(Number_of_users)])
            # Invert Y-axis so User 1 is at the top
            ax.invert_yaxis()

            # Set axis labels/limits/title
            ax.set_xlim([overall_min, overall_max])
            ax.set_xlabel("Rate", fontsize=12)
            ax.set_ylabel("User", fontsize=12)
            ax.set_title(f"Distribution of User Rates - Epoch 1/{Number_of_epochs}", fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Show legend for QoS line
            ax.legend(loc='upper left')
            
            # Calculate statistics for the first epoch
            current_rates = interpolated_rates_over_epochs[0]
            mean_rate = np.mean(current_rates)
            min_rate_cur = np.min(current_rates)
            max_rate_cur = np.max(current_rates)
            
            # Update the statistics text for the first epoch
            stats_text_local.set_text(
                f"Epoch 1/{Number_of_epochs}\n"
                f"Mean Rate: {mean_rate:.2f}\n"
                f"Min Rate: {min_rate_cur:.2f}\n"
                f"Max Rate: {max_rate_cur:.2f}"
            )
            
            # Return the bars, stats text, and threshold line so blit can track them
            return [*bar_container, stats_text_local, threshold_line]

        def animate(epoch):
            print(f"Animating epoch {epoch + 1}/{Number_of_interpolated_epochs}")
            """Update the existing bars with rates for the given epoch."""
            
            # 'bars' is ax.containers[0]
            bars = ax.containers[0]
            # stats_text is ax.texts[0], threshold_line is ax.lines[-1] if the only line
            stats_text_local = ax.texts[0]
            threshold_line = ax.lines[-1]
            
            # Update each bar's width and color
            for i, bar in enumerate(bars):
                bar.set_width(interpolated_rates_over_epochs[epoch, i])
                bar.set_color(sm.to_rgba(interpolated_rates_over_epochs[epoch, i]))

            # Convert from interpolated epoch number to original epoch number
            original_epoch = int(epoch * Number_of_epochs / Number_of_interpolated_epochs) + 1
            
            # Update the plot title to show current epoch
            ax.set_title(f"Distribution of User Rates - Epoch {original_epoch}/{Number_of_epochs}")

            # Calculate statistics for the current epoch
            current_rates = interpolated_rates_over_epochs[epoch]
            mean_rate = np.mean(current_rates)
            min_rate_cur = np.min(current_rates)
            max_rate_cur = np.max(current_rates)

            # Update the statistics text
            stats_text_local.set_text(
                f"Epoch {original_epoch}/{Number_of_epochs}\n"
                f"Mean Rate: {mean_rate:.2f}\n"
                f"Min Rate: {min_rate_cur:.2f}\n"
                f"Max Rate: {max_rate_cur:.2f}"
            )

            # Return bars, updated stats text, and threshold line
            return [*bars, stats_text_local, threshold_line]

        # Create the animation
        anim = FuncAnimation(
            fig,
            animate,
            frames=Number_of_interpolated_epochs,
            init_func=init,
            interval=100,
            blit=True, 
            repeat=True
        )

        # Save the animation to a GIF
        anim_path = "rate_animation.gif"
        anim.save(anim_path, writer='pillow', fps=60)
        print(f"Animation saved to {anim_path}")

        plt.show()

    # Generate the bar chart animation
    create_bar_chart_animation()
