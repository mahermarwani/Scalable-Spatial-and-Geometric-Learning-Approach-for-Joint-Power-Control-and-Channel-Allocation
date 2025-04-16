import sys
import os
import time
import statistics
import pickle
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import custom modules
from comparision_solutions.GA_solver import GA_solver
from comparision_solutions.cnn_model import CNN_model
from comparision_solutions.dnn_model import DNN_model
from comparision_solutions.gnn_model import Model, inference
from final_model import model_S
from wireless_config.WirelessNetwork import WirelessNetwork, cal_net_metrics


def ga_sol(net_par, csi, min_rate=300, eval=2000):
    """GA-based solution for wireless network optimization.
    
    Args:
        net_par: Network parameters dictionary
        csi: Channel State Information
        min_rate: Minimum rate requirement
        eval: Number of evaluations
        
    Returns:
        p: Power allocation
        rb: Resource block allocation
    """
    p, rb = GA_solver(csi, net_par, eval=eval, min_rate=min_rate)
    return p, rb


def cnn_sol(net_par, csi, min_rate, N, device="cpu"):
    """CNN-based solution for wireless network optimization.
    
    Args:
        net_par: Network parameters dictionary
        csi: Channel State Information
        min_rate: Minimum rate requirement
        N: Number of users
        device: Computation device (cpu/cuda)
        
    Returns:
        p: Power allocation
        rb: Resource block allocation
    """
    csi = csi.to(device)

    model = CNN_model(net_par).to(device)
    model_path = f'checkpoints/cnn_model/CNN_model_N={N}.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))

    p, rb = model(csi)
    p = p.squeeze(1)
    # Transform rb to one-hot vector with argmax
    rb = torch.eye(net_par["K"], device=device)[rb.argmax(dim=1)]
    return p, rb


def dnn_sol(net_par, csi, min_rate, N, device="cpu"):
    """DNN-based solution for wireless network optimization.
    
    Args:
        net_par: Network parameters dictionary
        csi: Channel State Information
        min_rate: Minimum rate requirement
        N: Number of users
        device: Computation device (cpu/cuda)
        
    Returns:
        p: Power allocation
        rb: Resource block allocation
    """
    csi = csi.to(device)

    model = DNN_model(net_par).to(device)
    model_path = f'checkpoints/dnn_model/DNN_model_N={N}.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))

    p, rb = model(csi)
    p = p.squeeze(1)
    # Transform rb to one-hot vector with argmax
    rb = torch.eye(net_par["K"], device=device)[rb.argmax(dim=1)]
    return p, rb


def gnn_sol(net_par, csi, min_rate, evaluations=300, device="cpu"):
    """GNN-based solution for wireless network optimization.
    
    Args:
        net_par: Network parameters dictionary
        csi: Channel State Information
        min_rate: Minimum rate requirement
        evaluations: Number of evaluations
        device: Computation device (cpu/cuda)
        
    Returns:
        p: Power allocation
        rb: Resource block allocation
    """
    csi = csi.to(device)

    # Define the model
    model = Model().to(device)
    model_path = 'checkpoints/gnn_model/GNN_model.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    p, rb = inference(model, csi, net_par, device, min_rate, evaluations)
    return p, rb


def my_model_sol(net_par, csi, min_rate, evaluations=300, device="cpu"):
    """Custom model solution for wireless network optimization.
    
    Args:
        net_par: Network parameters dictionary
        csi: Channel State Information
        min_rate: Minimum rate requirement
        evaluations: Number of evaluations
        device: Computation device (cpu/cuda)
        
    Returns:
        p: Power allocation
        rb: Resource block allocation
    """
    csi = csi.to(device)
    
    K = net_par["K"]
    model_version = "VGSAE"

    # Model hyperparameters
    latent_dim_node = 64
    latent_dim_edge = 64
    latent_dim_csi = 128
    latent_dim_shared = 300
    num_layers_shared = 3
    num_layers = 3
    intermediate_channels = [16, 24, 32]
    VGAE_path = "checkpoints/vgsae_model/VGAE_node64_edge64_layers3_epochs100.pt"
    VSAE_path = "checkpoints/vgsae_model/VSAE_[16, 24, 32].pt"
    
    # Execution parameters
    with_sampling = True
    num_epochs = evaluations
    learning_rate = 0.001
    return_history = False
    return_history_sol = False

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
                   VSAE_path).to(device)

    p, rb = model.model_execution(csi, 
                                 net_par,
                                 device, 
                                 with_sampling, 
                                 min_rate, 
                                 num_epochs,
                                 learning_rate,
                                 return_history,
                                 return_history_sol)

    return p, rb


def smooth_curve(points, factor=0.8):
    """Apply smoothing function to the data.
    
    Args:
        points: Data points to smooth
        factor: Smoothing factor (0-1)
        
    Returns:
        smoothed_points: Smoothed data
    """
    smoothed_points = np.empty_like(points)
    smoothed_points[0] = points[0]
    for i in range(1, len(points)):
        smoothed_points[i] = smoothed_points[i - 1] * factor + points[i] * (1 - factor)
    return smoothed_points


def plot_cdfs(index):
    """Generate and save CDF plots for different methods.
    
    Args:
        index: Index for file naming
    """
    # Define network parameters
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c_min_list = [300, 400, 500]
    rates = {key: -1 for key in ["GA", "CNN", "DNN", "GNN", "OURS"]}

    for c_min in c_min_list:
        print(f"****************************** {c_min} ***************************************")

        # Create wireless network and get CSI
        network = WirelessNetwork(net_par)
        csi = network.csi
        
        # Apply all solutions
        p_ga, rb_ga = ga_sol(net_par, csi, min_rate=c_min, eval=1000)
        p_cnn, rb_cnn = cnn_sol(net_par, csi, min_rate=c_min, N=net_par["N"])
        p_dnn, rb_dnn = dnn_sol(net_par, csi, min_rate=c_min, N=net_par["N"])
        p_gnn, rb_gnn = gnn_sol(net_par, csi, min_rate=c_min, evaluations=300, device="cpu")
        p_my_model, rb_my_model = my_model_sol(net_par, csi, min_rate=c_min, evaluations=300, device=device)

        # Calculate network metrics
        _, rates_ga = cal_net_metrics(csi, rb_ga, p_ga, net_par)
        _, rates_cnn = cal_net_metrics(csi, rb_cnn, p_cnn, net_par)
        _, rates_dnn = cal_net_metrics(csi, rb_dnn, p_dnn, net_par)
        _, rates_gnn = cal_net_metrics(csi, rb_gnn, p_gnn, net_par)
        _, rates_my_model = cal_net_metrics(csi, rb_my_model.to("cpu"), p_my_model.to("cpu"), net_par)

        # Store rates
        rates["GA"], rates["CNN"], rates["DNN"], rates["GNN"], rates["OURS"] = rates_ga, rates_cnn, rates_dnn, rates_gnn, rates_my_model

        # Create folder and save rates
        folder_path = os.path.join("results", "cdf")
        os.makedirs(folder_path, exist_ok=True)  # Ensure directory exists
        
        with open(os.path.join(folder_path, f"rates_MIN_RATE_{c_min}_{index}.pkl"), 'wb') as f:
            pickle.dump(rates, f)


def simulation_test():
    """Run simulation tests to compare different methods."""
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
    
    DEVICE = "cpu"
    Number_of_simulation = 10
    # Parameters that could be varied in future tests
    # N_values = [20, 30, 40]
    # wd_values = [50, 60, 70, 80]
    # shadow_std_values = [4, 7, 10, 12]
    # min_rate_values = [200, 300, 400, 500]

    # Fixed parameters for this test
    net_par["N"] = 30
    net_par["wc"] = 50
    net_par["shadow_std"] = 8
    MIN_RATE = 300
 
    # Lists to store results
    history_ga, history_cnn, history_dnn, history_gnn, history_my_model = [], [], [], [], []

    for i in range(Number_of_simulation):
        print(f"Running simulation {i+1}/{Number_of_simulation}")
        
        # Create new network for each simulation
        network = WirelessNetwork(net_par)
        csi = network.csi

        # Apply all solutions
        p_ga, rb_ga = ga_sol(net_par, csi, min_rate=MIN_RATE, eval=1000)
        p_cnn, rb_cnn = cnn_sol(net_par, csi, min_rate=MIN_RATE, N=net_par["N"], device=DEVICE)
        p_dnn, rb_dnn = dnn_sol(net_par, csi, min_rate=MIN_RATE, N=net_par["N"], device=DEVICE)
        p_gnn, rb_gnn = gnn_sol(net_par, csi, min_rate=MIN_RATE, evaluations=300, device=DEVICE)
        p_my_model, rb_my_model = my_model_sol(net_par, csi, min_rate=MIN_RATE, evaluations=300, device=DEVICE)

        # Calculate network metrics
        _, rates_ga = cal_net_metrics(csi, rb_ga, p_ga, net_par)
        _, rates_cnn = cal_net_metrics(csi, rb_cnn, p_cnn, net_par)
        _, rates_dnn = cal_net_metrics(csi, rb_dnn, p_dnn, net_par)
        _, rates_gnn = cal_net_metrics(csi, rb_gnn, p_gnn, net_par)
        _, rates_my_model = cal_net_metrics(csi, rb_my_model.to("cpu"), p_my_model.to("cpu"), net_par)

        # Calculate QoS violation percentage
        QoS_violation_ga = (torch.sum(rates_ga < MIN_RATE) / net_par["N"]) * 100 
        QoS_violation_cnn = (torch.sum(rates_cnn < MIN_RATE) / net_par["N"]) * 100
        QoS_violation_dnn = (torch.sum(rates_dnn < MIN_RATE) / net_par["N"]) * 100
        QoS_violation_gnn = (torch.sum(rates_gnn < MIN_RATE) / net_par["N"]) * 100
        QoS_violation_my_model = (torch.sum(rates_my_model < MIN_RATE) / net_par["N"]) * 100

        # Print individual simulation results
        print("Network Sum Rate:")
        print(f"GA: {torch.mean(rates_ga).item():.2f}")
        print(f"CNN: {torch.mean(rates_cnn).item():.2f}")
        print(f"DNN: {torch.mean(rates_dnn).item():.2f}")
        print(f"GNN: {torch.mean(rates_gnn).item():.2f}")
        print(f"My Model: {torch.mean(rates_my_model).item():.2f}")

        print("QoS violation:")
        print(f"GA: {QoS_violation_ga.item():.2f}%")
        print(f"CNN: {QoS_violation_cnn.item():.2f}%")
        print(f"DNN: {QoS_violation_dnn.item():.2f}%")
        print(f"GNN: {QoS_violation_gnn.item():.2f}%")
        print(f"My Model: {QoS_violation_my_model.item():.2f}%")
        
        # Store results
        history_ga.append((torch.mean(rates_ga).item(), QoS_violation_ga.item()))
        history_cnn.append((torch.mean(rates_cnn).item(), QoS_violation_cnn.item()))
        history_dnn.append((torch.mean(rates_dnn).item(), QoS_violation_dnn.item()))
        history_gnn.append((torch.mean(rates_gnn).item(), QoS_violation_gnn.item()))
        history_my_model.append((torch.mean(rates_my_model).item(), QoS_violation_my_model.item()))

    # Print summary of results
    print("#" * 100)
    print(f"N = {net_par['N']}, wc = {net_par['wc']}, shadow_std = {net_par['shadow_std']}, min_rate = {MIN_RATE}")

    # Print average and standard deviation
    print("Average and Std Network Sum Rate:")
    print(f"GA: {statistics.mean([x[0] for x in history_ga]):.2f} ± {statistics.stdev([x[0] for x in history_ga]):.2f}")
    print(f"CNN: {statistics.mean([x[0] for x in history_cnn]):.2f} ± {statistics.stdev([x[0] for x in history_cnn]):.2f}")
    print(f"DNN: {statistics.mean([x[0] for x in history_dnn]):.2f} ± {statistics.stdev([x[0] for x in history_dnn]):.2f}")
    print(f"GNN: {statistics.mean([x[0] for x in history_gnn]):.2f} ± {statistics.stdev([x[0] for x in history_gnn]):.2f}")
    print(f"My Model: {statistics.mean([x[0] for x in history_my_model]):.2f} ± {statistics.stdev([x[0] for x in history_my_model]):.2f}")

    print("Average and Std QoS violation:")
    print(f"GA: {statistics.mean([x[1] for x in history_ga]):.2f} ± {statistics.stdev([x[1] for x in history_ga]):.2f}%")
    print(f"CNN: {statistics.mean([x[1] for x in history_cnn]):.2f} ± {statistics.stdev([x[1] for x in history_cnn]):.2f}%")
    print(f"DNN: {statistics.mean([x[1] for x in history_dnn]):.2f} ± {statistics.stdev([x[1] for x in history_dnn]):.2f}%")
    print(f"GNN: {statistics.mean([x[1] for x in history_gnn]):.2f} ± {statistics.stdev([x[1] for x in history_gnn]):.2f}%")
    print(f"My Model: {statistics.mean([x[1] for x in history_my_model]):.2f} ± {statistics.stdev([x[1] for x in history_my_model]):.2f}%")
    print("#" * 100)


def execution_time_test():
    """Test and compare execution times of different solutions."""
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
        "N": 30,  # Increased network size
        "K": 5
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIN_RATE = 600

    print(f"Running execution time test with N={net_par['N']}, MIN_RATE={MIN_RATE}")
    print(f"Using device: {DEVICE}")
    
    # Create network
    network = WirelessNetwork(net_par)
    csi = network.csi

    execution_times = {}

    # Measure execution time for GA solution
    print("Testing GA solution...")
    start_time = time.time()
    p_ga, rb_ga = ga_sol(net_par, csi, min_rate=MIN_RATE, eval=20000)
    execution_times['GA Solution'] = time.time() - start_time

    # CNN and DNN solutions are commented out in the original code
    # print("Testing CNN solution...")
    # start_time = time.time()
    # p_cnn, rb_cnn = cnn_sol(net_par, csi, min_rate=MIN_RATE, N=net_par["N"])
    # execution_times['CNN Solution'] = time.time() - start_time

    # print("Testing DNN solution...")
    # start_time = time.time()
    # p_dnn, rb_dnn = dnn_sol(net_par, csi, min_rate=MIN_RATE, N=net_par["N"])
    # execution_times['DNN Solution'] = time.time() - start_time

    # Measure execution time for GNN solution
    print("Testing GNN solution...")
    start_time = time.time()
    p_gnn, rb_gnn = gnn_sol(net_par, csi, min_rate=MIN_RATE, evaluations=1000, device="cpu")
    execution_times['GNN Solution'] = time.time() - start_time

    # Measure execution time for custom model solution
    print("Testing custom model solution...")
    start_time = time.time()
    p_my_model, rb_my_model = my_model_sol(net_par, csi, min_rate=MIN_RATE, evaluations=300, device=DEVICE)
    execution_times['My Model Solution'] = time.time() - start_time

    # Print results
    print("\nExecution Time Results:")
    for key, value in execution_times.items():
        print(f"{key}: {value:.2f} seconds")


if __name__ == '__main__':
    # Choose which test to run
    print("Running simulation test...")
    simulation_test()
    
    # Uncomment to run other tests
    # print("\nRunning CDF plotting...")
    # plot_cdfs(index=1)
    
    # print("\nRunning execution time test...")
    # execution_time_test()