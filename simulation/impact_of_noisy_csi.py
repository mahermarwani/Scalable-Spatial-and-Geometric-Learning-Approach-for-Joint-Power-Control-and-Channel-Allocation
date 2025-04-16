import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from wireless_config.WirelessNetwork import WirelessNetwork, cal_net_metrics
from simulation_test import ga_sol, dnn_sol, cnn_sol, gnn_sol, my_model_sol
import statistics


def distortion(csi, fast_fading, r):
    # SNR_dB = 100  # SNR in decibels
    # SNR_linear = 10 ** (SNR_dB / 10)
    # csi_power = np.mean(csi)
    # # print(csi_power)
    # # exit()
    noise_variance = 1
    #
    noise_real = noise_variance * np.random.randn(*csi.shape)
    noise_imag = noise_variance * np.random.randn(*csi.shape)
    noise = noise_real + 1j * noise_imag
    #
    # csi_noisy = csi / np.abs(fast_fading) ** 2 * (
    #             r * np.abs(fast_fading + noise) ** 2 + (1 - r) * np.abs(fast_fading) ** 2)

    noisy_fading = np.abs(np.sqrt(1 - r ** 2) * fast_fading + r * noise) ** 2
    large_scale_fading = csi / np.abs(fast_fading) ** 2
    csi_noisy = large_scale_fading * noisy_fading
    return csi_noisy.float()


def plot_ANR_QoSpb_QoSviol(index):
    # Define network parameters
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
                "K": 5
            }
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    distortion_values = np.linspace(0, 1, 4)
    print(distortion_values)

    simulation_count = 100

    CSI = []
    for i in range(simulation_count):
        network = WirelessNetwork(net_par)
        csi, fast_fading = network.csi, network.fast_fading
        CSI.append((csi, fast_fading))

    # Define the minimum rate
    MIN_RATE = 300

    history_ga, history_cnn, history_dnn, history_gnn, history_my_model = [], [], [], [], []

    for i in range(simulation_count):
        iter_history_ga, iter_history_cnn, iter_history_dnn, iter_history_gnn, iter_history_my_model = [], [], [], [], []
        for dist_v in distortion_values:
            print("****************************** " + str(dist_v) + "****************************** ")
            d_csi = distortion(CSI[i][0], CSI[i][1], dist_v)


            p_ga, rb_ga = ga_sol(net_par, d_csi, min_rate=MIN_RATE, eval=3000)

            _, rates_d_ga = cal_net_metrics(d_csi, rb_ga, p_ga, net_par)
            QoS_viol_d_ga = (torch.sum(rates_d_ga < MIN_RATE) / net_par["N"]) * 100 

            _, rates_ga = cal_net_metrics(CSI[i][0], rb_ga, p_ga, net_par)
            QoS_viol_ga = (torch.sum(rates_ga < MIN_RATE) / net_par["N"]) * 100
            
            print("rate for solving with distorted csi:", rates_d_ga.mean().item())
            print("actual csi using previous sol:", rates_ga.mean().item())

            print("QoS violation for solving with distorted csi:", QoS_viol_d_ga.item())
            print("QoS violation using previous sol:", QoS_viol_ga.item())


            p_cnn, rb_cnn  = cnn_sol(net_par, d_csi, min_rate=MIN_RATE, N=net_par["N"])

            _, rates_d_cnn = cal_net_metrics(d_csi, rb_cnn, p_cnn, net_par)
            QoS_viol_d_cnn = (torch.sum(rates_d_cnn < MIN_RATE) / net_par["N"]) * 100

            _, rates_cnn = cal_net_metrics(CSI[i][0], rb_cnn, p_cnn, net_par)
            QoS_viol_cnn = (torch.sum(rates_cnn < MIN_RATE) / net_par["N"]) * 100

            print("rate for solving with distorted csi:", rates_d_cnn.mean().item())
            print("actual csi using previous sol:", rates_cnn.mean().item())

            print("QoS violation for solving with distorted csi:", QoS_viol_d_cnn.item())
            print("QoS violation using previous sol:", QoS_viol_cnn.item())

            p_dnn, rb_dnn = dnn_sol(net_par, d_csi, min_rate=MIN_RATE, N=net_par["N"])

            _, rates_d_dnn = cal_net_metrics(d_csi, rb_dnn, p_dnn, net_par)
            QoS_viol_d_dnn = (torch.sum(rates_d_dnn < MIN_RATE) / net_par["N"]) * 100

            _, rates_dnn = cal_net_metrics(CSI[i][0], rb_dnn, p_dnn, net_par)
            QoS_viol_dnn = (torch.sum(rates_dnn < MIN_RATE) / net_par["N"]) * 100

            print("rate for solving with distorted csi:", rates_d_dnn.mean().item())
            print("actual csi using previous sol:", rates_dnn.mean().item())

            print("QoS violation for solving with distorted csi:", QoS_viol_d_dnn.item())
            print("QoS violation using previous sol:", QoS_viol_dnn.item())

            p_gnn, rb_gnn = gnn_sol(net_par, d_csi, min_rate=MIN_RATE, evaluations=300, device="cpu")

            _, rates_d_gnn = cal_net_metrics(d_csi, rb_gnn, p_gnn, net_par)
            QoS_viol_d_gnn = (torch.sum(rates_d_gnn < MIN_RATE) / net_par["N"]) * 100

            _, rates_gnn = cal_net_metrics(CSI[i][0], rb_gnn, p_gnn, net_par)
            QoS_viol_gnn = (torch.sum(rates_gnn < MIN_RATE) / net_par["N"]) * 100

            print("rate for solving with distorted csi:", rates_d_gnn.mean().item())
            print("actual csi using previous sol:", rates_gnn.mean().item())

            print("QoS violation for solving with distorted csi:", QoS_viol_d_gnn.item())
            print("QoS violation using previous sol:", QoS_viol_gnn.item())


            p_my, rb_my = my_model_sol(net_par, d_csi, min_rate=MIN_RATE, evaluations=300, device=DEVICE)
            
            _, rates_d_my = cal_net_metrics(d_csi, rb_my, p_my, net_par)
            QoS_viol_d_my = (torch.sum(rates_d_my < MIN_RATE) / net_par["N"]) * 100

            _, rates_my = cal_net_metrics(CSI[i][0], rb_my.to("cpu"), p_my.to("cpu"), net_par)
            QoS_viol_my = (torch.sum(rates_my < MIN_RATE) / net_par["N"]) * 100

            print("rate for solving with distorted csi:", rates_d_my.mean().item())
            print("actual csi using previous sol:", rates_my.mean().item())

            print("QoS violation for solving with distorted csi:", QoS_viol_d_my.item())
            print("QoS violation using previous sol:", QoS_viol_my.item())


            iter_history_ga.append((dist_v, rates_ga.mean().item(), QoS_viol_ga.item()))
            iter_history_cnn.append((dist_v, rates_cnn.mean().item(), QoS_viol_cnn.item()))
            iter_history_dnn.append((dist_v, rates_dnn.mean().item(), QoS_viol_dnn.item()))
            iter_history_gnn.append((dist_v, rates_gnn.mean().item(), QoS_viol_gnn.item()))
            iter_history_my_model.append((dist_v, rates_my.mean().item(), QoS_viol_my.item()))

        history_ga.append(iter_history_ga)
        history_cnn.append(iter_history_cnn)
        history_dnn.append(iter_history_dnn)
        history_gnn.append(iter_history_gnn)
        history_my_model.append(iter_history_my_model)

        # print average


    for j in range(len(distortion_values)):
        print("##########  DISTORTION: {:.2f}  ##########".format(distortion_values[j]))
        mean_rate_ga = statistics.mean([history_ga[i][j][1] for i in range(simulation_count)])
        std_rate_ga = statistics.stdev([history_ga[i][j][1] for i in range(simulation_count)])

        mean_qos_viol_ga = statistics.mean([history_ga[i][j][2] for i in range(simulation_count)])
        std_qos_viol_ga = statistics.stdev([history_ga[i][j][2] for i in range(simulation_count)])

        print("ga_rate: {:.2f} +- {:.2f}".format(mean_rate_ga, std_rate_ga))
        print("ga_Qos_viol: {:.2f} +- {:.2f}".format(mean_qos_viol_ga, std_qos_viol_ga))

        mean_rate_cnn = statistics.mean([history_cnn[i][j][1] for i in range(simulation_count)])
        std_rate_cnn = statistics.stdev([history_cnn[i][j][1] for i in range(simulation_count)])

        mean_qos_viol_cnn = statistics.mean([history_cnn[i][j][2] for i in range(simulation_count)])
        std_qos_viol_cnn = statistics.stdev([history_cnn[i][j][2] for i in range(simulation_count)])

        print("cnn_rate: {:.2f} +- {:.2f}".format(mean_rate_cnn, std_rate_cnn))
        print("cnn_Qos_viol: {:.2f} +- {:.2f}".format(mean_qos_viol_cnn, std_qos_viol_cnn))

        mean_rate_dnn = statistics.mean([history_dnn[i][j][1] for i in range(simulation_count)])
        std_rate_dnn = statistics.stdev([history_dnn[i][j][1] for i in range(simulation_count)])

        mean_qos_viol_dnn = statistics.mean([history_dnn[i][j][2] for i in range(simulation_count)])
        std_qos_viol_dnn = statistics.stdev([history_dnn[i][j][2] for i in range(simulation_count)])

        print("dnn_rate: {:.2f} +- {:.2f}".format(mean_rate_dnn, std_rate_dnn))
        print("dnn_Qos_viol: {:.2f} +- {:.2f}".format(mean_qos_viol_dnn, std_qos_viol_dnn))

        mean_rate_gnn = statistics.mean([history_gnn[i][j][1] for i in range(simulation_count)])
        std_rate_gnn = statistics.stdev([history_gnn[i][j][1] for i in range(simulation_count)])

        mean_qos_viol_gnn = statistics.mean([history_gnn[i][j][2] for i in range(simulation_count)])
        std_qos_viol_gnn = statistics.stdev([history_gnn[i][j][2] for i in range(simulation_count)])

        print("gnn_rate: {:.2f} +- {:.2f}".format(mean_rate_gnn, std_rate_gnn))
        print("gnn_Qos_viol: {:.2f} +- {:.2f}".format(mean_qos_viol_gnn, std_qos_viol_gnn))

        mean_rate_my_model = statistics.mean([history_my_model[i][j][1] for i in range(simulation_count)])
        std_rate_my_model = statistics.stdev([history_my_model[i][j][1] for i in range(simulation_count)])

        mean_qos_viol_my_model = statistics.mean([history_my_model[i][j][2] for i in range(simulation_count)])
        std_qos_viol_my_model = statistics.stdev([history_my_model[i][j][2] for i in range(simulation_count)])

        print("my_model_rate: {:.2f} +- {:.2f}".format(mean_rate_my_model, std_rate_my_model))
        print("my_model_Qos_viol: {:.2f} +- {:.2f}".format(mean_qos_viol_my_model, std_qos_viol_my_model))

    print("finished...")


if __name__ == '__main__':
    
    plot_ANR_QoSpb_QoSviol(0)