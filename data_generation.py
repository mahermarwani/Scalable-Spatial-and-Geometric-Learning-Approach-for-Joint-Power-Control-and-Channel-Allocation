import os
import csv
import numpy as np
from tqdm import tqdm
from wireless_config.WirelessNetwork import WirelessNetwork, cal_net_metrics
from comparision_solutions.GA_solver import MINLP, GA_solver


def GA_data_generator(net_par, length, max_power, min_rate, dataset_name=None):
    if dataset_name is None:
        dataset_name = 'data_' + str(length) + 'p_max_' + str(max_power)

    path = os.path.join('data/', dataset_name)

    # Create folders
    path_csi = os.path.join(path, 'csi')
    path_p = os.path.join(path, 'p')
    path_rb = os.path.join(path, 'rb')
    path_csv = os.path.join(path, 'samples_list.csv')

    if os.path.exists(path):
        print("dataset folder name already exist...!")
        return None
    else:
        os.makedirs(path_csi)
        os.makedirs(path_p)
        os.makedirs(path_rb)
        with open(path_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["csi ID", "power ID", "rb ID"])
        file.close()

    K = net_par["K"]
    N = net_par["N"]

    for i in tqdm(range(length)):
        network = WirelessNetwork(net_par)
        p, rb = GA_solver(network.csi, net_par, eval=20000, min_rate=min_rate)

        np.save(os.path.join(path_csi, "csi_" + str(i) + ".npy"), network.csi)
        np.save(os.path.join(path_p, "p_" + str(i) + ".npy"), p)
        np.save(os.path.join(path_rb, "rb_" + str(i) + ".npy"), rb)
        with open(path_csv, 'a', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(["csi_" + str(i), "p_" + str(i), "rb_" + str(i)])
        file.close()


if __name__ == "__main__":
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
               "N": 40,
               "K": 5}
    

    LENGTH = 1000
    MAX_POWER = 1
    MIN_RATE = 100
    DATASET_NAME = "train_data_length_" + str(LENGTH) + "_max_power_" + str(MAX_POWER) + "_min_rate_" + str(MIN_RATE) + "_N_" + str(net_par["N"]) + "_K_" + str(net_par["K"])

    GA_data_generator(  net_par, 
                        length=LENGTH,
                        max_power=MAX_POWER,
                        min_rate=MIN_RATE,
                        dataset_name= DATASET_NAME)
