import os
import sys
import torch
import numpy as np
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.optimize import minimize
# Add the parent directory to the system path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wireless_config.WirelessNetwork import WirelessNetwork, cal_net_metrics





class MINLP(ElementwiseProblem):

    def __init__(self, csi, min_rate, net_par, **kwargs):
        self.net_par = net_par
        self.csi = csi
        self.min_rate = min_rate

        vars = {}

        for i in range(net_par["N"]):
            vars["rb_{}".format(i)] = Integer(bounds=(0, net_par["K"] - 1))
            vars["p_{}".format(i)] = Real(bounds=(0, 1))

        super().__init__(vars=vars, n_obj=1, n_ieq_constr=net_par["N"], **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        p = []
        rb = []
        for key, item in X.items():
            if key[0] == "r":
                rb.append(item)
            elif key[0] == "p":
                p.append(item)


        rb = torch.eye(self.net_par["K"])[torch.tensor(np.reshape(rb, self.net_par['N'])).long()].float()
        p = torch.tensor(np.reshape(p, self.net_par['N']) * 1.0).float()


        EEs, rates = cal_net_metrics(self.csi, rb, p, self.net_par)

        
        out["G"] = [self.min_rate - rates[i].item() for i in range(self.net_par["N"])]
        out["F"] = [ - rates.mean().item()]


def GA_solver(csi, net_par, eval=2000, min_rate=300):

    problem = MINLP(csi, min_rate, net_par)

    algorithm = MixedVariableGA(pop=1000, survival=RankAndCrowdingSurvival())

    res = minimize(problem, algorithm, termination=('n_evals', eval), verbose=False)

    if res.F is None:
        print("No solution has been found, Therefore print the one that has the least violations")

        best_violations = float('inf')
        best_EEs, best_p, best_rb = None, None, None

        for i in range(len(res.pop)):
            p, rb = [], []
            for key, item in res.pop[i].X.items():
                if key[0] == "r":
                    rb.append(item)
                else:
                    p.append(item)


            rb = torch.eye(net_par["K"])[torch.tensor(np.reshape(rb, net_par['N'])).long()].float()
            p = torch.tensor(np.reshape(p, net_par['N']) * 1.0).float()


            EEs, rates = cal_net_metrics(csi, rb, p, net_par)

            out = dict()
            out["F"], out["G"] = 0, 0
            problem._evaluate(res.pop[i].X, out)

            curr_violations = len([num for num in out["G"] if num > 0])
            if i == 0:
                best_EEs, best_violations, best_p, best_rb = EEs, curr_violations, p, rb
            else:
                if curr_violations < best_violations and EEs.max() > best_EEs.max():
                        best_EEs, best_violations, best_p, best_rb = EEs, curr_violations, p, rb


            return best_p, best_rb

    else:
        print("Solution has been found")
        p, rb = [], []
        for key, item in res.X.items():
            if key[0] == "r":
                rb.append(item)
            elif key[0] == "p":
                p.append(item)

        rb = torch.eye(net_par["K"])[torch.tensor(np.reshape(rb, net_par['N'])).long()].float()
        p = torch.tensor(np.reshape(p, net_par['N']) * 1.0).float()

        return p, rb



if __name__ == '__main__':
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
               "N": 300,
               "K": 5}

    network = WirelessNetwork(net_par)

    p, rb = GA_solver(network.csi, net_par, eval=20000, min_rate=100)

    EEs, rates = cal_net_metrics(network.csi, rb, p, net_par)

    print("p: ", p)
    print("rb: ", torch.where(rb != 0)[1])

    print("EEs: ", EEs)
    print("EEs mean: ", EEs.mean())
    print("rates: ", rates)
    print("rates mean: ", rates.mean())
    print("violations: ", len([num for num in rates if num < 100]))


