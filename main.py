# # DRO Price Uncertainty
# ## fixed demand and leadtime
import numpy as np
import pandas as pd
import pprint
import math
from gurobipy import *
import gurobipy as gp
import cvxpy as cp
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from oos_analys_DPL import OOS_analys
import warnings
import time
# Ignore all warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import warnings
from scipy.special import gamma
from scipy.stats import gamma
from scipy.optimize import fsolve
from scipy.special import gamma as gamma_func
from itertools import product
import torch




input_parameters_file = 'input_parameters.xlsx'
input_demand_file = 'input_demand_diff.xlsx'
input_leadtime_file = 'input_leadtimes_same.xlsx'
input_price_file = 'input_prices_difflarger_std.xlsx'

block_size = 48
T= 12

h = 5 # inventory holding cost
b = 300   # backlog cost
I_0 = 1800    # starting inventory level
B_0 = 0    
R = 0

k_fold = 3
oos_size = 100
planning_horizon = T
rho = 0 # co-variance in demand

input_demand_mean = 1807
#input_demand_std = 488
input_demand_std = 0.01   # fix demand equal to mean = 1807
input_price_mean = {'s1':1.2, 's2':1.3}
input_price_std = {'s1':4, 's2':3}  
seed = 25



########################==============================models for out sample generation==============================########################  
def get_gaussian_demand(time_horizon, mean, std_dev, rho, M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mean_vector = np.full((time_horizon,), mean)
    covariance_matrix = np.diag(np.full((time_horizon,), std_dev**2))
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, M)
    
    cov_value = rho * std_dev**2
    covariance_matrix = np.diag(np.full((time_horizon,), std_dev**2))

    for j in range(time_horizon - 1):
        covariance_matrix[j, j+1] = covariance_matrix[j+1, j] = cov_value
    
    samples_with_correlation = np.random.multivariate_normal(mean_vector, covariance_matrix, M)
    samples_with_correlation = np.round(samples_with_correlation).astype(int)
    samples_df = pd.DataFrame(samples_with_correlation.T)
    
    warnings.filterwarnings("ignore", message="covariance is not symmetric positive-semidefinite")

    return samples_df

def get_gamma_demand(time_horizon, mean, std_dev, M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Number of random variables to generate
    k = mean**2 / std_dev**2
    theta = std_dev**2 / mean
    m = time_horizon * M
    samples = np.random.gamma(shape=k, scale=theta, size=m)
    samples = np.round(samples).astype(int)
    reshaped_array = samples.reshape((time_horizon, M))
    samples_df = pd.DataFrame(reshaped_array)

    return samples_df

def get_lognormal_demand(time_horizon, mean, std_dev, rho, M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    normal_mean = np.log(mean**2 / np.sqrt(std_dev**2 + mean**2))
    normal_std_dev = np.sqrt(np.log(1 + (std_dev**2 / mean**2)))

    mean_vector = np.full((time_horizon,), normal_mean)
    covariance_matrix = np.diag(np.full((time_horizon,), normal_std_dev**2))
    cov_value = rho * normal_std_dev**2

    for j in range(time_horizon - 1):
        covariance_matrix[j, j+1] = covariance_matrix[j+1, j] = cov_value

    warnings.filterwarnings("ignore", message="covariance is not symmetric positive-semidefinite")
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, M)

    lognormal_samples = np.exp(samples)
    lognormal_samples = np.round(lognormal_samples).astype(int)
    samples_df = pd.DataFrame(lognormal_samples.T)

    return samples_df


def get_weibull_demand(time_horizon, mean, std_dev, M, seed=None):
    # 计算形状参数 k 和尺度参数 lambda
    # 使用公式：mean = lambda * Gamma(1 + 1/k) 和 std_dev = lambda * sqrt(Gamma(1 + 2/k) - Gamma(1 + 1/k)^2)
    # 解方程求解 k 和 lambda
    if seed is not None:
        np.random.seed(seed)
    def weibull_params(mean, std_dev):
        from scipy.optimize import fsolve
        def equations(vars):
            k, lambda_ = vars
            mean_eq = lambda_ * gamma(1 + 1/k) - mean
            std_dev_eq = lambda_ * np.sqrt(gamma(1 + 2/k) - gamma(1 + 1/k)**2) - std_dev
            return [mean_eq, std_dev_eq]
        
        k_guess, lambda_guess = 0.5, mean  # 初始猜测值
        k, lambda_ = fsolve(equations, (k_guess, lambda_guess))
        return k, lambda_

    k, lambda_ = weibull_params(mean, std_dev)
    samples = np.random.weibull(k, size=(time_horizon, M)) * lambda_    # 生成韦伯分布的随机样本
    samples = np.round(samples).astype(int)    # 将样本四舍五入为整数
    samples_df = pd.DataFrame(samples) # 转换为 DataFrame

    return samples_df


def get_leadtime(M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # fix leadtime
    values_s1 = [1, 2, 3]
    prob_s1 = [0, 0, 1]  
    values_s2 = [1, 2, 3]
    prob_s2 = [0, 0, 1]
    # values_s3 = [2, 3, 4]
    # prob_s3 = [0, 1, 0]  

    all_combinations = np.array(list(product(values_s1, values_s2)))
    valid_combinations = all_combinations[all_combinations.sum(axis=1) <= 6]

    selected_samples = np.zeros((T, M, 2), dtype=int)
    for i in range(T):
        for j in range(M):
            s1 = np.random.choice(values_s1, p=prob_s1)
            s2 = np.random.choice(values_s2, p=prob_s2)
            # s3 = np.random.choice(values_s3, p=prob_s3)
                #if s1 + s2 + s3 <= 6:
            selected_samples[i, j:] = [s1, s2]


    result_dict = {
        f's{i+1}': pd.DataFrame(selected_samples[:, :, i],  # 取第 i 列
                                columns=range(M), 
                                index=pd.Index(range(T), name="time"))
        for i in range(2)
    }

    return result_dict

# ============================== Gaussian ==============================
def get_gaussian_demand(time_horizon, mean, std_dev, rho, M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mean_vector = np.full((time_horizon,), mean)
    cov_value = rho * std_dev**2
    covariance_matrix = np.diag(np.full((time_horizon,), std_dev**2))

    for j in range(time_horizon - 1):
        covariance_matrix[j, j+1] = covariance_matrix[j+1, j] = cov_value

    warnings.filterwarnings("ignore", message="covariance is not symmetric positive-semidefinite")
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, M)
    samples = np.round(samples).astype(int)
    samples = np.clip(samples, 0, None)  # 设置负值为0
    samples_df = pd.DataFrame(samples.T)

    return samples_df

# ============================== Gamma ==============================
def get_gamma_demand(time_horizon, mean, std_dev, M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    k = mean**2 / std_dev**2
    theta = std_dev**2 / mean
    m = time_horizon * M
    samples = np.random.gamma(shape=k, scale=theta, size=m)
    samples = np.round(samples).astype(int)
    samples = np.clip(samples, 0, None)
    reshaped_array = samples.reshape((time_horizon, M))
    samples_df = pd.DataFrame(reshaped_array)

    return samples_df

# ============================== Lognormal ==============================
def get_lognormal_demand(time_horizon, mean, std_dev, rho, M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    normal_mean = np.log(mean**2 / np.sqrt(std_dev**2 + mean**2))
    normal_std_dev = np.sqrt(np.log(1 + (std_dev**2 / mean**2)))

    mean_vector = np.full((time_horizon,), normal_mean)
    cov_value = rho * normal_std_dev**2
    covariance_matrix = np.diag(np.full((time_horizon,), normal_std_dev**2))

    for j in range(time_horizon - 1):
        covariance_matrix[j, j+1] = covariance_matrix[j+1, j] = cov_value

    warnings.filterwarnings("ignore", message="covariance is not symmetric positive-semidefinite")
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, M)
    lognormal_samples = np.exp(samples)
    lognormal_samples = np.round(lognormal_samples).astype(int)
    lognormal_samples = np.clip(lognormal_samples, 0, None)
    samples_df = pd.DataFrame(lognormal_samples.T)

    return samples_df




def get_weibull_demand(time_horizon, mean, std_dev, M, seed=None):
    if seed is not None:
        np.random.seed(seed)


  

    def weibull_params(mean, std_dev):
        def equations(vars):
            k, lambda_ = vars
            mean_eq = lambda_ * gamma_func(1 + 1/k) - mean
            std_dev_eq = lambda_ * np.sqrt(gamma_func(1 + 2/k) - gamma_func(1 + 1/k)**2) - std_dev
            return [mean_eq, std_dev_eq]
        return fsolve(equations, (0.5, mean))

    k, lambda_ = weibull_params(mean, std_dev)
    samples = np.random.weibull(k, size=(time_horizon, M)) * lambda_
    samples = np.round(samples).astype(int)
    samples = np.clip(samples, 0, None)
    samples_df = pd.DataFrame(samples)

    return samples_df

class ModelsCvxpy:
    def __init__(self, h, b, I_0, B_0, R, input_parameters_file, dist, input_demand, N, input_price, input_leadtimes):
        data_price = pd.read_excel(input_parameters_file, sheet_name='price')
        data_supplier = pd.read_excel(input_parameters_file, sheet_name='supplier')
        data_capacity = pd.read_excel(input_parameters_file, sheet_name='capacity')

        self.h = h
        self.b = b
        self.I_0 = I_0
        self.B_0 = B_0
        self.N = N
        self.Nlist = list(range(N))
        self.R = R
        self.dist = dist
        self.demand = input_demand
        self.price = input_price
        self.leadtime = input_leadtimes

        self.time = range(input_demand.shape[0])
        self.supplier, self.order_cost, self.lead_time, self.quality_level = self.get_suppliers(data_supplier)
        self.prices, self.capacities = self.get_time_suppliers(data_price, data_capacity)

        self.t_supplier = self.get_structure(self.time, self.supplier)

    def get_structure(self, *args):
        if len(args) == 2:
            return [(a, b) for a in args[0] for b in args[1]]
        elif len(args) == 3:
            return [(a, b, c) for a in args[0] for b in args[1] for c in args[2]]
        elif len(args) == 4:
            return [(a, b, c, d) for a in args[0] for b in args[1] for c in args[2] for d in args[3]]

    def get_suppliers(self, data_supplier):
        supplier = data_supplier['supplier'].values
        order_cost = data_supplier['order_cost'].values
        lead_time = data_supplier['lead_time'].values
        quality_level = data_supplier['quality_level'].values
        supplier_dict = {supplier[i]: (order_cost[i], lead_time[i], quality_level[i]) for i in range(len(supplier))}
        return list(supplier_dict.keys()), {k: v[0] for k, v in supplier_dict.items()}, {k: v[1] for k, v in supplier_dict.items()}, {k: v[2] for k, v in supplier_dict.items()}

    def get_time_suppliers(self, data_price, data_capacity):
        price_sn, capacity_sn = [0], [0]
        for i in range(1, len(self.supplier) + 1):
            price_sn.append(data_price[f's{i}'].values)
            capacity_sn.append(data_capacity[f's{i}'].values)

        prices = {}
        capacities = {}
        for t in self.time:
            for s in self.supplier:
                n = int(s[1:])
                prices[(t, s)] = price_sn[n][t]
                capacities[(t, s)] = capacity_sn[n][t]

        return prices, capacities


    def optimize_DROprice(self, epsilon):
        Q = {(t, s): cp.Variable(nonneg=True) for t in self.time for s in self.supplier}
        Y = {(t, s): cp.Variable(boolean=True) for t in self.time for s in self.supplier}
        theta = {(t, s): cp.Variable(nonneg=True) for t in self.time for s in self.supplier}
        I = {t: cp.Variable(nonneg=True) for t in self.time}
        B = {t: cp.Variable(nonneg=True) for t in self.time}
        beta = {s: cp.Variable(nonneg=True) for s in self.supplier}
        alpha = {(s, n): cp.Variable() for s in self.supplier for n in self.Nlist}

        xi1 = {(t, s, n): cp.Variable(nonneg=True) for t in self.time for s in self.supplier for n in self.Nlist}
        xi2 = {(t, s, n): cp.Variable(nonneg=True) for t in self.time for s in self.supplier for n in self.Nlist}
        xi3 = {(s, n): cp.Variable(nonneg=True) for s in self.supplier for n in self.Nlist}

        constraints = []
        PI = self.I_0 - self.B_0
        for t in self.time:
            inflow = cp.sum([theta[t, s] for s in self.supplier])
            demand_t = 1807  # hardcoded as per original
            if t == 0:
                constraints.append(I[t] - B[t] == PI + inflow - demand_t)
            else:
                constraints.append(I[t] - B[t] == I[t - 1] - B[t - 1] + inflow - demand_t)

        for s in self.supplier:
            lead_df = self.leadtime[s]  # Ensure this is a DataFrame
            for t_prime in self.time:
                for n in self.Nlist:
                    match_terms = []
                    for t in self.time:
                        try:
                            lead = int(lead_df.iloc[t, n])
                        except Exception as e:
                            print(f"[leadtime access error] s={s}, t={t}, n={n} | {e}")
                            continue
                        if t + lead == t_prime:
                            match_terms.append(Q[t, s])
                    if match_terms:
                        constraints.append(theta[t_prime, s] == cp.sum(match_terms))


        for s in self.supplier:
            for n in self.Nlist:
                constraints.append(xi3[s, n] <= beta[s])
                price_term = cp.sum([self.price[s].iloc[t, n] * (xi1[t, s, n] - xi2[t, s, n]) for t in self.time])
                constraints.append(price_term <= alpha[s, n])
                for t in self.time:
                    constraints.append(Q[t, s] <= Y[t, s] * 1e6)
                    constraints.append(Q[t, s] <= self.capacities[(t, s)])
                    constraints.append(Q[t, s] - xi1[t, s, n] + xi2[t, s, n] <= 0)
                    constraints.append(xi1[t, s, n] + xi2[t, s, n] - xi3[s, n] <= 0)

        fixed_cost = cp.sum([self.order_cost[s] * Y[t, s] for t in self.time for s in self.supplier])
        expected_holding = cp.sum([self.h * I[t] + self.b * B[t] for t in self.time]) / self.N
        DRO_term = cp.sum([beta[s] * epsilon[s] + cp.sum([alpha[s, n] for n in self.Nlist]) / self.N for s in self.supplier])
        objective = cp.Minimize(expected_holding + fixed_cost + DRO_term)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCIPY, verbose= False)

        df_solution = pd.DataFrame([(f"order_quantity[{t},{s}]", Q[t, s].value) for t, s in Q], columns=["variable_name", "value"])
        df_theta = pd.DataFrame([(f"arrive_quantity[{t},{s}]", theta[t, s].value) for t, s in theta], columns=["variable_name", "value"])
        df_result = pd.concat([df_solution, df_theta], ignore_index=True)

        return prob.value, df_result

    def optimize_SAA(self):
        Q = {(t, s): cp.Variable(nonneg=True) for t in self.time for s in self.supplier}
        Y = {(t, s): cp.Variable(boolean=True) for t in self.time for s in self.supplier}
        theta = {(t, s, n): cp.Variable(nonneg=True) for t in self.time for s in self.supplier for n in self.Nlist}
        I = {(t, n): cp.Variable(nonneg=True) for t in self.time for n in self.Nlist}
        B = {(t, n): cp.Variable(nonneg=True) for t in self.time for n in self.Nlist}

        constraints = []
        PI = self.I_0 - self.B_0
        for n in self.Nlist:
            for t in self.time:
                inflow = cp.sum([theta[t, s, n] for s in self.supplier])
                demand_tn = self.demand.iloc[t, n]
                if t == 0:
                    constraints.append(I[t, n] - B[t, n] == PI + inflow - demand_tn)
                else:
                    constraints.append(I[t, n] - B[t, n] == I[t - 1, n] - B[t - 1, n] + inflow - demand_tn)

        for s in self.supplier:
            for t_prime in self.time:
                for n in self.Nlist:
                    match_terms = [Q[t, s] for t in self.time if t + int(self.leadtime[s].iloc[t, n]) == t_prime]
                    if match_terms:
                        constraints.append(theta[t_prime, s, n] == cp.sum(match_terms))

        for t, s in Q:
            constraints.append(Q[t, s] <= Y[t, s] * 1e6)
            constraints.append(Q[t, s] <= self.capacities[(t, s)])

        cost_expr = 0
        for n in self.Nlist:
            cost_expr += cp.sum([self.price[s].iloc[t, n] * Q[t, s] for t in self.time for s in self.supplier])
            cost_expr += cp.sum([self.h * I[t, n] + self.b * B[t, n] for t in self.time])
        cost_expr /= self.N
        cost_expr += cp.sum([self.order_cost[s] * Y[t, s] for t in self.time for s in self.supplier])

        objective = cp.Minimize(cost_expr)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCIPY, verbose= False)

        df_solution = pd.DataFrame([(f"order_quantity[{t},{s}]", Q[t, s].value) for t, s in Q], columns=["variable_name", "value"])
        df_theta = pd.DataFrame([(f"arrive_quantity[{t},{s},{n}]", theta[t, s, n].value) for t, s, n in theta], columns=["variable_name", "value"])
        df_result = pd.concat([df_solution, df_theta], ignore_index=True)

        return prob.value, df_result



# since fixed demand, demand is independent of distribution
oos_demands = get_gaussian_demand(planning_horizon, input_demand_mean, input_demand_std, rho, oos_size, seed)

oos_prices_normal, oos_prices_gamma, oos_prices_lognormal, oos_prices_weibull = {}, {}, {}, {}
oos_leadtimes = get_leadtime(oos_size)
for s in input_price_std:
    mean, std = input_price_mean[s], input_price_std[s]

    one_price_normal = get_gaussian_demand(planning_horizon, mean, std, rho, oos_size, seed)
    oos_prices_normal[s] = one_price_normal

    one_price_gamma = get_gamma_demand(planning_horizon, mean, std, oos_size, seed)
    oos_prices_gamma[s] = one_price_gamma

    one_price_lognormal = get_lognormal_demand(planning_horizon, mean, std, rho, oos_size, seed)
    oos_prices_lognormal[s] = one_price_lognormal
    
    one_price_weibull = get_weibull_demand(planning_horizon, mean, std, oos_size, seed)
    oos_prices_weibull[s] = one_price_weibull
    
oos_prices_dict = {'normal':oos_prices_normal, 'gamma':oos_prices_gamma, 'log_normal':oos_prices_lognormal, 'weibull': oos_prices_weibull}



# input_sample_no = [5,10,20,40,60,80,100,120,140,160,180,200][:]
input_sample_no = [10, 20][:]
oos_analys = OOS_analys(h, 
                        b, 
                        I_0, 
                        B_0, 
                        input_parameters_file)
all_res = {}
all_df_list = []
solution_list= []


for input_dist in ['normal', 'gamma', 'log_normal', 'weibull']:
    all_res['input_dist'] = input_dist

    for out_sample_dist in ['normal', 'gamma'][:1]:
        all_res['oos_dist'] = out_sample_dist

        for N in input_sample_no:
            all_res['N'] = N
            input_demand = pd.read_excel(
                input_demand_file, sheet_name=input_dist, index_col=0, usecols=range(N + 1)
            )

            input_prices = {}
            input_leadtimes = {}
            for s in input_price_std:
                input_prices[s] = pd.read_excel(
                    input_price_file, sheet_name=s + input_dist, index_col=0, usecols=range(N + 1), engine='openpyxl'
                )
                input_leadtimes[s] = pd.read_excel(
                    input_leadtime_file, sheet_name=s, index_col=0, usecols=range(N + 1), engine='openpyxl'
                )

            # ------------------------------- Cross-validation --------------------------------
            cross_res = {'SAA': -1, 'DROprice': {}}

            for model_name in ['DROprice']:
                min_epsilons = {}
                list_epsilon = [0, 10]

                for k in range(k_fold):
                    num_fold = N // k_fold
                    train_columns = [i for i in range(N) if not (k * num_fold <= i < (k + 1) * num_fold)]

                    train_demand = input_demand.iloc[:, train_columns]
                    train_price = {s: input_prices[s].iloc[:, train_columns] for s in input_price_std}
                    train_leadtime = {s: input_leadtimes[s].iloc[:, train_columns] for s in input_price_std}

                    solve = ModelsCvxpy(h, b, I_0, B_0, R, input_parameters_file,
                                        input_dist, train_demand, len(train_columns), train_price, train_leadtime)

                    min_cost = float('inf')
                    best_eps = {}

                    for eps_s1 in list_epsilon:
                        for eps_s2 in list_epsilon:
                            eps = {'s1': eps_s1, 's2': eps_s2}

                            try:
                                obj, solution = solve.optimize_DROprice(eps)
                            except Exception as e:
                                print(f"[CV ERROR] fold {k}, eps={eps} → {e}")
                                continue

                            # Placeholder cost function for validation step
                            tmp_cost = obj  # Replace with real cost function if needed

                            if tmp_cost < min_cost:
                                min_cost = tmp_cost
                                best_eps = eps.copy()

                    min_epsilons[k] = best_eps

                # Average optimal epsilon across folds
                keys = min_epsilons[0].keys() if min_epsilons and min_epsilons[0] else []
                cross_res[model_name] = {
                    k: sum(d[k] for d in min_epsilons.values() if k in d) / k_fold for k in keys
                }

            print(cross_res)

            if model_name in ['DROprice']:
                print(f"input_dist={input_dist}, sample sizes={N}, out_sample_dist={out_sample_dist}, model={model_name}")
                start = time.time()
                solve = ModelsCvxpy(h, b, I_0, B_0, R, input_parameters_file,
                                    input_dist, input_demand, N, input_prices, input_leadtimes)

                obj, solution_ = solve.optimize_DROprice(cross_res[model_name])

                solution_list.append(solution_)

# Extract order-related variables
order_df_ = [
    df[df['variable_name'].str.contains("order_quantity|arrive_quantity|order_decision")]
    for df in solution_list
]

order_matrix = pd.DataFrame()

for i, df in enumerate(order_df_):
    config_name = f"{all_res['input_dist']}_{all_res['oos_dist']}_N{all_res['N']}_run{i+1}"

    # Extract and align variable values by variable_name
    series = df.set_index("variable_name")["value"]

    order_matrix[config_name] = series

# Export to CSV
order_matrix.to_csv("order_matrix.csv")








