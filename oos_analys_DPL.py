import numpy as np
import pandas as pd
from gurobipy import *
import gurobipy as gp
import pprint
import matplotlib.pyplot as plt
import math


class OOS_analys:
    def __init__(self, h, b, I_0, B_0, input_data_file):
        #read excel file sheet
        data_price = pd.read_excel(input_data_file, sheet_name = 'price')
        data_supplier = pd.read_excel(input_data_file, sheet_name = 'supplier') 
        data_capacity = pd.read_excel(input_data_file, sheet_name = 'capacity')
        
        #initiate self parameters
        self.h = h
        self.b = b
        self.I_0 = I_0
        self.B_0 = B_0
        
        self.time = data_capacity['time'].values
        self.supplier, self.order_cost, self.lead_time, self.quality_level = self.get_suppliers(data_supplier)
        #self.prices, 
        self.capacities = self.get_time_suppliers(data_price,data_capacity)
        
        
    def get_suppliers(self, data_supplier):
        supplier = data_supplier['supplier'].values
        order_cost = data_supplier['order_cost'].values
        lead_time = data_supplier['lead_time'].values
        quality_level = data_supplier['quality_level'].values
        multi_temp = {}
        for i in range(len(supplier)):
            temp = [ order_cost[i], lead_time[i],quality_level[i]]
            multi_temp[supplier[i]] = temp
        return multidict(multi_temp)
    
    def get_time_suppliers(self, data_price, data_capacity):
        price_sn,  capacity_sn = [0], [0]
        for i in range(1, len(self.supplier)+1):
            price_sn.append(data_price['s'+str(i)].values)
            capacity_sn.append(data_capacity['s'+str(i)].values)
        prices = tupledict()
        capacities = tupledict()
        for t in self.time:
            for s in self.supplier:
                n = int(s[1:])
                prices[(t,s)] = price_sn[n][t]
                capacities[(t,s)] = capacity_sn[n][t]
        return prices, capacities
        
#     def plot_solution (self, key, single_solution):
#         #draw pictures
#         pd.options.mode.chained_assignment = None  # default='warn'
#         df = single_solution[single_solution['variable_name'].str.contains('order_quantity')]
#         # Parse the 'Variable Name' column to extract S0, S1, S2, and S3
#         df['Parsed'] = df['variable_name'].apply(lambda x: x.split('[')[1].replace(']', '').split(','))
#         df['order_time'] = df['Parsed'].apply(lambda x: int(x[0]))
#         df['S_no'] = df['Parsed'].apply(lambda x: x[1])
#         df['S_no'] = df['S_no'].astype(str)

#         grouped_df = df.groupby(['order_time', 'S_no'], as_index=False).agg({'value': 'sum'})
#         fig, ax = plt.subplots()
#         for s_no in grouped_df['S_no'].unique():
#             subset = grouped_df[grouped_df['S_no'] == s_no]
#             ax.plot(subset['order_time'], subset['value'], label=f'S_no: {s_no}', marker = 'o')
#         ax.set_xlabel('Order Time')
#         ax.set_ylabel('Order Quantity')
#         ax.legend()
#         ax.set_title(f'Order Quantity_input_dist={key[0]}_input_sizes={key[1]}_models={key[2]}_out_dist={key[3]}_obj={math.ceil(key[4])}')
#         plt.show()
        
    def plot_solution(self, key, single_solution):
        pd.options.mode.chained_assignment = None  # default='warn'
        df = single_solution[single_solution['variable_name'].str.contains('order_quantity')]

        df['Parsed'] = df['variable_name'].apply(lambda x: x.split('[')[1].replace(']', '').split(','))
        df['order_time'] = df['Parsed'].apply(lambda x: int(x[0]))
        df['S_no'] = df['Parsed'].apply(lambda x: x[1])
        df['S_no'] = df['S_no'].astype(str)

        grouped_df = df.groupby(['order_time', 'S_no'], as_index=False).agg({'value': 'sum'})

        total_order_quantity = grouped_df.groupby('S_no')['value'].sum()

        fig, ax = plt.subplots()
        for s_no in grouped_df['S_no'].unique():
            subset = grouped_df[grouped_df['S_no'] == s_no]
            total_quantity = total_order_quantity[s_no]
            ax.plot(subset['order_time'], subset['value'], label=f'S_no: {s_no} (Total: {int(total_quantity)})', marker='o')
            for _, row in subset.iterrows():
                ax.annotate(f"{int(row['value'])}", (row['order_time'], row['value']),
                            textcoords="offset points", xytext=(0, 5), ha='center')

        ax.set_xlabel('Order Time')
        ax.set_ylabel('Order Quantity')
        ax.legend()
        ax.set_title(f'Order Quantity_input_dist={key[0]}_input_sizes={key[1]}_models={key[2]}_out_dist={key[3]}_obj={math.ceil(key[4])}_price={key[5]}')
        plt.show()


    def cal_procurment_cost (self, filtered_order_quantity, prices):
        #print(filtered_order_quantity)
        filtered_order_quantity['order_time'] = filtered_order_quantity['variable_name'].apply(lambda x: int(x.split('[')[1].split(',')[0]))
        filtered_order_quantity['supplier'] = filtered_order_quantity['variable_name'].apply(lambda x: x.split(',')[1].split(']')[0])
        order_time_supplier = filtered_order_quantity.groupby(['order_time', 'supplier']).sum()['value']
        # print( order_time_supplier )
        fixed_order_cost = sum((self.order_cost).values()) / len(self.order_cost) * (filtered_order_quantity[filtered_order_quantity['value'] >= 1].shape[0])
        # print(filtered_order_quantity)
        purchase_cost = 0
        for s in self.supplier:
            for t in self.time:
                purchase_cost += order_time_supplier[t][s] * prices[s][t]
        #print(purchase_cost)
        return fixed_order_cost, purchase_cost

    def cal_inv_backlog (self, filtered_theta, demand, leadtime):
        # print('0',filtered_theta)
        filtered_theta['arrive_time'] = filtered_theta['variable_name'].apply(lambda x: int(x.split('[')[1].split(',')[0])+ leadtime[x.split(',')[1].split(']')[0]][0])
        # print('1',filtered_theta)
        # 1. 将 arrive_time > 7 的值改为 0
        filtered_theta.loc[filtered_theta['arrive_time'] > 7, 'arrive_time'] = 0
        
        # 2. 对 arrive_time 为 0 的行，将 value 也改为 0
        filtered_theta.loc[filtered_theta['arrive_time'] == 0, 'value'] = 0

        # 确保 arrive_time 只有 0-7 之间的数
        existing_times = set(filtered_theta['arrive_time'].unique())  # 已有的 arrive_time
        all_times = set(range(8))  # 期望的 arrive_time (0,1,2,3,4,5,6,7)
        
        # 找出缺失的 arrive_time
        missing_times = all_times - existing_times  
        
        # 如果有缺失的时间点，补充进 DataFrame
        if missing_times:
            missing_rows = pd.DataFrame({'arrive_time': list(missing_times), 'value': 0})
            filtered_theta = pd.concat([filtered_theta, missing_rows], ignore_index=True)
        
        # 按 arrive_time 排序
        filtered_theta = filtered_theta.sort_values(by='arrive_time').reset_index(drop=True)

        # print(filtered_theta['variable_name'])
        # print('2', filtered_theta)
        arrive_quantity = filtered_theta.groupby('arrive_time').sum()['value']

        inventory =  [0] * (len(demand))
        backlog =  [0] * (len(demand))
        #print('arrive:', arrive_quantity)
        for i in range(0, len(demand)):
            
            if i == 0 :
                temp = self.I_0 - self.B_0 + arrive_quantity.loc[i] - demand[i]
                if  temp > 0:
                    inventory[i] = temp
                else:
                    backlog[i] = - temp
            else:
                temp = inventory[i-1] - backlog[i-1] + arrive_quantity.loc[i] - demand[i]
                if temp >= 0:
                    inventory[i] = temp
                else:
                    backlog[i] = -temp

        inventory_cost = self.h * sum(inventory)
        backlog_cost = self.b * sum(backlog)
        df_detail = pd.DataFrame({'order_time':self.time, 'demand':demand, 'inventory': inventory, 'backlog': backlog})
        # print(inventory_cost)
        # print(backlog_cost)
        # print(df_detail)
        # print(df_detail['backlog'].sum())
        return inventory_cost, backlog_cost, df_detail

    #def cal_out_of_sample(self, solution, oos_demands, oos_prices):
    def cal_out_of_sample(self, solution, oos_demands, oos_prices, oos_leadtimes):
        df_oos_cost = []
        df_details= []
        #calculate procurment cost
        filtered_order_quantity = solution[solution['variable_name'].str.contains('order_quantity')]
        #print('0', filtered_order_quantity)
        # fixed_order_cost, purchase_cost = self.cal_procurment_cost(filtered_order_quantity)
        #calculate inv and backlog cost
        filtered_theta = solution[solution['variable_name'].str.contains('arrive_quantity')]
        #print(filtered_order_quantity)

        for i in range(oos_demands.shape[1]):
            demand = oos_demands[i].values.tolist()
            # price = oos_prices[i]
            price = {key: df.iloc[:, i].tolist() for key, df in oos_prices.items()}
            leadtime = {key: df.iloc[:, i].tolist() for key, df in oos_leadtimes.items()}
            # print(price)
            # print(price['s1'][0])
            inventory_cost_single, backlog_cost_single, df_detail= self.cal_inv_backlog(filtered_order_quantity, demand, leadtime)
            fixed_order_cost, purchase_cost = self.cal_procurment_cost(filtered_order_quantity, price)
            df_detail['out_sample_no'] = i
            df_details.append(df_detail)
            df_oos_cost.append([i, fixed_order_cost, purchase_cost, inventory_cost_single, backlog_cost_single,(fixed_order_cost + purchase_cost + inventory_cost_single + backlog_cost_single)])

        df_oos_cost = pd.DataFrame(df_oos_cost, columns=['out_sample_no', 'fixed_order_cost', 'purchase_cost','inv_cost','backlog_cost','total_cost'])
        # print(df_oos_cost)
        df_details = pd.concat(df_details)
        return df_oos_cost,df_details