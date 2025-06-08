from cProfile import label

import pandapower  as pp
import pandapower.networks as pn
import networkx as nx
import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize
from keras.src.ops import append
from scipy.integrate import quad
import matplotlib.pyplot as plt
a=0
delta=1e-4
mu=0.01
lambda_=5
v_x = lambda x:0.02*x**0.5
q_x = lambda x:np.exp(1e-6*x)
t=2000
n_operations = 100
P_p0 = 0.0069
n_local_cbs = 3
n_remote_cbs = 4
P_lk = P_rk = 0.81
P_lk_breaker_half = P_lk * 0.5
P_rk_breaker_half = P_lk_breaker_half * 0.5

def calculate_cb_failure_probability(t,n_operations):
    term1=-mu*(a + delta*t)
    term2=quad(lambda x : v_x * q_x(x), 0, t)
    term3=-(mu/(1+lambda_*mu)) *term2
    R_d=np.exp(term1-term2+term3)
    term4,_ = quad(v_x,0,t)
    term5,_=quad(lambda x : v_x(x)*q_x(x), 0, t)
    R_s = np.exp(-term4+term5)
    R_t=R_d+R_s
    P_cb = 1- R_t
    return P_cb , R_d , R_s
def plot_failure_probabilities():
    t_values = np.linspace(0)
    n_op_values = [50 , 100,150]
    degradation_probs = []
    operation_prob = []
    for n_op in n_op_values:
        deg_probs =[]
        op_probs =[]
        for t in t_values:
            _ , R_d , R_s = calculate_cb_failure_probability(t,n_op)
            deg_probs.append(1-R_d)
            op_probs.append(1-R_s)
        degradation_probs.append(deg_probs)
        operation_prob.append(op_probs)
    plt.figure(figsize=(10,5))
    for i , n_op in enumerate(n_op_values):
        plt.plot(t_values,operation_prob[i],label=f'n_operations={n_op}')
    plt.title('fig 12 probability of disturbance')
    plt.xlabel('time')
    plt.ylabel('probability')
    plt.legend()
    plt.show()
plot_failure_probabilities()
net_single_bus= pp.create_empty_network()
net_breaker_half = pp.create_empty_network()
for i in range(9):
    pp.create_empty_network(net_single_bus, vn_kv = 1 , name=f'Bus_{i+1}')
    pp.create_bus(net_breaker_half,vn_kv=1 , name=f'BBaH_{i+1}')
pp.create_gen(net_single_bus , bus=0 , p_mw = 80 , name='G2' , max_mw = 100)
pp.create_gen(net_single_bus, bus = 2 , p_mw=70 , name='G3' , max_mw = 90)
pp.create_gen(net_breaker_half,bus=0 , p_mw = 80 , max_mw = 100 , name = 'G2')
pp.create_gen(net_breaker_half, bus=2 , p_mw=70 , max_mw = 90 , name='G3')
pp.create_load(net_single_bus, bus=4 , p_mw = 50, name='Load_4')
pp.create_load(net_single_bus, bus=6 , p_mw=40, name='Load_6')
pp.create_load(net_single_bus , bus=8 , p_mw=30 , name='Load_8')
pp.create_load(net_breaker_half , bus=4 , p_mw=50 , name='Load_4')
pp.create_load(net_breaker_half, bus=6 , p_mw=40, name='Load_6')
pp.create_load(net_breaker_half , bus=8 , p_mw=30 , name='Load_8')
lines = [
    {'from_bus': 0, 'to_bus': 1, 'length_km': 1, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'name': 'Line_1', 'relay': 'overcurrent'},
    {'from_bus': 1, 'to_bus': 2, 'length_km': 1, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'name': 'Line_2', 'relay': 'overcurrent'},
    {'from_bus': 2, 'to_bus': 3, 'length_km': 1, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'name': 'Line_3', 'relay': 'overcurrent'},
    {'from_bus': 3, 'to_bus': 4, 'length_km': 1, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'name': 'Line_4', 'relay': 'directional'},
    {'from_bus': 4, 'to_bus': 5, 'length_km': 1, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'name': 'Line_5', 'relay': 'directional'},
    {'from_bus': 5, 'to_bus': 6, 'length_km': 1, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'name': 'Line_6', 'relay': 'directional'}
]
for line in lines:
    pp.create_line_from_parameters(net_single_bus, **line)
    pp.create_line_from_parameters(net_breaker_half, **line)

def check_connectivity(net):
    G= nx.Graph()
    for _, line in net.line.iterrows():
        if line.in_service:
            G.add_edge(line.from_bus,line.to_bus)
        return nx.is_connected(G)
def run_opf(net):
    try:
        pp.runopp(net,verbose=False)
        return True,0
    except:
        return False,0
def load_curtailment(net):
    load_loss=0
    original_load = net.load.p_mw.copy()
    while True:
        try:
            pp.runopp(net,verbose=False)
            return load_loss
        except:
            for i in net.load.index:
                net.load.p_mw[i] *=0.95
            load_loss+=sum(original_load - net.load.p_mw)
        return load_loss
def analyze_topology(net,line_id):
    net.line.in_service[line_id]=False
    if not check_connectivity(net):
        net.line.in_service[line_id] = True
        return float('inf')
    success , load_loss = run_opf(net)
    if not success:
        load_loss = load_curtailment(net)
    net.line.in_service[line_id] = True
    return load_loss
def calculate_risk(net,line_id,P_p0 , P_lk , P_rk,config='Single Bus'):
    results=[]
    LS_G=125
    P_local_success = (1 - P_lk) ** n_local_cbs
    P_local_failure = 1 - P_local_success
    P_remote_success = (1 - P_rk) ** n_remote_cbs
    P_remote_failure = 1 - P_remote_success
    P_G1 = (1 - P_p0)*P_local_success*P_remote_success
    L1 = P_G1 * LS_G
    results.append({'Combination' : '1 (Normal)' , 'Probability' : P_G1 , 'Load Loss(MW)' : LS_G , 'Risk (MW)' : L1})

    P_Gi = (1-P_p0)*P_local_failure*P_remote_success
    L2 = P_Gi * LS_Gi
    results.append({'Combination': '2 (Local CB Failure)' , 'Probability':P_Gi,'Load Loss(MW)':LS_G,'Risk (MW)':L2})

    P_Gj = (1-P_p0)*P_local_success*P_remote_failure
    L3 = P_Gj * LS_G
    results.append({'Combination' : '3 (Remote CB Failure)' , 'Probability' : P_Gj , 'Load_loss(MW)' : LS_G,'Risk (Mw)':L3})

    P_G0 = P_p0
    L4 = P_G0 * LS_G
    results.append({'Combination' : '4 (Relay Failure)' , 'Probability':P_G0 , 'Load_loss(MW)' : LS_G, 'Risk (MWW)':L4})

    L_total = L1 + L2 + L3 + L4
    results.append({'Combination': 'Total','Probability' : None , 'Load_loss(MW)' : None , 'Risk' : L_total})
    return results,L_total
scenarios = [
    {'line_id': 0, 'table': 'IV', 'config': 'Single Bus', 'fault': 'Fault 1', 'net': net_single_bus, 'P_lk': P_lk, 'P_rk': P_rk},
    {'line_id': 1, 'table': 'V', 'config': 'Single Bus', 'fault': 'Fault 2', 'net': net_single_bus, 'P_lk': P_lk, 'P_rk': P_rk},
    {'line_id': 0, 'table': 'VI', 'config': 'Breaker-and-a-Half', 'fault': 'Fault 1', 'net': net_breaker_half, 'P_lk': P_lk_breaker_half, 'P_rk': P_rk_breaker_half},
    {'line_id': 1, 'table': 'VII', 'config': 'Breaker-and-a-Half', 'fault': 'Fault 2', 'net': net_breaker_half, 'P_lk': P_lk_breaker_half, 'P_rk': P_rk_breaker_half}
]
all_results = []
table_viii_data = []
for scenario in scenarios:
    results, L_total = calculate_risk(
        scenario['net'], scenario['line_id'], P_p0, scenario['P_lk'], scenario['P_rk'], scenario['config'], scenario['fault']
    )
    for res in results:
        res['Table'] = scenario['table']
        res['Config'] = scenario['config']
        res['Fault'] = scenario['fault']
    all_results.extend(results)
    table_viii_data.append({
        'Table': scenario['table'],
        'Config': scenario['config'],
        'Fault': scenario['fault'],
        'Total Risk (MW)': L_total
    })
df_results = pd.DataFrame(all_results)
print("\nجداول IV-VII:")
for table in ['IV', 'V', 'VI', 'VII']:
    print(f"\nجدول {table}:")
    print(df_results[df_results['Table'] == table][['Fault', 'Config', 'Combination', 'Probability', 'Load Loss (MW)', 'Risk (MW)']])

# ------------------- تولید جدول VIII -------------------
df_table_viii = pd.DataFrame(table_viii_data)
print("\nجدول VIII:")
print(df_table_viii[['Table', 'Fault', 'Config', 'Total Risk (MW)']])




