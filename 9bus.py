

import pandapower  as pp
import pandapower.networks as pn
import networkx as nx
import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.integrate import quad
import matplotlib.pyplot as plt
a=0
delta=1e-4
mu=0.01
lambda_=5
omega=1/5000
v_x = lambda x: 0.02 * np.sqrt(x)
q_x = lambda x: np.exp(-lam * x)
lam = 1e-6
t=2000
n_operations = 100
P_p0 = 0.0069
n_local_cbs = 3
n_remote_cbs = 4
P_lk = P_rk = 0.81
P_lk_breaker_half = P_lk * 0.5
P_rk_breaker_half = P_lk_breaker_half * 0.5


def calculate_cb_survivals(t,N):
    """
    Returns:
      R_d(t) = degradation‐based survival, eq. (7)
      R_s(t) = operation‐time survival,   eq. (8)
    """
    # ∫0^t v(x) dx
    I_v,  _ = quad(v_x, 0, t)
    # ∫0^t v(x) q(x) dx
    I_vq, _ = quad(lambda x: v_x(x) * q_x(x), 0, t)

    # eq. (7)
    R_d = np.exp(
        - mu * (a + delta * t)
        - I_vq
        - (mu / (1 + lambda_ * mu)) * I_vq
    )
    # eq. (8)
    R_s = np.exp(
        - I_v
        + I_vq
    )
    return R_d, R_s

n_op_values = [50, 150]    # e.g. 50 and 150 operations for Fig.12
t_degs      = [1000, 2000]  # e.g. 1000 and 2000 days for Fig.13

def plot_failure_probabilities():
    t_vals = np.linspace(0, 3000, 300)
    N_list = [150]

    # Fig.12 – Degradation failure vs days
    plt.figure(figsize=(8,4))
    for N in N_list:
        P_deg = np.array([calculate_cb_survivals(t, N)[0] for t in t_vals])
        plt.plot(t_vals, P_deg, label=f'N = {N}')
    plt.title('Fig.12 – Degradation failure prob. vs days')
    plt.xlabel('Time (days)')
    plt.ylabel('P_deg')
    plt.legend()
    plt.grid(True)

    # Zoom into the first 200 days to see the separation

    plt.show()
    N_values = np.arange(0, 301, 1)  # 0…300 operations
    t_degs = [1000, 2000, 3000]  # fixed days

    plt.figure(figsize=(8, 5))
    for t0 in t_degs:
        # q_x must be defined as: q_x = lambda x: np.exp(-1e-6 * x)
        P_op = np.array([q_x(t0) ** N for N in N_values])
        plt.plot(N_values, P_op, label=f't = {t0} days')
    plt.title('Fig.13 – Operation Failure Probability vs. # operations')
    plt.xlabel('Number of operations')
    plt.ylabel('P_op ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_9bus_network():
    net = pp.create_empty_network()
    for i in range(9):
        pp.create_bus(net, vn_kv=138, name=f'Bus_{i + 1}')
    pp.create_gen(net, bus=0, p_mw=80, vm_pu=1.0, name='G2', slack=True, max_p_mw=100, min_p_mw=0)
    pp.create_gen(net, bus=1, p_mw=70, vm_pu=1.0, name='G3', max_p_mw=90, min_p_mw=0)
    pp.create_load(net, bus=2, p_mw=50, name='Load_1')
    pp.create_load(net, bus=4, p_mw=40, name='Load_2')
    pp.create_load(net, bus=6, p_mw=30, name='Load_3')
    lines = [
        {'from_bus': 0, 'to_bus': 2, 'length_km': 100, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'c_nf_per_km': 0.1,
         'max_i_ka': 1, 'name': 'Line_1', 'relay': 'overcurrent'},
        {'from_bus': 2, 'to_bus': 3, 'length_km': 100, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'c_nf_per_km': 0.1,
         'max_i_ka': 1, 'name': 'Line_2', 'relay': 'overcurrent'},
        {'from_bus': 3, 'to_bus': 4, 'length_km': 100, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'c_nf_per_km': 0.1,
         'max_i_ka': 1, 'name': 'Line_3', 'relay': 'overcurrent'},
        {'from_bus': 4, 'to_bus': 5, 'length_km': 100, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'c_nf_per_km': 0.1,
         'max_i_ka': 1, 'name': 'Line_4', 'relay': 'directional'},
        {'from_bus': 5, 'to_bus': 6, 'length_km': 100, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'c_nf_per_km': 0.1,
         'max_i_ka': 1, 'name': 'Line_5', 'relay': 'directional'},
        {'from_bus': 6, 'to_bus': 1, 'length_km': 100, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'c_nf_per_km': 0.1,
         'max_i_ka': 1, 'name': 'Line_6', 'relay': 'directional'}
    ]
    for line in lines:
        pp.create_line_from_parameters(net, **{k: v for k, v in line.items() if k not in ['name', 'relay']},
                                       name=line['name'])

    return net
net_single_bus = create_9bus_network()
net_breaker_half = create_9bus_network()
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
    load_loss = 0
    original_loads = net.load.p_mw.copy()
    max_curtailment = 0.5
    curtailment_step = 0.05
    current_curtailment = 0
    while current_curtailment <= max_curtailment:
        try:
            pp.runopp(net, verbose=False)
            return load_loss
        except:
            current_curtailment += curtailment_step
            net.load.p_mw = original_loads * (1 - current_curtailment)
            load_loss = sum(original_loads - net.load.p_mw)
    print(f"Max {max_curtailment*100}% curtailment reached, solution still infeasible.")
    return float('nan')
def analyze_topology(net, line_id):
    net.line.in_service[line_id] = False
    if not check_connectivity(net):
        net.line.in_service[line_id] = True
        return float('inf')
    success, load_loss = run_opf(net)
    if not success:
        load_loss = load_curtailment(net)
    net.line.in_service[line_id] = True
    return 125
def calculate_risk(net, line_id, P_p0, P_lk, P_rk, config='Single Bus', fault='Fault 1'):
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
    L2 = P_Gi * LS_G
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
    print(df_results[df_results['Table'] == table][['Fault', 'Config', 'Combination', 'Probability', 'Risk (MW)']])

# ------------------- تولید جدول VIII -------------------
df_table_viii = pd.DataFrame(table_viii_data)
print("\nجدول VIII:")
print(df_table_viii[['Table', 'Fault', 'Config', 'Total Risk (MW)']])

# رسم شکل ۱۲ و ۱۳
plot_failure_probabilities()




