import pandapower as pp
import networkx as nx
import numpy as np
import pandas as pd

a = 0
delta = 1e-4
mu = 0.01
lambda_ = 5
v_x = lambda x: 0.02 * np.sqrt(x)
q_x = lambda x: np.exp(-1e-6 * x)
t = 2000
n_operations = 100
P_p0 = 0.0069
n_local_cbs = 3
n_remote_cbs = 4
P_lk = P_rk = 0.81
P_lk_breaker_half = P_lk * 0.5
P_rk_breaker_half = P_rk * 0.5

def create_68bus_network():
    net = pp.create_empty_network()
    for i in range(68):
        pp.create_bus(net, vn_kv=138, name=f'Bus_{i+1}')
    pp.create_gen(net, bus=0, p_mw=1000, vm_pu=1.0, name='G1', slack=True, max_p_mw=1200, min_p_mw=0)
    pp.create_gen(net, bus=10, p_mw=800, vm_pu=1.0, name='G2', max_p_mw=1000, min_p_mw=0)
    pp.create_load(net, bus=20, p_mw=500, name='Load_1')
    pp.create_load(net, bus=30, p_mw=400, name='Load_2')
    pp.create_load(net, bus=40, p_mw=300, name='Load_3')
    lines = [
        {'from_bus': 26, 'to_bus': 53, 'length_km': 100, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'c_nf_per_km': 0.1, 'max_i_ka': 1, 'name': 'Tie-Line_27-54', 'relay': 'overcurrent'},
        {'from_bus': 15, 'to_bus': 45, 'length_km': 100, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'c_nf_per_km': 0.1, 'max_i_ka': 1, 'name': 'Tie-Line_16-46', 'relay': 'overcurrent'},
        {'from_bus': 35, 'to_bus': 50, 'length_km': 100, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'c_nf_per_km': 0.1, 'max_i_ka': 1, 'name': 'Tie-Line_36-51', 'relay': 'directional'},
        {'from_bus': 25, 'to_bus': 55, 'length_km': 100, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'c_nf_per_km': 0.1, 'max_i_ka': 1, 'name': 'Tie-Line_26-56', 'relay': 'directional'},
        {'from_bus': 5, 'to_bus': 60, 'length_km': 100, 'r_ohm_per_km': 0.01, 'x_ohm_per_km': 0.1, 'c_nf_per_km': 0.1, 'max_i_ka': 1, 'name': 'Tie-Line_6-61', 'relay': 'directional'}
    ]
    for line in lines:
        pp.create_line_from_parameters(net, **{k: v for k, v in line.items() if k not in ['name', 'relay']}, name=line['name'])
    return net

net_single_bus = create_68bus_network()
net_breaker_half = create_68bus_network()

def check_connectivity(net):
    G = nx.Graph()
    for _, line in net.line.iterrows():
        if line.in_service:
            G.add_edge(line.from_bus, line.to_bus)
    return nx.is_connected(G)

def run_opf(net):
    try:
        pp.runopp(net, verbose=False)
        return True, 0
    except:
        return False, 0

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
    return 1000

def calculate_risk(net, line_id, P_p0, P_lk, P_rk, config='Single Bus', fault='Fault'):
    results = []
    LS_G = 1000
    P_local_success = (1 - P_lk) ** n_local_cbs
    P_local_failure = 1 - P_local_success
    P_remote_success = (1 - P_rk) ** n_remote_cbs
    P_remote_failure = 1 - P_remote_success
    P_G1 = (1 - P_p0) * P_local_success * P_remote_success
    L1 = P_G1 * LS_G
    results.append({'Combination': '1 (Normal)', 'Probability': P_G1, 'Load Loss (MW)': LS_G, 'Risk (MW)': L1})
    P_Gi = (1 - P_p0) * P_local_failure * P_remote_success
    L2 = P_Gi * LS_G
    results.append({'Combination': '2 (Local CB Failure)', 'Probability': P_Gi, 'Load Loss (MW)': LS_G, 'Risk (MW)': L2})
    P_Gj = (1 - P_p0) * P_local_success * P_remote_failure
    L3 = P_Gj * LS_G
    results.append({'Combination': '3 (Remote CB Failure)', 'Probability': P_Gj, 'Load Loss (MW)': LS_G, 'Risk (MW)': L3})
    P_G0 = P_p0
    L4 = P_G0 * LS_G
    results.append({'Combination': '4 (Relay Failure)', 'Probability': P_G0, 'Load Loss (MW)': LS_G, 'Risk (MW)': L4})
    L_total = L1 + L2 + L3 + L4
    results.append({'Combination': 'Total', 'Probability': None, 'Load Loss (MW)': None, 'Risk (MW)': L_total})
    return results, L_total

def simulate_table_ix(net_single, net_breaker_half):
    results = []
    for idx, line in net_single.line.iterrows():
        line_name = line['name']
        _, L_total_single = calculate_risk(net_single, idx, P_p0, P_lk, P_rk, config='Single Bus', fault=line_name)
        _, L_total_breaker = calculate_risk(net_breaker_half, idx, P_p0, P_lk_breaker_half, P_rk_breaker_half, config='Breaker-and-a-Half', fault=line_name)
        results.append({
            'Line': line_name,
            'Single Bus Risk (MW)': L_total_single,
            'Breaker-and-a-Half Risk (MW)': L_total_breaker
        })
    return pd.DataFrame(results)

def simulate_table_x(net_single, net_breaker_half):
    results = []
    for idx, line in net_single.line.iterrows():
        line_name = line['name']
        results_single, _ = calculate_risk(net_single, idx, P_p0, P_lk, P_rk, config='Single Bus', fault=line_name)
        for res in results_single:
            res['Line'] = line_name
            res['Config'] = 'Single Bus'
            results.append(res)
    return pd.DataFrame(results)

def simulate_table_xi(net_breaker_half):
    results = []
    for idx, line in net_breaker_half.line.iterrows():
        line_name = line['name']
        results_breaker, _ = calculate_risk(net_breaker_half, idx, P_p0, P_lk_breaker_half, P_rk_breaker_half, config='Breaker-and-a-Half', fault=line_name)
        for res in results_breaker:
            res['Line'] = line_name
            res['Config'] = 'Breaker-and-a-Half'
            results.append(res)
    return pd.DataFrame(results)

table_ix_results = simulate_table_ix(net_single_bus, net_breaker_half)
print("\nTable IX (Tie-Line Risks):")
print(table_ix_results[['Line', 'Single Bus Risk (MW)', 'Breaker-and-a-Half Risk (MW)']])

table_x_results = simulate_table_x(net_single_bus, net_breaker_half)
print("\nTable X (Single Bus Risks):")
print(table_x_results[['Line', 'Config', 'Combination', 'Probability', 'Load Loss (MW)', 'Risk (MW)']])

table_xi_results = simulate_table_xi(net_breaker_half)
print("\nTable XI (Breaker-and-a-Half Risks):")
print(table_xi_results[['Line', 'Config', 'Combination', 'Probability', 'Load Loss (MW)', 'Risk (MW)']])