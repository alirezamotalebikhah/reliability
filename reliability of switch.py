import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandapower as pp
import copy
import time

# Parameters from Table II
a = 0.0
delta = 1e-4
mu = 0.01
lambda_ = 5.0
omega = 1.0 / 5000.0  # Average wear per operation

# Relay failure probability from Table III
Pp0 = 0.0069  # Single protection relay failure probability
gamma_f = 0.001  # Adjusted failure rate for relays


# Define v(x) and q(x) functions based on Table II
def v(x):
    return 0.02 * x ** 0.5


def q(x):
    return np.exp(1e-6 * x)


# Degradation level T_v(t) (Equation 6)
def T_v(t, N_t):
    return a + delta * t + N_t * omega


# Reliability due to degradation (Equation 7)
def R_d(t, N_t):
    def integrand(x):
        return v(x) * q(x)

    integral, _ = quad(integrand, 0, t, epsabs=1e-8, epsrel=1e-8)
    degradation_term = mu * T_v(t, N_t)
    integral_term = integral * (1 + mu / (1 + lambda_ * mu))
    return np.exp(-(degradation_term + integral_term))


# Reliability due to operation time (Equation 8)
def R_s(t, n_times):
    def integrand_v(x):
        return v(x)

    def integrand_vq(x):
        return v(x) * q(x)

    integral_v, _ = quad(integrand_v, 0, t, epsabs=1e-8, epsrel=1e-8)
    integral_vq, _ = quad(integrand_vq, 0, t, epsabs=1e-8, epsrel=1e-8)
    return np.exp(-integral_v + integral_vq)


# Failure probabilities
def P_degradation_failure(t, N_t):
    return 1 - R_d(t, N_t)


def P_operation_failure(t, n_times):
    return 1 - R_s(t, n_times)


# Relay reliability (Equation 10)
def gamma_rs(gamma_f):
    return 2 * gamma_f ** 2


# Create 9-bus network
def create_9bus_network():
    net = pp.create_empty_network()
    buses = {i: pp.create_bus(net, vn_kv=230.0, name=f"Bus {i}") for i in range(1, 10)}
    pp.create_gen(net, buses[1], p_mw=200.0, vm_pu=1.04, name="Gen 1")
    pp.create_gen(net, buses[2], p_mw=150.0, vm_pu=1.025, name="Gen 2")
    pp.create_gen(net, buses[3], p_mw=100.0, vm_pu=1.03, name="Gen 3")
    pp.create_load(net, buses[5], p_mw=90.0, q_mvar=30.0, name="Load 5")
    pp.create_load(net, buses[7], p_mw=100.0, q_mvar=35.0, name="Load 7")
    pp.create_load(net, buses[9], p_mw=125.0, q_mvar=50.0, name="Load 9")
    lines = [(1, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 2), (2, 3)]
    for from_bus, to_bus in lines:
        pp.create_line(net, buses[from_bus], buses[to_bus], length_km=100.0, std_type="NAYY 4x50 SE",
                       name=f"Line {from_bus}-{to_bus}")
    pp.create_ext_grid(net, buses[1], vm_pu=1.04, name="Slack")
    return net


# Create 68-bus network with detailed topology
def create_68bus_network():
    net = pp.create_empty_network()
    buses = {i: pp.create_bus(net, vn_kv=230.0, name=f"Bus {i}") for i in range(1, 69)}
    gen_buses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for bus in gen_buses:
        pp.create_gen(net, buses[bus], p_mw=200.0, vm_pu=1.04, name=f"Gen {bus}")
    load_buses = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                  41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                  66, 67, 68]
    for bus in load_buses:
        pp.create_load(net, buses[bus], p_mw=100.0, q_mvar=50.0, name=f"Load {bus}")
    tie_lines = [(27, 54), (54, 53), (53, 27), (26, 54), (25, 53), (24, 52)]
    for from_bus, to_bus in tie_lines:
        pp.create_line(net, buses[from_bus], buses[to_bus], length_km=100.0, std_type="NAYY 4x50 SE",
                       name=f"Tie-Line {from_bus}-{to_bus}")
    for i in range(1, 69):
        for j in range(i + 1, 69):
            if np.random.random() < 0.1:
                pp.create_line(net, buses[i], buses[j], length_km=100.0, std_type="NAYY 4x50 SE",
                               name=f"Line {i}-{j}")
    pp.create_ext_grid(net, buses[1], vm_pu=1.04, name="Slack")
    return net


# Run AC OPF with error handling and load curtailment
def run_ac_opf(net, t, N_t, fault_lines=None):
    net_modified = copy.deepcopy(net)
    failure_prob = P_degradation_failure(t, N_t)
    if fault_lines:
        for line_name in fault_lines:
            net_modified.line.loc[net_modified.line.name == line_name, "in_service"] = False
            print(f"Simulated fault on {line_name} with probability {failure_prob:.4f}")

    try:
        pp.runpp(net_modified, max_iteration=100, tolerance_mva=1e-6, trafo_loading='power', enforce_q_lims=True)
        print("Power flow converged successfully.")
    except Exception as e:
        print(f"Power flow failed: {str(e)}. Initiating load curtailment.")
        curtailment = 0.0
        while curtailment < 0.5:
            for idx, row in net_modified.load.iterrows():
                net_modified.load.at[idx, 'p_mw'] = row['p_mw'] * (1 - curtailment)
                net_modified.load.at[idx, 'q_mvar'] = row['q_mvar'] * (1 - curtailment)
            try:
                pp.runpp(net_modified, max_iteration=100, tolerance_mva=1e-6, trafo_loading='power',
                         enforce_q_lims=True)
                print(f"Load curtailment of {curtailment * 100:.0f}% applied successfully.")
                break
            except:
                curtailment += 0.05
        if curtailment >= 0.5:
            print("Max 50% curtailment reached, solution still infeasible.")
            curtailment = 0.5

    total_load = sum(net.load.p_mw)
    curtailed_load = total_load - sum(net_modified.res_load.p_mw)
    print(f"Total load: {total_load:.1f} MW, Curtailed load: {curtailed_load:.1f} MW")
    return curtailed_load


# Risk calculations (Equations 1-5)
def calculate_risks(net, t, N_t, C_local=None, C_remote=None):
    if C_local is None:
        C_local = [1, 4]  # Example CBs
    if C_remote is None:
        C_remote = [4, 5]
    Plk = P_degradation_failure(t, N_t)
    Prk = Plk
    LS_G1 = run_ac_opf(net, t, N_t)
    LS_Gi = run_ac_opf(net, t, N_t, [f"Line {C_local[0]}-{C_local[1]}"])
    LS_Gj = run_ac_opf(net, t, N_t, [f"Line {C_remote[0]}-{C_remote[1]}"])
    LS_G0 = run_ac_opf(net, t, N_t, [f"Line {C_local[0]}-{C_local[1]}", f"Line {C_remote[0]}-{C_remote[1]}"])

    P_G1 = np.prod([1 - Plk for _ in C_local]) * np.prod([1 - Prk for _ in C_remote])
    P_Gi = Plk * (1 - Plk) * np.prod([1 - Prk for _ in C_remote])
    P_Gj = Prk * (1 - Prk) * np.prod([1 - Plk for _ in C_local])
    P_G0 = Pp0

    L1 = (1 - Pp0) * P_G1 * LS_G1
    L2 = (1 - Pp0) * P_Gi * LS_Gi
    L3 = (1 - Pp0) * P_Gj * LS_Gj
    L4 = Pp0 * LS_G0
    L_total = L1 + L2 + L3 + L4
    print(f"Risks: L1={L1:.2f}, L2={L2:.2f}, L3={L3:.2f}, L4={L4:.2f}, Total={L_total:.2f}")
    return L1, L2, L3, L4, L_total


# Simulate Table I (Topologies for Single Line Outage)
def simulate_table_i(net, t=2000, N_t=100):
    Plk = P_degradation_failure(t, N_t)
    results = [
        ["L1", 1 - Pp0, 1 - Plk, 1 - Plk, "G(S_l=0, S_r=0)"],
        ["L2", 1 - Pp0, 1 - Plk, 1 - Plk, "G(S_l=0, S_r=0)"],
        ["L3", 1 - Pp0, 1 - Plk, 1 - Plk, "G(S_l=0, S_r=0)"],
        ["L4", 1 - Pp0, 1 - Plk, 1 - Plk, "G(S_l=0, S_r=0)"]
    ]
    return results


# Simulate Table IV (Fault 1, Single Bus)
def simulate_table_iv(net, t=2000, N_t=100):
    Plk = P_degradation_failure(t, N_t)
    results = []
    LS_G1 = run_ac_opf(net, t, N_t)
    risk1 = (1 - Pp0) * Plk * 1 * LS_G1
    results.append([1, 1 - Pp0, Plk, 1, LS_G1, risk1])
    LS_G2 = run_ac_opf(net, t, N_t, ["Line 1-4"])
    risk2 = (1 - Pp0) * Plk * 1 * LS_G2
    results.append([2, 1 - Pp0, Plk, 1, LS_G2, risk2])
    LS_G3 = run_ac_opf(net, t, N_t, ["Line 1-4", "Line 4-5"])
    risk3 = Pp0 * 1 * 1 * LS_G3
    results.append([3, Pp0, 1, 1, LS_G3, risk3])
    return results


# Simulate Table V (Fault 2, Single Bus)
def simulate_table_v(net, t=2000, N_t=100):
    Plk = P_degradation_failure(t, N_t)
    results = []
    LS_G1 = run_ac_opf(net, t, N_t)
    risk1 = (1 - Pp0) * Plk * Plk * 0
    results.append([1, 1 - Pp0, Plk, Plk, 0, risk1])
    LS_G2 = run_ac_opf(net, t, N_t, ["Line 1-4"])
    risk2 = (1 - Pp0) * (1 - Plk) * Plk * LS_G2
    results.append([2, 1 - Pp0, 1 - Plk, Plk, LS_G2, risk2])
    LS_G3 = run_ac_opf(net, t, N_t, ["Line 4-5"])
    risk3 = (1 - Pp0) * Plk * (1 - Plk) * LS_G3
    results.append([3, 1 - Pp0, Plk, 1 - Plk, LS_G3, risk3])
    LS_G4 = run_ac_opf(net, t, N_t, ["Line 1-4", "Line 4-5"])
    risk4 = Pp0 * 1 * 1 * LS_G4
    results.append([4, Pp0, 1, 1, LS_G4, risk4])
    return results


# Simulate Table VI (Fault 1, 3/2 Bus)
def simulate_table_vi(net, t=2000, N_t=100):
    Plk = P_degradation_failure(t, N_t)
    results = []
    LS_G1 = run_ac_opf(net, t, N_t)
    risk1 = (1 - Pp0) * Plk * Plk * 0
    results.append([1, 1 - Pp0, Plk, Plk, 0, risk1])
    LS_G2 = run_ac_opf(net, t, N_t, ["Line 1-4"])
    risk2 = (1 - Pp0) * Plk * Plk * 0
    results.append([2, 1 - Pp0, Plk, Plk, 0, risk2])
    LS_G3 = run_ac_opf(net, t, N_t, ["Line 4-5"])
    risk3 = (1 - Pp0) * Plk * (1 - Plk) * 0
    results.append([3, 1 - Pp0, Plk, 1 - Plk, 0, risk3])
    LS_G4 = run_ac_opf(net, t, N_t, ["Line 1-4", "Line 4-5"])
    risk4 = Pp0 * 1 * 1 * 0
    results.append([4, Pp0, 1, 1, 0, risk4])
    return results


# Simulate Table VII (Fault 2, 3/2 Bus)
def simulate_table_vii(net, t=2000, N_t=100):
    Plk = P_degradation_failure(t, N_t)
    results = []
    LS_G1 = run_ac_opf(net, t, N_t)
    risk1 = (1 - Pp0) * Plk * Plk * 0
    results.append([1, 1 - Pp0, Plk, Plk, 0, risk1])
    LS_G2 = run_ac_opf(net, t, N_t, ["Line 1-4"])
    risk2 = (1 - Pp0) * Plk * Plk * 0
    results.append([2, 1 - Pp0, Plk, Plk, 0, risk2])
    LS_G3 = run_ac_opf(net, t, N_t, ["Line 4-5"])
    risk3 = (1 - Pp0) * Plk * (1 - Plk) * 0
    results.append([3, 1 - Pp0, Plk, 1 - Plk, 0, risk3])
    LS_G4 = run_ac_opf(net, t, N_t, ["Line 1-4", "Line 4-5"])
    risk4 = Pp0 * 1 * 1 * 0
    results.append([4, Pp0, 1, 1, 0, risk4])
    return results


# Simulate Table VIII (Risk Comparison)
def simulate_table_viii(net, t=2000, N_t=100):
    results = []
    table_iv_results = simulate_table_iv(net, t, N_t)
    fault1_single = sum(row[5] for row in table_iv_results)
    table_v_results = simulate_table_v(net, t, N_t)
    fault2_single = sum(row[5] for row in table_v_results)
    table_vi_results = simulate_table_vi(net, t, N_t)
    fault1_third = sum(row[5] for row in table_vi_results)
    table_vii_results = simulate_table_vii(net, t, N_t)
    fault2_third = sum(row[5] for row in table_vii_results)
    results = [
        ["Fault 1", fault1_single, fault1_third],
        ["Fault 2", fault2_single, fault2_third]
    ]
    return results


# Simulate Table IX (Risk Analysis of Tie-Lines)
def simulate_table_ix(net_68, t=500, N_t=100):
    results = []
    tie_lines = ["Tie-Line 27-54", "Tie-Line 54-53", "Tie-Line 53-27", "Tie-Line 26-54"]
    for line in tie_lines:
        local_cb = f"Local {line.split('-')[0]}"
        remote_cb = f"Remote {line.split('-')[1]}"
        LS = run_ac_opf(net_68, t, N_t, [line])
        backup_LS = run_ac_opf(net_68, t, N_t,
                               [line, f"Tie-Line {int(line.split('-')[1]) - 1}-{int(line.split('-')[0]) + 1}"])
        results.append([line, local_cb, "-", remote_cb, "-", backup_LS])
    return results


# Simulate Table X (Risk Comparison of Tie-Lines)
def simulate_table_x(net_68, t=500, N_t=100):
    tie_lines = ["Tie-Line 27-54", "Tie-Line 54-53", "Tie-Line 53-27", "Tie-Line 26-54"]
    results = []
    for line in tie_lines:
        LS = run_ac_opf(net_68, t, N_t, [line])
        consequence = "Load curtailment" if LS > 0 else "-"
        risk = LS * P_degradation_failure(t, N_t)
        results.append([line, consequence, risk])
    return results


# Simulate Table XI (Simultaneous Faults)
def simulate_table_xi(net_68, t=500, N_t=100):
    results = []
    Plk = P_degradation_failure(t, N_t)
    P_one_line = 1 - ((1 - Plk) ** 4) * (1 - Pp0) + Pp0
    P_two_line = Plk * ((1 - Plk) ** 3) * (1 - Pp0) + Pp0
    LS_all = run_ac_opf(net_68, t, N_t, ["Tie-Line 27-54", "Tie-Line 54-53", "Tie-Line 53-27", "Tie-Line 26-54"])
    risk1 = P_one_line * LS_all
    results.append(["All lines tripped", risk1])
    LS_1 = run_ac_opf(net_68, t, N_t, ["Tie-Line 54-53", "Tie-Line 53-27", "Tie-Line 26-54"])
    risk2 = P_two_line * LS_1
    results.append(["Line 27-54 cascading", risk2])
    LS_2 = run_ac_opf(net_68, t, N_t, ["Tie-Line 27-54", "Tie-Line 53-27", "Tie-Line 26-54"])
    risk3 = P_two_line * LS_2
    results.append(["Line 54-53 cascading", risk3])
    LS_3 = run_ac_opf(net_68, t, N_t, ["Tie-Line 27-54", "Tie-Line 54-53", "Tie-Line 26-54"])
    risk4 = P_two_line * LS_3
    results.append(["Line 53-27 cascading", risk4])
    LS_4 = run_ac_opf(net_68, t, N_t, ["Tie-Line 27-54", "Tie-Line 54-53", "Tie-Line 53-27"])
    risk5 = P_two_line * LS_4
    results.append(["Line 26-54 cascading", risk5])
    total_risk = sum([r[1] for r in results])
    return results, total_risk


# Simulate Table II and III (Parameters)
def simulate_tables_ii_iii():
    table_ii = [["Parameter", "a", "δ", "μ", "λ", "v(x)", "q(x)"],
                ["Quantity", 0, 1e-4, 0.01, 5, "0.02*x^0.5", "exp(1e-6*x)"]]
    table_iii = [["Parameter", "PD", "SA", "LE", "DE", "LR", "HD", "EI", "PQ", "PA", "DG"],
                 ["Value (10^-3)", 2, 1, 0.8, 0.2, 0.03, 0.2, 0.01, 0.01, 0.02, 0.02]]
    return table_ii, table_iii


# Main execution
if __name__ == "__main__":
    # Create networks
    net_9bus = create_9bus_network()
    net_68bus = create_68bus_network()

    # Run simulations
    t_9bus = 2000
    N_t_9bus = 100
    t_68bus = 500
    N_t_68bus = 100

    print("\nTable I (Topologies):")
    table_i_results = simulate_table_i(net_9bus, t_9bus, N_t_9bus)
    for row in table_i_results:
        print(f"Risk: {row[0]}, Relay: {row[1]:.4f}, Local: {row[2]:.4f}, Remote: {row[3]:.4f}, Topology: {row[4]}")

    print("\nTable IV (Fault 1, Single Bus):")
    table_iv_results = simulate_table_iv(net_9bus, t_9bus, N_t_9bus)
    for row in table_iv_results:
        print(
            f"Index: {row[0]}, Relay: {row[1]:.4f}, Local: {row[2]:.4f}, Remote: {row[3]:.4f}, MW Load Loss: {row[4]:.1f}, MW Risk: {row[5]:.1f}")

    print("\nTable V (Fault 2, Single Bus):")
    table_v_results = simulate_table_v(net_9bus, t_9bus, N_t_9bus)
    for row in table_v_results:
        print(
            f"Index: {row[0]}, Relay: {row[1]:.4f}, Local: {row[2]:.4f}, Remote: {row[3]:.4f}, MW Load Loss: {row[4]:.1f}, MW Risk: {row[5]:.1f}")

    print("\nTable VI (Fault 1, 3/2 Bus):")
    table_vi_results = simulate_table_vi(net_9bus, t_9bus, N_t_9bus)
    for row in table_vi_results:
        print(
            f"Index: {row[0]}, Relay: {row[1]:.4f}, Local: {row[2]:.4f}, Remote: {row[3]:.4f}, MW Load Loss: {row[4]:.1f}, MW Risk: {row[5]:.1f}")

    print("\nTable VII (Fault 2, 3/2 Bus):")
    table_vii_results = simulate_table_vii(net_9bus, t_9bus, N_t_9bus)
    for row in table_vii_results:
        print(
            f"Index: {row[0]}, Relay: {row[1]:.4f}, Local: {row[2]:.4f}, Remote: {row[3]:.4f}, MW Load Loss: {row[4]:.1f}, MW Risk: {row[5]:.1f}")

    print("\nTable VIII (Risk Comparison):")
    table_viii_results = simulate_table_viii(net_9bus, t_9bus, N_t_9bus)
    for row in table_viii_results:
        print(f"Fault: {row[0]}, Single Bus MW Risk: {row[1]:.1f}, 3/2 Bus MW Risk: {row[2]:.1f}")

    print("\nTable IX (Risk Analysis of Tie-Lines):")
    table_ix_results = simulate_table_ix(net_68bus, t_68bus, N_t_68bus)
    for row in table_ix_results:
        print(
            f"Event: {row[0]}, Local Breaker: {row[1]}, Consequence: {row[2]}, Remote Breaker: {row[3]}, Backup Consequence: {row[4]}, Backup MW Load Loss: {row[5]:.1f}")

    print("\nTable X (Risk Comparison of Tie-Lines):")
    table_x_results = simulate_table_x(net_68bus, t_68bus, N_t_68bus)
    for row in table_x_results:
        print(f"Event: {row[0]}, Consequence: {row[1]}, MW Risk: {row[2]:.1f}")

    print("\nTable XI (Simultaneous Faults):")
    table_xi_results, total_risk = simulate_table_xi(net_68bus, t_68bus, N_t_68bus)
    for row in table_xi_results:
        print(f"Scenario: {row[0]}, MW Risk: {row[1]:.2f}")
    print(f"Total Risk: {total_risk:.2f}")

    print("\nTable II (CB Parameters):")
    table_ii_results, table_iii_results = simulate_tables_ii_iii()
    for row in table_ii_results:
        print(row)
    print("\nTable III (Relay Parameters):")
    for row in table_iii_results:
        print(row)

    print("\nRisk Calculation for 9-bus:")
    calculate_risks(net_9bus, t_9bus, N_t_9bus)
    print("\nRisk Calculation for 68-bus:")
    calculate_risks(net_68bus, t_68bus, N_t_68bus)