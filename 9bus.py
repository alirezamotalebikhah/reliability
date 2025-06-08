import pandapower  as pp
import pandapower.networks as pn
import networkx as nx
import numpy as np
import pandas as pd
from Cython.Includes.cpython.time import result
from scipy.integrate import quad
a=0
delta=1e-4
mu=0.01
lambda_=5
v_x = lambda x:0.02*x**0.5
q_x = lambda x:np.exp(1e-4*x)
t=2000
n_operations = 100
P_p0 = 0.0069
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
    return P_cb
P_lk = P_rk = 0.81
net=pn.create_ieee9bus()
net.gen.loc[net.gen.bus==1,'p_mw'] = 0
net.gen.loc[net.gen.bus==2,'p_mw'] +=50
net.gen.loc[net.gen.bus==3,'p_mw'] +=50
line_config={
    0: {'relay_type':'overcurent'},
    1: {'relay_type' : 'overcurrent'},
    2: {'relay_type' : 'overcurrent'},
    3: {'relay_type' : 'directional'},
    4: {'relay_type' : 'directional'},
    5: {'relay_type' : 'directional'}
    }
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
def calculate_risk(net,line_id,P_p0 , P_lk , P_rk):
    results=[]
    P_G1 = (1 - P_p0)*(1-P_lk)*(1-P_rk)
    LS_G1 = analyze_topology(net,line_id)
    L1 = P_G1 * LS_G1
    results.append({'Combination' : '1 (Normal)' , 'Probability' : P_G1 , 'Load Loss(MW)' : LS_G1 , 'Risk' : L1})

    P_Gi = (1-P_p0)*(P_lk)*(1-P_rk)
    LS_Gi = analyze_topology(net,line_id)
    L2 = P_Gi * LS_Gi
    results.append({'Combination': '2 (Local CB Failure)' , 'Probability':P_Gi,'Load Loss(MW)':LS_Gi,'Risk':L2})

    P_Gj = (1-P_p0)*(P_lk)*(P_rk)
    LS_Gj = analyze_topology(net,line_id)
    L3 = P_Gj * LS_Gj
    results.append({'Combination' : '3 (Remote CB Failure)' , 'Probability' : P_Gj , 'Load_loss(MW)' : LS_Gj,'Risk':L3})

    P_G0 = P_p0
    LS_G0 = analyze_topology(net,line_id)
    L4 = P_G0 * LS_G0
    results.append({'Combination' : '4 (Relay Failure)' , 'Probability':P_G0 , 'Load_loss(MW)' : LS_G0, 'Risk':L4})

    L_total = L1 + L2 + L3 + L4
    results.append({'Combination': 'Total','Probability' : None , 'Load_loss(MW)' : None , 'Risk' : L_total})
    return results





