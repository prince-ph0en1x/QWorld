# ..:$ python3 code_001.py 

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from qxelarator import qxelarator

######################################################

class QAOA(object):

    def __init__(self):
        self.minimizer = minimize
        self.minimizer_kwargs = {'method':'Nelder-Mead', 'options':{'maxiter':40, 
                                 'ftol':1.0e-2, 'xtol':1.0e-2, 'disp':True}}
        self.p_name = "test_output/qaoa_run.qasm"
        
    def qaoa_run(self, wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas):
        n_qubits = len(wsopp[0][1])
        pqasm = []
        coeffs = []
        ang_nos = []
        params = []
        for gate in initstate:
            pqasm.append(gate)
        for p in range(0,steps):
            for gate in ansatz:
                pqasm.append(gate)
            coeffs = np.hstack((coeffs,cfs))
            ang_nos = np.hstack((ang_nos,aid))
            params.append(init_gammas[p])
            params.append(init_betas[p]) 
        probs = np.zeros(2**n_qubits)

        def qasmify(params, wpp):
            prog = open(self.p_name,"w")
            prog.write("qubits "+str(n_qubits)+"\n")
            
            # De-parameterize pqasm
            # a_id = 0
            # a_ctr = 0
            # c_ctr = 0
            # for i in pqasm:
            #      # 1-qubit parametric gates
            #     if i[0] == 'rx' or i[0] == 'ry' or i[0] == 'rz':
            #         prog.write(i[0]+" q"+str(i[1])+","+str(cfs[c_ctr]*params[a_id])+"\n")
            #         c_ctr += 1
            #         a_ctr += 1
            #         if a_ctr >= ang_nos[a_id]:
            #             a_id += 1
            #             a_ctr = 0
            #      # 1-qubit discrete gates
            #     elif i[0] == 'x' or i[0] == 'y' or i[0] == 'z' or i[0] == 'h':
            #         prog.write(i[0]+" q"+str(i[1])+"\n")
            #      # 2-qubit discrete gates
            #     else:
            #         prog.write(i[0]+" q"+str(i[1][0])+",q"+str(i[1][1])+"\n")
            
            # Pre-rotation for Z-basis measurement
            tgt = n_qubits-1
            for pt in wpp:
                if pt == "X":
                    prog.write("ry q"+str(tgt)+",1.5708\n")
                elif pt == "Y":
                    prog.write("rx q"+str(tgt)+",-1.5708\n")
                # else Z or Identity
                tgt -= 1

            # Measure all
            for i in range(n_qubits):
                prog.write("measure q"+str(i)+"\n")
            prog.close()        

        def expectation(params):
            for wpp in wsopp:
                print(wpp)
                qasmify(params,wpp[1])
                break

            print("e")

        args = [expectation, params]
        expectation(params)
        # return self.minimizer(*args, **self.minimizer_kwargs)  
        return "hi"

######################################################

def graph_problem():
    g = nx.Graph()
    g.add_edge(0,1)
    g.add_edge(1,2)
    return g

g = graph_problem() # Barbell graph [0-1-2]

######################################################

def graph_to_wsopp(g, n_qubits):
    wsopp = []
    Iall = "I"*n_qubits
    for i,j in g.edges():
        # 0.5*Z_i*Z_j
        sopp = Iall[:n_qubits-1-i]+"Z"+Iall[n_qubits-1-i+1:]
        sopp = sopp[:n_qubits-1-j]+"Z"+sopp[n_qubits-1-j+1:]
        wsopp.append((0.5,sopp))
        # -0.5*I_0
        wsopp.append((-0.5,Iall))
    return wsopp
    
wsopp = graph_to_wsopp(g, len(g.nodes()))
# [(0.5, 'IZZ'), (-0.5, 'III'), (0.5, 'ZZI'), (-0.5, 'III')]

######################################################

initstate = []
for i in range(0,len(g.nodes())): # Reference state preparation
    initstate.append(("h",i))

def graph_to_pqasm(g,n_qubits):
    coeffs = [] # Weights for the angle parameter for each gate
    angles = [0,0] # Counts for [cost,mixing] Hamiltonian angles
    Iall = ""
    for i in range(n_qubits):
        Iall += "I"
    ansatz = [] # qasm tokens
    for i,j in g.edges():
        # 0.5*Z_i*Z_j
        ansatz.append(("cnot",[i,j]))
        ansatz.append(("rz",j))
        coeffs.append(2*0.5)
        angles[0] += 1 # gamma: cost Hamiltonian
        ansatz.append(("cnot",[i,j]))
        # -0.5*I_0
        ansatz.append(("x",0))
        ansatz.append(("rz",0))
        coeffs.append(-1*-0.5)
        angles[0] += 1 # gamma: cost Hamiltonian
        ansatz.append(("x",0))
        ansatz.append(("rz",0))
        coeffs.append(-1*-0.5)
        angles[0] += 1 # gamma: cost Hamiltonian
    for i in g.nodes():
        # -X_i
        ansatz.append(("h",i))
        ansatz.append(("rz",i))
        coeffs.append(2*-1)
        angles[1] += 1 # beta: mixing Hamiltonian
        ansatz.append(("h",i))
    return ansatz, coeffs, angles

ansatz, cfs, aid = graph_to_pqasm(g,len(g.nodes()))

steps = 2 # number of steps (QAOA blocks per iteration)

init_betas = np.random.uniform(0, np.pi, steps) # Initial angle parameters for mixing/driving Hamiltonian
init_gammas = np.random.uniform(0, 2*np.pi, steps) # Initial angle parameters for cost Hamiltonian

######################################################

qaoa_obj = QAOA()
result = qaoa_obj.qaoa_run(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)

print(result)

# print(r.status, r.fun, r.x)

# rdx = [5.21963138, 2.62196640, 4.52995014, 1.20937913] # from Forest (g1,b1,g2,b2)
# rdx = [5.13465537, 1.39939047, 0.68591120, 3.22152587] # from last run (g1,b1,g2,b2)
# rdx = r.x

# probs = qaoa_obj.get_probs(n_qubits,ansatz,rdx,ang_nos,coeffs,steps)
# print(probs)

# plt.ylim((0,1))
# plt.plot(probs)