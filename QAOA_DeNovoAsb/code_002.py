# ..:$ python3 code_001.py 

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from qxelarator import qxelarator
import matplotlib.pyplot as plt

######################################################

track_opt = []
track_optstep = 0
track_probs = []

class QAOA(object):

    def __init__(self, maxiter, shots):
        self.qx = qxelarator.QX()
        self.minimizer = minimize
        self.minimizer_kwargs = {'method':'Nelder-Mead', 'options':{'maxiter':maxiter, 
                                 'ftol':1.0e-2, 'xtol':1.0e-2, 'disp':True, 'return_all':True}}
        self.p_name = "test_output/qaoa_run.qasm"
        self.shots = shots 
        self.expt = 0    
    
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

        def qasmify(params, wpp):
            prog = open(self.p_name,"w")
            prog.write("# De-parameterized QAOA ansatz\n")
            prog.write("version 1.0\n")
            prog.write("qubits "+str(n_qubits)+"\n")
            
            prog.write(".qk(1)\n")
            
            # De-parameterize pqasm
            a_id = 0
            a_ctr = 0
            c_ctr = 0
            for i in pqasm:
                # 1-qubit parametric gates
                if i[0] == 'rx' or i[0] == 'ry' or i[0] == 'rz':
                    prog.write(i[0]+" q["+str(i[1])+"],"+str(coeffs[c_ctr]*params[a_id])+"\n")
                    c_ctr += 1
                    a_ctr += 1
                    if a_ctr >= ang_nos[a_id]:
                        a_id += 1
                        a_ctr = 0
                # 1-qubit discrete gates
                elif i[0] == 'x' or i[0] == 'y' or i[0] == 'z' or i[0] == 'h':
                    prog.write(i[0]+" q["+str(i[1])+"]\n")
                # 2-qubit discrete gates
                else:
                    prog.write(i[0]+" q["+str(i[1][0])+"],q["+str(i[1][1])+"]\n")
            
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
                prog.write("measure q["+str(i)+"]\n")
            prog.close()        

        def expectation(params):
            E = 0
            self.expt = 0
            xsgn = [-1,1] # Try [1,-1] with ry +pi/2 in qasmify for pt == 'X'
            zsgn = [1,-1]
            isgn = [1,-1]
            global track_probs
            track_probs = np.zeros(2**n_qubits)

            for wpp in wsopp:
                qasmify(params,wpp[1])
                self.qx.set(self.p_name)

                Epp = 0
                p = np.zeros(2**n_qubits)
                c = np.zeros(n_qubits,dtype=bool)
                for i in range(self.shots):
                    self.qx.execute()
                    # self.qx.execute(1)
                    for i in range(n_qubits):
                        c[i] = self.qx.get_measurement_outcome(i)
                    idx = sum(v<<i for i, v in enumerate(c[::-1]))    
                    p[idx] += 1/self.shots
                
                psgn = [1]
                for pt in wpp[1]:
                    if pt == "X":
                        psgn = np.kron(psgn,xsgn)
                    #elif pt == "Y":
                    #    psgn = np.kron(psgn,xsgn) # TBD
                    elif pt == "Z":
                        psgn = np.kron(psgn,zsgn)
                    else: # Identity
                        psgn = np.kron(psgn,isgn)
                for pn in range(2**n_qubits):
                    Epp += psgn[pn]*p[pn]                
                E += wpp[0]*Epp
                self.expt += E
                
                # for pn in range(2**n_qubits):
                #     track_probs += wpp[0]*p[pn]
                if wpp[1] == 'III':
                    track_probs = p
                # break
            return E

        def expectation_isv(params):
            # With get_state()
            return expectation(params)

        def intermediate(cb):
            global track_opt
            global track_optstep
            global track_probs
            print("Step: ",track_optstep)
            print("Current Optimal Parameters: ",cb)
            print("Current Expectation Value: ",self.expt)
            print("Current Optimal Probabilities: ",track_probs)
            track_optstep += 1
            # input("Press Enter to continue to step "+str(track_optstep))
            track_opt.append([track_optstep, cb, track_probs])
               
        args = [expectation_isv, params]
        r = self.minimizer(*args, callback=intermediate, **self.minimizer_kwargs) 
        return r

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

init_gammas = np.random.uniform(0, 2*np.pi, steps) # Initial angle parameters for cost Hamiltonian
init_betas = np.random.uniform(0, np.pi, steps) # Initial angle parameters for mixing/driving Hamiltonian

init_gammas = [0, 0]
init_betas = [0, 0]

# init_gammas = [5.13465537, 0.6859112]
# init_betas = [1.39939047, 3.22152587]
# 2 -0.12200000000000007 [5.13978292 1.32175648 0.71116758 3.28077758]
# [39, array([5.13978292, 1.32175648, 0.71116758, 3.28077758]), array([0.078, 0.096, 0.238, 0.094, 0.11 , 0.228, 0.086, 0.07 ])]

# init_gammas = [5.21963138, 4.52995014]
# init_betas = [2.62196640, 1.20937913]
# 2 -0.11000000000000004 [5.21963138 2.75306472 4.52995014 1.20937913]
# [39, array([5.21963138, 2.75306472, 4.52995014, 1.20937913]), array([0.006, 0.006, 0.474, 0.014, 0.006, 0.472, 0.012, 0.01 ])]

######################################################

maxiter = 40
shots = 500 # should be some factor of number of qubits to have the same precision

qaoa_obj = QAOA(maxiter, shots)
res = qaoa_obj.qaoa_run(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)
print(res.status, res.fun, res.x)
print(track_opt[-1])
# %matplotlib inline
plt.ylim((0,1))
plt.plot(track_opt[0][2],'--')  # Initial
plt.plot(track_opt[-1][2]) # Final
plt.show()