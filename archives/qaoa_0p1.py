from scipy.optimize import minimize
import re
from qxelarator import qxelarator
from functools import reduce
import numpy as np

class VQE(object):
    
    def __init__(self):
        self.minimizer = minimize
        self.minimizer_kwargs = {'method':'Nelder-Mead', 'options':{'maxiter':200, 'ftol':1.0e-8, 'xtol':1.0e-8, 'disp':True}}
    
    def vqe_run(self, ansatz, h, steps, x0, aid, cfs):
        
        """
        args:
            ansatz: variational functional closure in cQASM
            h: hamiltonian
            x0: initial parameters for generating the function of the functional
        return:
            x: set of ansats parameters
            fun: scalar value of the objective function
        """
        t_name = "test_output/"+ansatz+".qasm"
        p_name = "test_output/"+ansatz+"_try"+".qasm"
                
        def objective_func(x):
            add_param(x) # If parameterised program construct not available, define new program here            
            return expectation(h)
        
        def add_param(x):
            template = open(t_name,"r")
            prog = open(p_name,"w")
            param_ctr = 0
            s = 0
            param_max = len(cfs)
            for line in template:
                if re.search('\*',line):
                    if aid[param_ctr] == 0: # beta replacer
                        theta = x[s]
                    else: # gamma replacer
                        theta = x[s+steps]
                    line_new = re.sub('\*',str(theta*cfs[param_ctr]), line)
                    param_ctr += 1
                    if param_ctr == param_max:
                        param_ctr = 0
                        s += 1
                    prog.write(line_new)
                else:
                    prog.write(line)
            template.close()
            prog.close()     
            
        def expectation(h):
            # We will not use the wavefunction (display command) as is not possible in a real QC
            # E = <wf|H|wf> = real(dot(transjugate(wf),dot(H,wf))) 
            
            # WATSON: correct this for n-qubits
            qx = qxelarator.QX()
            qx.set(p_name)
            shots = 1000
            p0 = 0
            for i in range(shots):
                qx.execute()
                c0 = qx.get_measurement_outcome(0)
                if c0 == False:
                    p0 = p0+1
            E = (p0/shots)**2 - ((shots-p0)/shots)**2
            return E
        
        args = [objective_func, x0]
        return self.minimizer(*args, **self.minimizer_kwargs)


class QAOA(object):      
    def get_angles(self, qubits, steps, betas, gammas, ham, ang_id, coeffs):
        # Finds optimal angles with the quantum variational eigensolver method.
        t_name = "test_output/graph.qasm"
        tv_name = "test_output/qaoa.qasm"
        p_name = "test_output/qaoa_try.qasm"

        def make_qaoa():
            cfs = []
            # Make VQE ansatz template from QAOA ansatz
            prog = open(tv_name,"w")
            prog.write("qubits "+str(qubits)+"\n")
            # Reference state preparation
            for i in range(0,qubits):
                prog.write("h q"+str(i)+"\n")
            # Repeat ansatz for specified steps
            for i in range(0,steps):
                template = open(t_name,"r")
                for line in template:
                    prog.write(line)
                template.close()
                cfs = np.hstack((cfs,coeffs))
            prog.close()
            return cfs
            
        full_coeffs = make_qaoa()
        #H_cost = []
        angles = np.hstack((betas, gammas)) # A concatenated list of angles [betas]+[gammas]
        
        v = VQE()
        result = v.vqe_run("qaoa", ham, steps, angles, ang_id, coeffs) # VQE for PauliTerm Hamiltonian and coefficients       
        return result
        
    def probabilities(ang):
        # Computes the probability of each state given a particular set of angles.
        prog = "test_output/qaoa_try.qasm"
        probs = []
        # RUN AND MEASURE ALL n QUBITS, TO DETERINE PROBABILITY OF ALL 2^n STATES
        return probs
        
        
    #def get_string():
        # Compute the most probable string.
        
###################################################################################################################

import networkx as nx

def graph_to_pqasm(g):
    # Specific for Max-Cut Hamiltonian
    # PauliTerm to Gates concept from rigetti/pyquil/pyquil/paulis.py
    coeffs = [] # Weights for the angle parameter for each gate
    angle_id = []
    sZ = np.array([[1,0],[0,-1]])
    sX = np.array([[0,1],[1,0]])
    I = np.eye(2)   
    H_cost = np.kron(I,np.kron(I,I))
    H_cost = np.dot(np.kron(I,np.kron(I,sZ)),H_cost)
    H_cost = np.dot(np.kron(I,np.kron(sZ,I)),H_cost)
    H_cost = np.dot(np.kron(I,np.kron(I,sX)),H_cost)
    H_cost = np.dot(np.kron(I,np.kron(sZ,I)),H_cost)
    H_cost = np.dot(np.kron(sZ,np.kron(I,I)),H_cost)
    H_cost = np.dot(np.kron(I,np.kron(sX,I)),H_cost)
    #print(H_cost)
    t_name = "test_output/graph.qasm"
    ansatz = open(t_name,"w")   
    for i,j in g.edges():
        # 0.5*Z_i*Z_j
        ansatz.write("cnot q"+str(i)+",q"+str(j)+"\n")
        ansatz.write("rz q"+str(i)+",*\n")
        coeffs.append(2*0.5)
        angle_id.append(0) # beta
        ansatz.write("cnot q"+str(i)+",q"+str(j)+"\n")
        # -0.5*I_0
        ansatz.write("x q"+str(0)+"\n")
        ansatz.write("rz q"+str(0)+",*\n")
        coeffs.append(-1*0.5)
        angle_id.append(0) # beta
        ansatz.write("x q"+str(0)+"\n")
        ansatz.write("rz q"+str(0)+",*\n")
        coeffs.append(-1*0.5)
        angle_id.append(0) # beta
    for i in g.nodes():
        # -X_i
        ansatz.write("h q"+str(i)+"\n")
        ansatz.write("rz q"+str(i)+",*\n")
        coeffs.append(2*-1)
        angle_id.append(1) # gamma
        ansatz.write("h q"+str(i)+"\n")
    ansatz.close()
    return H_cost, coeffs, angle_id
    
###################################################################################################################

# Barbell graph
g = nx.Graph()
g.add_edge(0,1)
g.add_edge(1,2)
hc, coeffs, aid = graph_to_pqasm(g)

steps = 2
qb = len(g.nodes()) # Number of qubits
b = np.random.uniform(0, np.pi, steps) # Initial beta angle parameters of cost Hamiltonian
g = np.random.uniform(0, 2*np.pi, steps) # Initial gamma angle parameters of driving/mixing Hamiltonian

#print(qb,steps,b,g,hc,aid,coeffs)

qaoa_obj = QAOA()

r = qaoa_obj.get_angles(qb,steps,b,g,hc,aid,coeffs)
print(r.status, r.fun, r.x)
'''
Optimization terminated successfully.
         Current function value: 1.000000
         Iterations: 25
         Function evaluations: 149
(array([2.32105514, 2.0138622 ]), array([2.20695693, 1.86485137]))
'''

# The last qaoa_try will have the optimal angles
probs = qaoa_obj.probabilities()
#print(probs)