###################################################################################################################

from scipy.optimize import minimize
from qxelarator import qxelarator
import numpy as np

class VQE(object):
    
    def __init__(self):
        self.minimizer = minimize
        self.minimizer_kwargs = {'method':'Nelder-Mead', 'options':{'maxiter':40, 'ftol':1.0e-2, 'xtol':1.0e-2, 'disp':True}}
    
    def vqe_run(self,wsopp,ansatz,n_qubits,depth,init_params,evaluate=False):
        
        p_name = "test_output/vqe_run.qasm"
               
        def qasmify(params,wpp):
            prog = open(p_name,"w")
            prog.write("qubits "+str(n_qubits)+"\n")
            for j in range(depth):
                p_ctr = 0
                for i in ansatz:
                    if i[0] == 'rx' or i[0] == 'ry' or i[0] == 'rz':
                        prog.write(i[0]+" q"+str(i[1])+","+str(params[p_ctr])+"\n")
                        p_ctr += 1
                    else: # currently handles only cnot, need to extend to hadamard
                        prog.write(i[0]+" q"+str(i[1][0])+",q"+str(i[1][1])+"\n")

            if wpp[1] == "X":
                prog.write("ry q0,1.5708\n")
            elif wpp[1] == "Y":
                prog.write("rx q0,-1.5708\n")
            # else Z or Identity

            for i in range(n_qubits):
                prog.write("measure q"+str(i)+"\n")
            prog.close()
            
        def expectation(params):
            # We will not use the wavefunction (display command) as is not possible in a real QC
            # E = <wf|H|wf> = real(dot(transjugate(wf),dot(H,wf))) 
            
            E = 0
            shots = 1000
            for wpp in wsopp: # currently only supported for single qubit
                
                qasmify(params,wpp)
                
                qx = qxelarator.QX()
                qx.set(p_name)

                p = np.zeros(2**n_qubits)
                c = np.zeros(n_qubits,dtype=bool)
                for i in range(shots):
                    qx.execute()
                    for i in range(n_qubits):
                        c[i] = qx.get_measurement_outcome(i)
                    idx = sum(v<<i for i, v in enumerate(c[::-1]))    
                    p[idx] += 1/shots
                    
                if wpp[1] == "X":
                    E += -p[0]+p[1]
                elif wpp[1] == "Y":
                    E += -p[0]+p[1] # check
                else: #Z or Identity
                    E += p[0]-p[1]

            return E
        
        if evaluate:
            return expectation(init_params)
        args = [expectation, init_params]
        return self.minimizer(*args, **self.minimizer_kwargs)
    
###################################################################################################################
    
import math

def check_hermitian(h):
    adjoint = h.conj().T # a.k.a. conjugate-transpose, transjugate, dagger
    return np.array_equal(h,adjoint)

def matrixify(n_qubits,wsopp):
    """
        wsopp: Weighted Sum-of-Product of Paulis
    """
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,complex(0,-1)],[complex(0,1),0]])
    Z = np.array([[1,0],[0,-1]])
    hamiltonian = np.zeros([2**n_qubits,2**n_qubits])
    for wpp in wsopp: # currently only supported for single qubit
        if wpp[1] == "X":
            hamiltonian += wpp[0]*X
        elif wpp[1] == "Y":
            hamiltonian += wpp[0]*Y
        elif wpp[1] == "Z":
            hamiltonian += wpp[0]*Z
        # else Identity
    assert(check_hermitian(hamiltonian))
    return hamiltonian
    
# Example 1: Single Qubit Hamiltonians

n_qubits = 1

wsopp = [] # {weight | pauli | qubit}
wsopp.append((1,"Z",0)) # sZ 
wsopp.append((1,"X",0)) # sX

ansatz = [] # qasm tokens
depth = 1
ansatz.append(("ry",0))

init_params = np.random.uniform(0.0, 2*np.pi, size=n_qubits)

hamiltonian = matrixify(n_qubits, wsopp)
w, v = np.linalg.eig(hamiltonian)
# print(min(w))
# Forest Answer: {'x': array([3.54943635]), 'fun': -0.7615739924772624}
# init_params = [3.54943635]

# Run Variational Quantum Eigensolver

v = VQE()

print("wsopp = ",wsopp,"\nmin EV = ",min(w),"\nInit_Params = ",init_params,"\nInit_Exp = ",v.vqe_run(wsopp,ansatz,n_qubits,depth,init_params,evaluate=True)) 
r = v.vqe_run(wsopp,ansatz,n_qubits,depth,init_params,evaluate=False)
print(r.status, r.fun, r.x)