class VQE(object):
    
    def __init__(self):
        self.minimizer = minimize
        self.minimizer_kwargs = {'method':'Nelder-Mead', 'options':{'ftol':1.0e-2, 'xtol':1.0e-2, 'disp':True}}
    

    
    def vqe_run(self,hamiltonian,ansatz,n_qubits,depth,init_params):
        
        p_name = "test_output/vqe_run.qasm"
        
        def objective_func(params):
            qasmify(params)           
            return expectation(hamiltonian)
        
        def qasmify(params):
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
            for i in range(n_qubits):
                prog.write("measure q"+str(i)+"\n")
            prog.close()
            
        def expectation(h):
            # We will not use the wavefunction (display command) as is not possible in a real QC
            # E = <wf|H|wf> = real(dot(transjugate(wf),dot(H,wf))) 
            qx = qxelarator.QX()
            qx.set(p_name)
            shots = 1000
            p = np.zeros(2**n_qubits)
            c = np.zeros(n_qubits,dtype=bool)
            for i in range(shots):
                qx.execute()
                for i in range(n_qubits):
                    c[i] = qx.get_measurement_outcome(i)
                idx = sum(v<<i for i, v in enumerate(c[::-1]))    
                p[idx] += 1/shots
            E = np.dot(p.T,np.dot(h,p))
            return E
            
        args = [objective_func, init_params]
        return self.minimizer(*args, **self.minimizer_kwargs)