import re      

template = open(t_name,"r")
param_ctr = 0
for line in template:
    if re.search('\*',line):
        line_new = re.sub('\*',str(x[param_ctr]), line)
        param_ctr += 1
        prog.write(line_new)
    else:
        prog.write(line)
template.close()
prog.close() 




print(ansatz)
print(ansatz[1])
print(ansatz[1][0])
print(ansatz[1][1])
print(ansatz[1][1][0])
qubits 1
.QK1
    ry q0, *
    measure q0


###################################################################################################################
   

    def vqe_run(self, ansatz, h, x0):
        
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
            for line in template:
                if re.search('\*',line):
                    line_new = re.sub('\*',str(x[param_ctr]), line)
                    param_ctr += 1
                    prog.write(line_new)
                else:
                    prog.write(line)
            template.close()
            prog.close()     
            
        def expectation(h):
            # We will not use the wavefunction (display command) as is not possible in a real QC
            # E = <wf|H|wf> = real(dot(transjugate(wf),dot(H,wf))) 
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


import math

# LAST GOOD RESULT

h = np.array([[1,0],[0,-1]])
r = v.vqe_run("vqe",h,[math.pi/2]) # optimal angle = pi/2, optimal value = -1.0

Optimization terminated successfully.
         Current function value: -1.000000
         Iterations: 25
         Function evaluations: 66
0 -1.0 [3.14159265]

'''
# KNOWN ISSUES

* Does not work for initial angle of 0
* Does not work for H bigger than 2x2
'''