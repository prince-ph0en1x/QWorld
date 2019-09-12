import dimod
solver = dimod.ExactSolver()

# Hf = - 0.5*n0t0 + 1.0*n0t1 - 1.0*n0t0*n0t1
f = open("test_install.qubo", "r")
qubo_header = f.readline().split()
hii = {}
Jij = {}
Q = {}
for i in range(0,int(qubo_header[4])):
	x = f.readline().split()
	hii[x[0]] = float(x[2])	
	Q[(x[0],x[1])] = float(x[2])	
for i in range(0,int(qubo_header[5])):
	x = f.readline().split()
	Jij[(x[0],x[1])] = float(x[2])
	Q[(x[0],x[1])] = float(x[2])
f.close()

# print(hii, Jij)
response = solver.sample_ising(hii, Jij)
for sample, energy in response.data(['sample', 'energy']): print(sample, energy)
'''
{'n0t0': -1, 'n0t1': -1} -1.5
{'n0t0': 1, 'n0t1': -1} -0.5
{'n0t0': 1, 'n0t1': 1} -0.5
{'n0t0': -1, 'n0t1': 1} 2.5
'''

# print(Q)
response = solver.sample_qubo(Q)
for sample, energy in response.data(['sample', 'energy']): print(sample, energy)
'''
{'n0t0': 1, 'n0t1': 0} -0.5
{'n0t0': 1, 'n0t1': 1} -0.5
{'n0t0': 0, 'n0t1': 0} 0.0
{'n0t0': 0, 'n0t1': 1} 1.0
'''