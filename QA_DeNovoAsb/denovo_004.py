"""
Take Q-matrix from De novo TSP example and solve the QUBO and Ising model using dimod exact solver

Expected output:

{'n0t0': 1, 'n0t1': 0, 'n0t2': 0, 'n0t3': 0, 'n1t0': 0, 'n1t1': 1, 'n1t2': 0, 'n1t3': 0, 'n2t0': 0, 'n2t1': 0, 'n2t2': 1, 'n2t3': 0, 'n3t0': 0, 'n3t1': 0, 'n3t2': 0, 'n3t3': 1}
{'n0t0': 0, 'n0t1': 1, 'n0t2': 0, 'n0t3': 0, 'n1t0': 0, 'n1t1': 0, 'n1t2': 1, 'n1t3': 0, 'n2t0': 0, 'n2t1': 0, 'n2t2': 0, 'n2t3': 1, 'n3t0': 1, 'n3t1': 0, 'n3t2': 0, 'n3t3': 0}
{'n0t0': 0, 'n0t1': 0, 'n0t2': 1, 'n0t3': 0, 'n1t0': 0, 'n1t1': 0, 'n1t2': 0, 'n1t3': 1, 'n2t0': 1, 'n2t1': 0, 'n2t2': 0, 'n2t3': 0, 'n3t0': 0, 'n3t1': 1, 'n3t2': 0, 'n3t3': 0}
{'n0t0': 0, 'n0t1': 0, 'n0t2': 0, 'n0t3': 1, 'n1t0': 1, 'n1t1': 0, 'n1t2': 0, 'n1t3': 0, 'n2t0': 0, 'n2t1': 1, 'n2t2': 0, 'n2t3': 0, 'n3t0': 0, 'n3t1': 0, 'n3t2': 1, 'n3t3': 0}

{'n0t0': 1, 'n0t1': -1, 'n0t2': -1, 'n0t3': -1, 'n1t0': -1, 'n1t1': 1, 'n1t2': -1, 'n1t3': -1, 'n2t0': -1, 'n2t1': -1, 'n2t2': 1, 'n2t3': -1, 'n3t0': -1, 'n3t1': -1, 'n3t2': -1, 'n3t3': 1}
{'n0t0': -1, 'n0t1': 1, 'n0t2': -1, 'n0t3': -1, 'n1t0': -1, 'n1t1': -1, 'n1t2': 1, 'n1t3': -1, 'n2t0': -1, 'n2t1': -1, 'n2t2': -1, 'n2t3': 1, 'n3t0': 1, 'n3t1': -1, 'n3t2': -1, 'n3t3': -1}
{'n0t0': -1, 'n0t1': -1, 'n0t2': 1, 'n0t3': -1, 'n1t0': -1, 'n1t1': -1, 'n1t2': -1, 'n1t3': 1, 'n2t0': 1, 'n2t1': -1, 'n2t2': -1, 'n2t3': -1, 'n3t0': -1, 'n3t1': 1, 'n3t2': -1, 'n3t3': -1}
{'n0t0': -1, 'n0t1': -1, 'n0t2': -1, 'n0t3': 1, 'n1t0': 1, 'n1t1': -1, 'n1t2': -1, 'n1t3': -1, 'n2t0': -1, 'n2t1': 1, 'n2t2': -1, 'n2t3': -1, 'n3t0': -1, 'n3t1': -1, 'n3t2': 1, 'n3t3': -1}
"""

import dimod
solver = dimod.ExactSolver()

Q_matrix = [
[  0, 13, 13, 13, 13,-14,  0,  0,  0, -8,  0,  0, 13, -2,  0,  0],
[ 13,  0, 13, 13,  0, 13,-14,  0,  0,  0, -8,  0,  0, 13, -2,  0],
[ 13, 13,  0, 13,  0,  0, 13,-14,  0,  0,  0, -8,  0,  0, 13, -2],
[ 13, 13, 13,  0,-14,  0,  0, 13, -8,  0,  0,  0, -2,  0,  0, 13],
[ 13, -6,  0,  0,  0, 13, 13, 13, 13,-14,  0,  0,  0, -8,  0,  0],
[  0, 13, -6,  0, 13,  0, 13, 13,  0, 13,-14,  0,  0,  0, -8,  0],
[  0,  0, 13, -6, 13, 13,  0, 13,  0,  0, 13,-14,  0,  0,  0, -8],
[ -6,  0,  0, 13, 13, 13, 13,  0,-14,  0,  0, 13, -8,  0,  0,  0],
[  0,-12,  0,  0, 13, -6,  0,  0,  0, 13, 13, 13, 13,-14,  0,  0],
[  0,  0,-12,  0,  0, 13, -6,  0, 13,  0, 13, 13,  0, 13,-14,  0],
[  0,  0,  0,-12,  0,  0, 13, -6, 13, 13,  0, 13,  0,  0, 13,-14],
[-12,  0,  0,  0, -6,  0,  0, 13, 13, 13, 13,  0,-14,  0,  0, 13],
[ 13,-18,  0,  0,  0,-12,  0,  0, 13, -6,  0,  0,  0, 13, 13, 13],
[  0, 13,-18,  0,  0,  0,-12,  0,  0, 13, -6,  0, 13,  0, 13, 13],
[  0,  0, 13,-18,  0,  0,  0,-12,  0,  0, 13, -6, 13, 13,  0, 13],
[-18,  0,  0, 13,-12,  0,  0,  0, -6,  0,  0, 13, 13, 13, 13,  0]]

Q = {}
for i in range(0,16):
	ni = 'n'+str(int(i/4))+'t'+str(int(i%4))
	for j in range(0,16):
		nj = 'n'+str(int(j/4))+'t'+str(int(j%4))
		if Q_matrix[i][j] != 0:
			Q[(ni,nj)] = Q_matrix[i][j]

response = solver.sample_qubo(Q)

minE = min(response.data(['sample', 'energy']), key=lambda x: x[1])
for sample, energy in response.data(['sample', 'energy']): 
	if energy == minE[1]:
		print(sample)

hii, Jij, offset = dimod.qubo_to_ising(Q)

response = solver.sample_ising(hii,Jij)

print()
minE = min(response.data(['sample', 'energy']), key=lambda x: x[1])
for sample, energy in response.data(['sample', 'energy']): 
	if energy == minE[1]:
		print(sample)

import matplotlib.pyplot as plt
y = []
for sample, energy in response.data(['sample', 'energy']): y.append(energy)
plt.plot(y)
plt.xlabel('Solution landscape')
plt.ylabel('Energy')
plt.show()