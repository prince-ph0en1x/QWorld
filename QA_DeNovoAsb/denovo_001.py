import dimod
solver = dimod.ExactSolver()
#response = solver.sample_ising({'n0t0': -0.5, 'n0t1': 1.0}, {('n0t0', 'n0t1'): -1})
#for sample, energy in response.data(['sample', 'energy']): print(sample, energy)

'''
Hf = - 0.5a + 1.0b - 1.0ab

{'a': -1, 'b': -1} -1.5
{'a': 1, 'b': -1} -0.5
{'a': 1, 'b': 1} -0.5
{'a': -1, 'b': 1} 2.5
'''

f = open("denovo_001.qubo", "r")
qubo_header = f.readline().split()
hii = {}
Jij = {}
for i in range(0,int(qubo_header[4])):
	x = f.readline().split()
	hii[x[0]] = float(x[2])	
for i in range(0,int(qubo_header[5])):
	x = f.readline().split()
	Jij[(x[0],x[1])] = float(x[2])
f.close()

print(hii)
print(Jij)

response = solver.sample_ising(hii, Jij)
for sample, energy in response.data(['sample', 'energy']): print(sample, energy)