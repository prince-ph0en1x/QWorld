"""
Connect to D-Wave and solve De novo TSP
"""




import numpy as np
import dimod
import matplotlib.pyplot as plt
import dwave_networkx as dnx
import networkx as nx
import minorminer

"""
Reads to TSP graph
"""

# Overlap between pair-wise reads
def pwalign(read1,read2,mm):
	l1 = len(read1)
	l2 = len(read2)
	for shift in range(l1-l2,l1):
		mmr = 0
		r2i = 0
		for r1i in range(shift,l1):
			if read1[r1i]!=read2[r2i]:
				mmr += 1
			r2i += 1
			if mmr > mm:
				break
		if mmr <= mm:
			return l2-shift
	return 0

reads = ['ATGGCGTGCA','GCGTGCAATG','TGCAATGGCG','AATGGCGTGC']
n_reads = len(reads)
allowed_mismatched = 0

O_matrix = np.zeros((n_reads,n_reads))
for r1 in range(0,n_reads):
	for r2 in range(0,n_reads):
		if r1!=r2:
			O_matrix[r1][r2] = pwalign(reads[r1],reads[r2],allowed_mismatched) # edge directivity = (row id, col id)
# print(O_matrix)

"""
TSP Graph to Q-Matrix
"""

# Qubit index semantics: {c0t0,c0t1,c0t2,c0t3,c1t0,c1t1,c1t2,c1t3,c2t0,c2t1,c2t2,c2t3,c3t0,c3t1,c3t2,c3t3}

# Initialize
Q_matrix = np.zeros((n_reads**2,n_reads**2))

# Assignment reward (self-bias)
p0 = 0
for ct in range(0,n_reads**2):
	Q_matrix[ct][ct] += p0

# Multi-location penalty
p1 = 4 # fixed emperically by trail-and-error
for c in range(0,n_reads):
	for t1 in range(0,n_reads):
		for t2 in range(0,n_reads):
			if t1!=t2:
				Q_matrix[c*n_reads+t1][c*n_reads+t2] += p1

# Visit Repetation penalty
p2 = p1
for t in range(0,n_reads):
	for c1 in range(0,n_reads):
		for c2 in range(0,n_reads):
			if c1!=c2:
				Q_matrix[c1*n_reads+t][c2*n_reads+t] += p2

# Path cost
# kron of O_matrix and a shifted diagonal matrix
for ci in range(0,n_reads):
	for cj in range(0,n_reads):
		for ti in range(0,n_reads):
			tj = (ti+1)%n_reads
			Q_matrix[ci*n_reads+ti][cj*n_reads+tj] += -O_matrix[ci][cj]

# print(Q_matrix)

"""
Solve the QUBO using dimod exact solver
"""

solver = dimod.ExactSolver()

Q = {}
for i in range(0,n_reads**2):
	ni = 'n'+str(int(i/n_reads))+'t'+str(int(i%n_reads))
	for j in range(0,n_reads**2):
		nj = 'n'+str(int(j/n_reads))+'t'+str(int(j%n_reads))
		if Q_matrix[i][j] != 0:
			Q[(ni,nj)] = Q_matrix[i][j]

# response = solver.sample_qubo(Q)

# minE = min(response.data(['sample', 'energy']), key=lambda x: x[1])
# for sample, energy in response.data(['sample', 'energy']): 
# 	if energy == minE[1]:
# 		print(sample)

"""
Solve the Ising using dimod exact solver
"""

hii, Jij, offset = dimod.qubo_to_ising(Q)

# response = solver.sample_ising(hii,Jij)
# # print()
# minE = min(response.data(['sample', 'energy']), key=lambda x: x[1])
# for sample, energy in response.data(['sample', 'energy']): 
# 	if energy == minE[1]:
# 		print(sample,energy)

# y = []
# for sample, energy in response.data(['sample', 'energy']): y.append(energy)
# plt.plot(y)
# plt.xlabel('Solution landscape')
# plt.ylabel('Energy')
# plt.show()

"""
Embed the QUBO graph in Chimera graph
"""

connectivity_structure = dnx.chimera_graph(3,3) # try to minimize
G = nx.from_numpy_matrix(Q_matrix)

max_chain_length = 0
while(max_chain_length == 0):
	embedded_graph = minorminer.find_embedding(G.edges(), connectivity_structure)
	for _, chain in embedded_graph.items():
	    if len(chain) > max_chain_length:
	        max_chain_length = len(chain)
# Display maximum chain length and Chimera embedding
print("max_chain_length",max_chain_length) # try to minimize
# dnx.draw_chimera_embedding(connectivity_structure, embedded_graph)
# plt.show()


"""
TRY
"""





"""
Solve the embedded Ising using D-Wave solver
"""

import random
from dwave.cloud import Client
from dwave.embedding import embed_ising, unembed_sampleset #, edgelist_to_adjacency
from dwave.embedding.utils import edgelist_to_adjacency

config_file='/media/sf_QWorld/QWorld/QA_DeNovoAsb/dwcloud.conf'

client = Client.from_config(config_file, profile='aritra')
solver = client.get_solver() # Available QPUs: DW_2000Q_2_1 (2038 qubits), DW_2000Q_5 (2030 qubits)

edgelist = solver.edges

adjdict = edgelist_to_adjacency(edgelist)
# https://github.com/dwavesystems/dwave-system/blob/master/dwave/embedding/utils.py DOESNT WORK
# adjdict = dict()
# for u, v in edgelist:
#     if u in adjdict:
#         adjdict[u].add(v)
#     else:
#         adjdict[u] = {v}
#     if v in adjdict:
#         adjdict[v].add(u)
#     else:
#         adjdict[v] = {u}

embed = minorminer.find_embedding(Jij.keys(),edgelist)
[h_qpu, j_qpu] = embed_ising(hii, Jij, embed, adjdict)

response_qpt = solver.sample_ising(h_qpu, j_qpu, num_reads=1)
response_qpt.wait() # print(response_qpt.done())
client.close()

res = response_qpt.result()
# print(res)

bqm = dimod.BinaryQuadraticModel.from_ising(hii, Jij)
print(embed)

variables = list(bqm) 
print(variables)
record = response_qpt.record
chains = [embed[v] for v in variables]
print(chains)

# print(bqm)

# unembedded_res = unembed_sampleset(response_qpt, embed, bqm, chain_break_method ='majority_vote')
# print(unembedded_res)

	# for sam in range(len(res['samples'])):
	# 	print((res['samples'][sam][u],res['samples'][sam][v],res['energies'][sam],res['occurrences'][sam]))

	# response_qpt.cancel()