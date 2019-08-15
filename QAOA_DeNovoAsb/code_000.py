# Test qxelerator issues

from qxelarator import qxelarator
import numpy as np

def test_01():
	p_name = "test_output/multi_run.qasm"
	n_qubits = 3
	shots = 400

	qx = qxelarator.QX()
	for p_loop in range(0,40):
		# qx = qxelarator.QX()	# Inside loop cause problems
		qx.set(p_name)
		p = np.zeros(2**n_qubits)
		c = np.zeros(n_qubits,dtype=bool)
		for i in range(shots):
			print(p_loop,i)
			# qx.execute(1)
			qx.execute() # Both works for p_loop 40 x shots 400
			for i in range(n_qubits):
				c[i] = qx.get_measurement_outcome(i)
			idx = sum(v<<i for i, v in enumerate(c[::-1]))    
			p[idx] += 1/shots

	print(p)

# test_01()

def test_02():
	p_name = "test_output/multi_run_02.qasm" # Works with/without kernel encapsulation
	qx = qxelarator.QX()
	for p_loop in range(0,6):
		qx.set(p_name)
		# qx.execute(1) # Doesn't work with get_state()
		qx.execute() 
		isv = qx.get_state()
		print(isv)

test_02()