from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *

qvm = api.QVMConnection()
def smallish_ansatz(params):
    return Program(RX(params[0], 0), RX(params[0], 1), CNOT(0,1), H(0), CNOT(1,0), H(1))

print(smallish_ansatz([1.0, 2.0]))


from pyquil.paulis import *
initial_angle = [0.0]
# Our Hamiltonian is just \sigma_z on the zeroth qubit
hamiltonian = 0.5 *(sI(0) * sI(1)) + 0.5 * (sZ(0) * sZ(1)) - 0.5 * (sX(0) * sX(1)) - 0.5 * (sY(0) * sY(1))

print("HAM: ", hamiltonian)
from grove.pyvqe.vqe import VQE
from scipy.optimize import minimize
import numpy as np

vqe_inst = VQE(minimizer=minimize,
               minimizer_kwargs={'method': 'nelder-mead'})

# angle = 2.0
# print("RES:", vqe_inst.expectation(smallish_ansatz([angle]), hamiltonian, None, qvm))


angle_range = np.linspace(0.0, 2 * np.pi, 20)
data = [vqe_inst.expectation(smallish_ansatz([angle]), hamiltonian, None, qvm)
        for angle in angle_range]

import matplotlib.pyplot as plt
plt.xlabel('Angle [radians]')
plt.ylabel('Expectation value')
plt.plot(angle_range, data)
plt.show()