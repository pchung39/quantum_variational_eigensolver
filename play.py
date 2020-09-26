import scipy.optimize as optimize
import numpy as np
from vqe import VQE
from skquant.opt import minimize
import matplotlib.pyplot as plt
from qiskit.aqua.components.optimizers import COBYLA, L_BFGS_B, NELDER_MEAD


# let's test our optimisation method on a 2d paraboloid
#  centered at (0.5, 2)
def objective_function(x):
    """
    v - 1d, length 2 array representing a point in the domain of the paraboloid
    """
    # return (x[0]-0.5)**2 + (x[1]-2.)**2
    # print("X: ", x)
    return (x[0]) **2

# print(np.random.rand(2) * 10)


# result = optimize.minimize(objective_function,
#                         np.array([-np.pi, np.pi], dtype=float), # randomly select initial point 
#                         method='COBYLA',
#                         tol=1e-3) # desired error tolerance tells our search when to stop
# print(result)

# print(f"Result {result.x} found in {result.nit} steps")

# method can be ImFil, SnobFit, Orbit, or Bobyqa
# bounds = np.array([-np.pi, np.pi], dtype=float)
# x0 = np.array([-np.pi])
# budget = 100
# result, history = minimize(objective_function, x0, bounds, budget, method='imfil')
# print(result)
# print(history)

# cob = COBYLA(disp=True, tol=1e-3)
# results = cob.optimize(num_vars=1, objective_function=objective_function,gradient_function=None, variable_bounds=None, initial_point=x0)
# print("COBYLA: ", results)


# bfgs = L_BFGS_B()
# bfgs.optimize(num_vars=1, objective_function=objective_function,gradient_function=None, variable_bounds=None, initial_point=x0)
# print("BFGS: ", results)


# opt = NELDER_MEAD(disp=True)
steps = 50

points_costs_adam_optimizer = []
obj = VQE(angle_range=np.linspace(0, 2 * np.pi, 20))
bounds = np.array([[0, np.pi]], dtype=float)
budget = 100
initial_point = np.array([0], dtype=float)
result, history = minimize(obj.get_energy_state, initial_point, bounds, budget, method='nomad')

print("RES: ", result)
print("Hist: ", history)

radians = []
energy = []

for h in history:
    energy.append(h[0])
    radians.append(h[1])

plt.xlabel('Angle [radians]')
plt.ylabel('Expectation value')
plt.title("Snobfit results")
plt.plot(radians, energy)
plt.show()