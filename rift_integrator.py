import RIFT.integrators.MonteCarloEnsemble as monte_carlo_integrator
import numpy as np
import sys

# toy problem, integrate x^2 from -1 to 1

dim = 1
bounds = [-1, 1]
gmm_dict = {(0,): None}
ncomp = 1

# uniform prior 1/N
def prior(samples):
    return np.ones_like(samples)/samples.shape[0]

def residual_function(samples):
    return -0.5*(samples**2)

samples = np.random.uniform(-1, 1, 10000)

integrator = monte_carlo_integrator.integrator(dim, bounds, gmm_dict, ncomp, proc_count=None, use_lnL=False, user_func=sys.stdout.flush(), prior=prior)
integrator.integrate(residual_function, min_iter=20, max_iter=20, progress=True, epoch=5) 

