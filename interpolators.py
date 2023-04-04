import numpy as np

class GP(object):

	def __init__(self):
		return

	def train(self, inputs, targets, nro=1):

		from sklearn import gaussian_process
		from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

		length_scale = np.std(inputs, axis=0)
		length_scale_bounds = [(j, k) for j, k in zip(np.min(np.abs(inputs), axis=0), np.max(np.abs(inputs), axis=0))]
                
		kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1)) + C(0.1, (1e-3, 1e1)) * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
		
		self.gpr = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=nro).fit(inputs, targets)

	def evaluate(self, inputs, return_std=True):

		output, std = self.gpr.predict(inputs, return_std=return_std)

		return output, std

class RF(object):

	def __init__(self, n_estimators=250, max_depth=1000, max_features=None, max_leaf_nodes=500):
		
        # Random Search Cross Validation found these to be ideal parameters when using *ALL* time/angle/wavelength data
        # RandomForestRegressor(max_depth=1000, max_features=None, max_leaf_nodes=500,
        #              n_estimators=250)

		from sklearn.ensemble import RandomForestRegressor
		
		self.rfr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, max_leaf_nodes=max_leaf_nodes)

	def train(self, inputs, targets):

		self.rfr = self.rfr.fit(inputs, targets)

	def evaluate(self, inputs):
		
		output = self.rfr.predict(inputs)

		return output

class NN(object):

	class Net():
	
		def __init__(self):
			return

		def forward(self, x):
			return x

	def __init__(self, learning_rate_init, max_iter, solver, activation, learning_rate):
		from sklearn.neural_network import MLPRegressor

		self.nnr = MLPRegressor(learning_rate_init=learning_rate_init, max_iter=max_iter)
		
		return

	def set_separation(self, inputs, targets):
		return 

	def train(self, inputs, targets):
		self.nnr = self.nnr.fit(inputs, targets)
		return

	def evaluate(self, inputs):
		output = self.nnr.predict(inputs)
		return output
