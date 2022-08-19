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

	def __init__(self):
		return

	def train(self, inputs, targets, n_estimators=1000, max_depth=None):

		from sklearn.ensemble import RandomForestRegressor
		
		regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

		self.rfr = regressor.fit(inputs, targets)

	def evaluate(self, inputs):
		
		output = self.rfr.predict(inputs)

		return output

class NN(object):

	class Net()
	
		def __init__(self):
			return

		def forward(self, x):
			return x

	def __init__(self):
		return

	def set_separation(self):
		return

	def train(self):
		return

	def evaluate(self, inputs):
		return
