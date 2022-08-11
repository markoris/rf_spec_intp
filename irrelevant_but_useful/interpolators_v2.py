import numpy as np

class Interpolator(object):
	
	def __init__(self, inputs, targets, keep_test_event=False, wrap_log=False):

		self.wrap_log = wrap_log

		#np.random.seed(22)

		bands = 'grizyJHKS'

		if keep_test_event:
			idx = np.random.choice(inputs.shape[0])
			self.test_input = inputs[idx]
			self.test_target = targets[idx]
			inputs = np.delete(inputs, idx, axis=0)
			targets = np.delete(targets, idx, axis=0)

		self.inputs, self.targets = inputs, targets
				
class Linear_IDW(Interpolator):

	def evaluate(self, inputs=None, targets=None):

		if inputs is None or targets is None:
			inputs, targets = self.inputs, self.targets

		weights = np.zeros(targets.shape[0])
		
		for event in xrange(targets.shape[0]):
			weights[event] = np.sqrt((self.test_input[0] - inputs[event, 0])**2 + (self.test_input[1] - inputs[event, 1])**2)

		self.weights = np.abs(1./weights**(inputs.shape[1]+1))


		interpolated = np.zeros(targets.shape[1])
		interpolated2 = np.zeros(targets.shape[1])

		for point in xrange(self.targets.shape[1]):
			interp_point = np.sum(self.weights*self.targets[:, point])/np.sum(self.weights) + (np.random.random(1)-0.5)*0.2
			interpolated[point] = interp_point

		if self.wrap_log:
			interpolated = np.power(10, interpolated)

		return interpolated

class Log_Linear_IDW(Interpolator):

	def evaluate(self, inputs=None, targets=None):

		if inputs is None or targets is None:
			inputs, targets = self.inputs, self.targets

		weights = np.zeros(targets.shape[0])
		
		for event in xrange(targets.shape[0]):
			weights[event] = (self.test_input[0] - inputs[event, 0])**2 + (self.test_input[1] - inputs[event, 1])**2

		self.weights = 1./np.log(np.abs(weights))

		interpolated = np.zeros(targets.shape[1])
		interpolated2 = np.zeros(targets.shape[1])

		for point in xrange(self.targets.shape[1]):
			interp_point = np.sum(self.weights*self.targets[:, point])/np.sum(self.weights) + (np.random.random(1)-0.5)*0.2
			interpolated[point] = interp_point 

		return interpolated

class Gaussian_Process(Interpolator):

	def train(self, inputs=None, targets=None, n_restarts_optimizer=5, normalize_y=False):

		from sklearn import gaussian_process
		from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel as C

		if inputs is None or targets is None:
			inputs, targets = self.inputs, self.targets
	
		#length_scale = np.array([1, 1e-3])
		#length_scale_bounds = np.array([(1e-3, 1e-1), (0.1, 0.1)])

		length_scale = np.std(inputs, axis=0)
                #length_scale_bounds = [(1e-3, k) for k in np.max(np.abs(inputs), axis=0)]
		length_scale_bounds = [(j, k) for j, k in zip(np.min(np.abs(inputs), axis=0), np.max(np.abs(inputs), axis=0))]
                
                #print length_scale, length_scale_bounds
		
		kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1)) + C(0.1, (1e-3, 1e1)) * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
		
		#print kernel.theta
		
		#kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1)) + C(0.5, (1e-3, 1e1)) * ExpSineSquared(length_scale=5e-2, length_scale_bounds=(1e-3, 3e-1))

		#kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)

		self.gpr = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y).fit(inputs, targets)

	def evaluate(self, inputs=None, return_std=True):

		if inputs is None:
			try:
				inputs = self.test_input[np.newaxis, :]
			except AttributeError:
				print ('Test input does not exist; pass inputs to evaluate function or change keep_test_event to True.')

		output, std = self.gpr.predict(inputs, return_std=return_std)
		output = output.flatten()
                
		if self.wrap_log:
			return np.power(10,output)

		return output, std

class Random_Forest(Interpolator):

	def train(self, inputs=None, targets=None, n_estimators=1000, max_depth=None):

		from sklearn.ensemble import RandomForestRegressor
		
		if inputs is None or targets is None:
			inputs, targets = self.inputs, self.targets

		regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

		self.rfr = regressor.fit(inputs, targets)

	def evaluate(self, inputs=None):

		if inputs is None:
			inputs = self.test_input[np.newaxis, :]

		output = self.rfr.predict(inputs).flatten()

		if self.wrap_log:
			output = np.power(10, output)

		return output
		
class Bagging_Regressor(Interpolator):

	def train(self, inputs=None, targets=None):
	
		from sklearn.ensemble import BaggingRegressor
		from sklearn.tree import ExtraTreeRegressor
	
		if inputs is None or targets is None:
			inputs, targets = self.inputs, self.targets
		
		extra_tree = ExtraTreeRegressor()
		self.br = BaggingRegressor(extra_tree)
		self.br.fit(inputs, targets)
		
	def evaluate(self, inputs=None):
	
		if inputs is None:
			inputs = self.test_input[np.newaxis, :]
			
		output = self.br.predict(inputs).flatten()
		
		return output
