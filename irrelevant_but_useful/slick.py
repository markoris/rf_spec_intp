# SLICK: Statistical and machine Learning Interpolator for light Curves emitted by Kilonovae

class Interpolator(object):

	def __init__(self):
		return

	def load_data(self, path, bands, n_iters=191, angle_avg=False, load_al=False, path_al='', shorten=True):

		import numpy as np
		import glob

		self.bands = bands
		self.n_iters = n_iters
		self.angle_avg = angle_avg
		self.shorten = shorten

		events = glob.glob(path)

		events = sorted(events)
		
		all_mags = np.array([])
		all_params = np.array([])

		for event in events:

			if '2020-02-07' in event: continue # weird restricted access files
			if '2020-07-02' in event: continue # weird restricted access files

			x = np.loadtxt(event)
		
			x = np.array(np.split(x, len(bands))) # three-dimensional array, first dimension is the band, 2nd dimension is the iteration, 3rd dimension is the magnitude in the k-th bin
			
			iterations = x[0, :n_iters, 0]

			times = x[0, :n_iters, 1]

			mags = np.array([])

			for idx in range(len(bands)):
				try:	
					dat = x[idx, :n_iters, 2:]
					dat = dat[None, :]
					mags = np.append(mags, dat, axis=0)
				except ValueError:
					mags = x[idx, :n_iters, 2:]
					mags = mags[None, :]

			try:
				mags = mags[None, :]
				all_mags = np.append(all_mags, mags, axis=0)
			except ValueError:
				all_mags = mags

			name_parse = event.split('/')[-1].split('_')

			params = np.array([name_parse[1], name_parse[5][-1], name_parse[7][2:], name_parse[8][2:], name_parse[9][2:], name_parse[10][2:]]) # morphology, wind, dynamical mass, dynamical velocity, wind mass, wind velocity
			
			try:
				all_params = np.append(all_params, params[None, :], axis=0)
			except ValueError:
				all_params = params[None, :]

		if all_mags.shape[0] == 0: return [], [], []

		if self.angle_avg == True:
			all_mags = np.mean(all_mags, axis=3)
		else:
			all_mags = all_mags[:, :, :, 0]#np.random.choice(54)] # randomly chooses one of the angular bins

		if shorten:
			all_params = all_params[:, 2:].astype('float')

		if load_al:
			_, all_params_al, all_mags_al = self.load_data(path_al, bands, self.n_iters, self.angle_avg, False, shorten=self.shorten)
			all_mags = np.concatenate((all_mags, all_mags_al), axis=0)
			all_params = np.concatenate((all_params, all_params_al), axis=0)

		self.path = path
		self.path_al = path_al
		self.load_al = load_al

		return times, all_params, all_mags

	def train(self, full_params, full_data, GP_restarts=5, RF_estimators=250, color_index='g-r', bol=False):

		import sys
		sys.path.append('../')
		import numpy as np
		from interpolators_v2 import Gaussian_Process, Random_Forest, Bagging_Regressor

		#if bol:
		self.full_data = full_data

		#else:
		#	band1 = self.bands.index(color_index[0])
		#	band2 = self.bands.index(color_index[-1])
		#	self.full_data = full_data[:, band1, :] - full_data[:, band2, :]

		self.full_params = full_params

		self.mean = np.mean(self.full_data)
		self.std = np.std(self.full_data)

		self.full_data -= self.mean
		self.full_data /= self.std

		#self.GP = Gaussian_Process(self.full_params, self.full_data)
		self.RF = Random_Forest(self.full_params, self.full_data)
		#self.BR = Bagging_Regressor(self.full_params, self.full_data)

		#self.GP.train(n_restarts_optimizer=GP_restarts)	
		self.RF.train(n_estimators=RF_estimators)
		#self.BR.train()

	def evaluate(self, input_params, return_std=True):

		import numpy as np
		import sys

		try:
			check_training = np.where((self.full_params[:, 0] == input_params[:, 0]) & (self.full_params[:, 1] == input_params[:, 1]) & (self.full_params[:, 2] == input_params[:, 2]) & 
				(self.full_params[:, 3] == input_params[:, 3]))[0]

			if check_training.shape[0] > 0:
				print ("Input found in training set, returned respective output.")
				return self.full_data[check_training].flatten(), self.full_data[check_training].flatten(), self.full_data[check_training].flatten(), 0
				sys.exit()

			#GP_out, GP_err = self.GP.evaluate(input_params)
			RF_out = self.RF.evaluate(input_params)
			#BR_out = self.BR.evaluate(input_params)

			#GP_out += self.mean
			#GP_out *= self.std

			RF_out *= self.std
			RF_out += self.mean

			#BR_out += self.mean
			#BR_out *= self.std

			#return GP_out, RF_out, BR_out, GP_err
			return RF_out

		except AttributeError:
			print ("Training not performed prior to evaluate call - please use train() first.")
			sys.exit()

