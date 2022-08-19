import numpy as np
import glob

class intp_rf(object):


	def __init__(self, seed=None, verbose=True, debugging=False):

		"""
		Sets up the Random Forest (RF) interpolator for interpolation of kilonova spectra in the LANL published format.

		path_to_sims_dir: 	string
				  	Path to directory containing simultions of spectra using LANL naming convention (as provided in https://zenodo.org/record/5745556)

		path_to_spectra_h5: 	string
				    	Path to .hdf5 file containing spectra matching those in path_to_sims_dir. See included convert_dat_to_h5.py file for proper conversion

		short_wavs: 		bool, default=True
					If true, uses only wavelengths relevant to the included grizyJHKS filters (0.4 to 10 microns)
		
		shorten:		bool, default=True
					If true, shortens the parameter inputs to exclude morphology/composition information and report only numerical ejecta parameters (M_d, v_d, M_w, v_w)

		verbose:		bool, default=False
					If true, prints detailed program information

		cutoff:			float, default=21
					Time in days to serve as the upper limit for spectra. This value is ceiling-limited by the choice of time cutoff used in convert_dat_to_h5.py
		"""

		self.verbose = verbose

		if not seed == None: np.random.seed(seed)

		### Potential interpolators

		self.intp = None
		self.mu_spec = None
		self.std_spec = None
		self.debugging = debugging

		return

	def load_data(self, path_to_sims_dir, path_to_spectra_h5, short_prms=True, short_wavs=True, t_max=38.055, theta=0, downsample_theta=False):

		import h5py, sys
			
		def get_params_from_filename(string):
			name_parse = string.split('/')[-1].split('_')
			params = np.array([name_parse[1], name_parse[5][-1], name_parse[7][2:], name_parse[8][2:], name_parse[9][2:], name_parse[10][2:]])
			return params

		self.t_max = t_max
		self.theta = theta

		### Prepare wavelength information	
	
		wavs_full = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # in microns (from 1e4 scaling factor)
		
		if short_wavs:
			self.wav_idxs = np.where((wavs_full>0.4) & (wavs_full<10))[0] # above .4 micron and below 10 microns
		else: 
			self.wav_idxs = np.arange(len(wavs_full))
		
		self.wavs = wavs_full[self.wav_idxs]

		### Prepare ejecta parameters to serve as interpolation inputs
	
		files = np.array(glob.glob(path_to_sims_dir))
		files.sort()
		if len(files) < 1: 
			print("There don't appear to be any spectra files in that directory! Exiting without loaded data.")
			sys.exit()
		for idx in range(files.shape[0]):
			params = get_params_from_filename(files[idx]) # parse filenames into parameters
			try: params_all = np.concatenate((params_all, params[None, :]), axis=0)
			except NameError: params_all = params[None, :]
		if short_prms: params_all = params_all[:, 2:].astype('float')
		if self.verbose: print("Parameter array shape: ", params_all.shape)

		### Prepare spectra to serve as interpolation targets

		### First choose the cutoff time for spectra
	
		t_a = 0.125 # spectral start time
		#t_b = 49.351 # spectral end time
		t_b = 20.749 # spectral end time FIXME this is only temporary, need to remake .hdf5 to include time up to 66 iteration
		# FIXME re-run the .hdf5 creation using longest time available to all spectra. then re-write self.times to fit that number
		self.times = np.logspace(np.log10(t_a), np.log10(t_b), 60) # this should cover the full spectral possibilities range
		#self.times = np.logspace(np.log10(t_a), np.log10(t_b), 70) # this should cover the full spectral possibilities range FIXME
		self.angles = np.degrees([np.arccos((1 - (i-1)*2/54.)) for i in np.arange(1, 55, 1)])
	
		if self.verbose: print("Spectra times: ", self.times)
	
		if self.t_max is not None:
			t_idx = np.argmin(np.abs(self.times-self.t_max))
			self.times = self.times[:t_idx+1]
			if self.verbose: print("Time index of %d corresponds to %f days" % (t_idx, self.times[t_idx]))

		if self.theta is not None:
			angle_idx = np.argmin(np.abs(self.angles-theta))
			if self.verbose: print("Angle index of %d corresponds to %f degrees" % (angle_idx, self.angles[angle_idx]))

		if self.verbose: print("Loading h5 file, this may take a while on the first execution")
	
		h5_load = h5py.File(path_to_spectra_h5, 'r')
		data = h5_load['spectra'][:]
		spec_all = data.reshape(files.shape[0], self.times.shape[0], wavs_full.shape[0], self.angles.shape[0])
		
		if self.verbose: print('Data loaded and reshaped')

		self.params = params_all
		self.spectra = spec_all

		if self.debugging:
			self.params = self.params[:20, ...]
			self.spectra = self.spectra[:20, ...]

		if self.t_max is None and theta is None: # does not specify time or angle index
			self.spectra = self.spectra[:, :, self.wav_idxs, :]
			if downsample_theta: 
				self.spectra = self.spectra[:, :, :, np.arange(0, self.angles.shape[0]-1, 2)]
				self.angles = np.degrees([np.arccos((1 - (i-1)*2/27.)) for i in np.arange(1, 28, 1)])
			return

		if self.t_max is None or theta is None: # one of angle or time unspecified

			if self.t_max is None:
				self.spectra = self.spectra[:, :, self.wav_idxs, angle_idx] # only angle specified, all times
				return

			if self.theta is None:
				self.spectra = self.spectra[:, t_idx, self.wav_idxs, :] # only time specified, all angles
				if downsample_theta: 
					self.spectra = self.spectra[:, :, np.arange(0, self.angles.shape[0]-1, 2)]
					self.angles = np.degrees([np.arccos((1 - (i-1)*2/27.)) for i in np.arange(1, 28, 1)])
				return

		self.spectra = self.spectra[:, t_idx, self.wav_idxs, angle_idx]	# time and angle specified
		return

	def append_input_parameter(self, values_to_append, axis_to_append):

		'''
		Reduces the target array by 1 dimension to treat the replaced dimension as an input training variable.
		Spectra by default have shape [N, time, wavelength, angle] where N is the number of simulations.
		To add time as a training parameter, set t_max=None and angle=0 in load_data(), for example.
		This yields self.spectra.shape = [N, time, wavelength].
		The values_to_append array will be the times corresponding to the time column (axis=1) of self.spectra.
		Thus, this function would be called as append_input_parameter(self, self.times, 1).
		This would then yield self.params = [Md, vd, Mw, vd, t], and self.spectra would have shape [N*time, wavelength].
		Therefore, the dimension of self.spectra is reduced by 1, and the dimension of self.params is increased by 1.
		'''

		import data_format_1d as df1d

		self.params, self.spectra = df1d.format(self.params, self.spectra, values_to_append, axis_to_append)

		print(self.params.shape, self.spectra.shape)

		return	

	def create_test_set(self, size=1):

		if self.params is None or self.spectra is None:
			print('No data loaded, run load_data() first!')
			return

		test_indices = np.random.choice(np.arange(self.params.shape[0]), size=size, replace=False)
		test_indices = np.sort(test_indices)

		if self.verbose: print('Test set parameters should be: ', self.params[test_indices])

		for idx in test_indices:

			param = self.params[idx]
			spec = self.spectra[idx]
			self.params = np.delete(self.params, idx, axis=0)
			self.spectra = np.delete(self.spectra, idx, axis=0)

			try:
				self.params_test = np.concatenate((self.params_test, param[None, :]), axis=0)
				self.spectra_test = np.concatenate((self.spectra_test, spec[None, :]), axis=0)
			except AttributeError:
				self.params_test = param[None, :]
				self.spectra_test = spec[None, :]

			test_indices -= 1

		if self.verbose: 
			print('Test set parameters are: ', self.params_test)
			print('Test set shape is: ', self.params_test.shape)
			print('Training set shape is ', self.params.shape)

			

		return
			
	def preprocess(self):

		if self.params is None or self.spectra is None:
			print('No data loaded, run load_data() first!')
			return

		if self.verbose: print("Preprocessing data for training")

		self.spectra[np.where(self.spectra <= 0)] = np.min(self.spectra[np.nonzero(self.spectra)])/10
		self.spectra = np.log10(self.spectra)

		self.mu_spec = np.mean(self.spectra, axis=0)
		self.std_spec = np.std(self.spectra, axis=0)
		self.std_spec[np.where(self.std_spec <= 0)] += 1e-10

		if self.verbose: print(self.mu_spec.shape, self.std_spec.shape)

		self.spectra -= self.mu_spec
		self.spectra /= self.std_spec

		### Convert to log-mass for more training-friendly inputs

		self.params[:, [0, 2]] = np.log10(self.params[:, [0, 2]])
		if self.t_max is None and self.theta is not None: self.params[:, 4] = np.log10(self.params[:, 4])
		if self.theta is None and self.t_max is not None: self.params[:, 4] = np.cos(np.radians(self.params[:, 4]))
		if self.t_max is None and self.theta is None: # FIXME add a way to differentiate which axis is time/angle to not assume time gets added as an input parameter prior to angle
			self.params[:, 4] = np.log10(self.params[:, 4])
			self.params[:, 5] = np.cos(np.radians(self.params[:, 5]))

		return

	def train(self, rf=False, gp=False, nn=False, nro=1, n_estimators=1000, max_depth=None):

		if self.mu_spec is None or self.std_spec is None:
			print("Run preprocess() before training!")
			return

		if np.any([rf, gp, nn]) == False:
			print("Pick an interpolator to train!")
			return

		import interpolators as intps # REWRITE INTERPOLATORS_V2 TO BE LESS GARBAGE
		
		intp_library = {'rf': intps.RF(),
				'gp': intps.GP(),
				'nn': intps.NN(), }

		self.intp_choice = np.array(['rf', 'gp', 'nn'])[[rf, gp, nn]][0]

		if self.verbose: print("Using %s interpolator" % self.intp_choice)
			
		self.intp = intp_library[self.intp_choice]

		if self.verbose: print("Starting training, this may take a while...")

		self.intp.train(self.params, self.spectra)

		# choose self.intp from dict/list based on kwarg to train()

		return

	def evaluate(self, inputs=None, ret_out=False):

		# add check to see if training has been run

		if self.intp == None:
			print("No interpolators trained! Please run train() first.")
			return

		if inputs is None:
			inputs = self.params_test

		if inputs.ndim < 2:
			inputs.reshape(1, -1)

		inputs[:, [0, 2]] = np.log10(inputs[:, [0, 2]])
		if self.t_max is None and self.theta is not None: inputs[:, 4] = np.log10(inputs[:, 4])
		if self.theta is None and self.t_max is not None: inputs[:, 4] = np.cos(np.radians(inputs[:, 4]))
		if self.t_max is None and self.theta is None: # FIXME same as in preprocess(), check data to not assume that time added as input parameter before angle
			inputs[:, 4] = np.log10(inputs[:, 4])
			inputs[:, 5] = np.log10(inputs[:, 5])

		self.prediction = self.intp.evaluate(inputs)  #make the actual interpolators interface

		self.prediction *= self.std_spec
		self.prediction += self.mu_spec

		self.prediction = 10**self.prediction

		if self.verbose: print('Prediction shape is ', self.prediction.shape)

		if ret_out == True:
			return self.prediction # if this exists, need to add an option to make_plots to plot external input

		return

	def make_plots(self, filter_bands=True,):

		import os
		import matplotlib.pyplot as plt
		from matplotlib import ticker
		import matplotlib.patheffects as PathEffects

		plt.rc('font', size=30)
		plt.rc('lines', lw=3)

		if not os.path.isdir('intp_figures'): os.mkdir('intp_figures')

		self.params_test[:, [0, 2]] = 10**self.params_test[:, [0,2]]
		if self.t_max is None and self.theta is not None: self.params_test[:, 4] = 10**self.params_test[:, 4]
		if self.theta is None and self.t_max is not None: self.params_test[:, 4] = np.degrees(np.arccos(self.params_test[:, 4]))
		if self.t_max is None and self.theta is None: # FIXME same as preprocess(), remove assumption that time added as input param before angle
			self.params_test[:, 4] = 10**self.params_test[:, 4]
			self.params_test[:, 5] = np.degrees(np.arccos(self.params_test[:, 5]))

		for idx in range(self.params_test.shape[0]):

			plt.figure(figsize=(19.2, 10.8))
			plt.subplots_adjust(right=0.95)
			
			if filter_bands:	
				filters = np.array(glob.glob('filters/*'))
				wavelengths = 'grizyJHKS'
				colors = {"g": "blue", "r": "cyan", "i": "lime", "z": "green", "y": "greenyellow", "J": "gold",
				         "H": "orange", "K": "red", "S": "darkred"}
				for fltr in range(len(filters)):
					filter_wavs = np.loadtxt(filters[fltr])
					filter_wavs = filter_wavs[:, 0]*1e-4 # factor of 1e-4 to go from Angstroms to microns
					text_loc = np.mean(filter_wavs)
					text_loc = (np.log10(text_loc)-np.log10(self.wavs.min()))/(np.log10(self.wavs.max())-np.log10(self.wavs.min()))
					wav_low, wav_upp = filter_wavs[0], filter_wavs[-1]
					fltr_band = filters[fltr].split('/')[-1][0]
					fltr_indx = wavelengths.find(fltr_band)
					plt.axvspan(wav_low, wav_upp, alpha=0.5, color=colors[wavelengths[fltr_indx]])
					#plt.text(text_loc_relative[fltr_indx], 1.015, fltr_band, color=colors[wavelengths[fltr_indx]], transform=plt.gca().transAxes, path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="black")])
					plt.text(text_loc, 1.015, fltr_band, color=colors[wavelengths[fltr_indx]], transform=plt.gca().transAxes, path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="black")])
			
			if self.t_max is not None and self.theta is not None: 
				plt.title(r"TP   wind2   $M_d$={0:.4f}   $v_d$={1:.3f}   $M_w$={2:.4f}   $v_w$={3:.3f}".format(*self.params_test[idx][0:4]), y=1.06)
			if self.t_max is None and self.theta is not None:
				plt.title(r"TP   wind2   $M_d$={0:.4f}   $v_d$={1:.3f}   $M_w$={2:.4f}   $v_w$={3:.3f}   $t$={4:.3f}".format(*self.params_test[idx][0:5]), y=1.06)
			if self.theta is None and self.t_max is not None:
				plt.title(r"TP   wind2   $M_d$={0:.4f}   $v_d$={1:.3f}   $M_w$={2:.4f}   $v_w$={3:.3f}   $\theta$={4:.3f}".format(*self.params_test[idx][0:5]), y=1.06)
			if self.theta is None and self.t_max is None:
				plt.title(r"TP  wind2  $M_d$={0:.4f}  $v_d$={1:.3f}  $M_w$={2:.4f}  $v_w$={3:.3f}  $t$={4:.3f}  $\theta$={5:.3f}".format(*self.params_test[idx][0:6]), y=1.06)
			plt.plot(self.wavs, self.spectra_test[idx], label='true', color='k')
			plt.plot(self.wavs, self.prediction[idx], label='intp', color='red')
			plt.xscale('log')
			plt.yscale('log')
			plt.gca().set_xticks([0.5, 1, 2, 3, 5, 10])
			plt.gca().get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
			plt.gca().set_ylim(bottom=1e-12)
			#plt.xlabel(r'$\lambda \ (\mu m)$')
			plt.ylabel(r'$F_{\lambda} (erg \ s^{-1} \ cm^{-2} \ \mu m^{-1})$')
			plt.xlabel(r'$\lambda$ ($\mu$m)')
			plt.legend()
			if self.t_max is not None and self.theta is not None:
				plt.savefig("intp_figures/{0}_TP_wind2_md{1:.4f}_vd{2:.3f}_mw{3:.4f}_vw{4:.3f}.pdf".format(self.intp_choice, *self.params_test[idx][0:4]))
			if self.t_max is None and self.theta is not None:
				plt.savefig("intp_figures/{0}_TP_wind2_md{1:.4f}_vd{2:.3f}_mw{3:.4f}_vw{4:.3f}_t{5:.3f}.pdf".format(self.intp_choice, *self.params_test[idx][0:5]))
			if self.theta is None and self.t_max is not None:
				plt.savefig("intp_figures/{0}_TP_wind2_md{1:.4f}_vd{2:.3f}_mw{3:.4f}_vw{4:.3f}_theta{5:.3f}.pdf".format(self.intp_choice, *self.params_test[idx][0:5]))
			if self.t_max is None and self.theta is None:
				plt.savefig("intp_figures/{0}_TP_wind2_md{1:.4f}_vd{2:.3f}_mw{3:.4f}_vw{4:.3f}_t{5:.3f}_theta{6:.3f}.pdf".format(self.intp_choice, *self.params_test[idx][0:6]))
			plt.close()

		return

