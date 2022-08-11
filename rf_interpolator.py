class intp_rf(object):
	
	def __init__(self, path_to_sims_dir, path_to_spectra_h5, short_wavs=True, microns=True, seed=None, shorten=True, verbose=False, cutoff=60):

		"""
		Sets up the Random Forest (RF) interpolator for interpolation of kilonova spectra in the LANL published format.

		path_to_sims_dir: 	string
				  	Path to directory containing simultions of spectra using LANL naming convention (as provided in https://zenodo.org/record/5745556)

		path_to_spectra_h5: 	string
				    	Path to .hdf5 file containing spectra matching those in path_to_sims_dir. See included convert_dat_to_h5.py file for proper conversion

		short_wavs: 		bool
					If true, uses only wavelengths relevant to included grizyJHKS filters  
		"""

		import numpy as np
	
		self.verbose = verbose

		wavs_full = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024) # in cm
		
		if short_wavs:
			self.wav_idxs = np.where((wavs>4e-5) & (wavs<1e-3))[0] # limits to above .4 micron and up to 10 micron, these are cm wavelengths!!
		else: 
			self.wav_idxs = np.arange(len(self.wavs))
		
		self.wavs = wavs_full[self.wav_idxs]
	
		if microns:
			self.wavs *= 1e4 # from cm to microns

		if not seed == None: np.random.seed(seed)

		self.params, self.spectra = self.load_data(path_to_sims_dir, path_to_spectra_h5, shorten)

	def load_data(self, path_to_sims_dir, path_to_spectra_h5, shorten, cutoff):

		import glob
		import h5py
			
		def get_params_from_filename(string):
			name_parse = string.split('/')[-1].split('_')
			params = np.array([name_parse[1], name_parse[5][-1], name_parse[7][2:], name_parse[8][2:], name_parse[9][2:], name_parse[10][2:]])
			return params

		files = np.array(glob.glob(path_to_sims_dir))
		files.sort()
		
		for idx in range(files.shape[0]):
			params = get_params_from_filename(files[idx]) # parse filenames into parameters
			try: params_all = np.concatenate((params_all, params[None, :]), axis=0)
			except NameError: params_all = params[None, :]
		if shorten: params_all = params_all[:, 2:].astype('float')
		if self.verbose: print("Parameter array shape: ", params_all.shape)
	
		# time_index = np.argmin(np.abs(times-time)), times = np.logspace(times), time = user input
		# same for theta

		t_a = 0.125 # spectral start time
		t_b = 20.749 # spectral end time assuming stopping at cutoff=60
		self.times = np.logspace(np.log10(t_a), np.log10(t_b), 60)
	
		h5_load = h5py.File(path_to_spectra_h5, 'r')
		data = h5_load['spectra'][:]
		spec_all = data.reshape(412, 60, 1024, 54) # this is predetermined right now, consider relaxing in the future
		spec_all = spec_all[:, time_index, self.wav_idxs, theta] 

		

		self.params = 
		self.spectra = 

		return


# sample code for separate script

intp = rf_intp('/home/marko.ristic/lanl/knsc1_active_learning/*spec*', 'h5_data/TP_wind2_spectra.h5',)













#wavs = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024) # in cm
#wav_idxs = np.where((wavs>4e-5) & (wavs<1e-3))[0] # limits to above .4 micron and up to 10 micron, these are cm wavelengths!!
#wavs *= 1e4 # from cm to microns

#np.random.seed(100)

files = np.array(glob.glob('/home/marko.ristic/lanl/knsc1_active_learning/*spec*'))
files.sort()

for idx in range(files.shape[0]):
	params = get_params(files[idx]) # parse filenames into parameters
	try: params_all = np.concatenate((params_all, params[None, :]), axis=0)
	except NameError: params_all = params[None, :]
params_all = params_all[:, 2:].astype('float')
print(params_all.shape)

h5_load = h5py.File('h5_data/TP_wind2_spectra.h5', 'r')
data = h5_load['spectra'][:]
spec_all = data.reshape(412, 60, 1024, 54)
spec_all = spec_all[:, 50, wav_idxs, 0] # t ~ 10 days, 0 <= theta <= 17 deg, 0.4 mu <= lambda <= 10 mu




import h5py
import glob
import slick
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patheffects as PathEffects

plt.rc('font', size=30)
plt.rc('lines', lw=3)

def get_params(string):
	name_parse = string.split('/')[-1].split('_')
	params = np.array([name_parse[1], name_parse[5][-1], name_parse[7][2:], name_parse[8][2:], name_parse[9][2:], name_parse[10][2:]])
	return params

wavs = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024) # in cm
wav_idxs = np.where((wavs>4e-5) & (wavs<1e-3))[0] # limits to above .4 micron and up to 10 micron, these are cm wavelengths!!
wavs *= 1e4 # from cm to microns

np.random.seed(100)

files = np.array(glob.glob('/home/marko.ristic/lanl/knsc1_active_learning/*spec*'))
files.sort()

for idx in range(files.shape[0]):
	params = get_params(files[idx]) # parse filenames into parameters
	try: params_all = np.concatenate((params_all, params[None, :]), axis=0)
	except NameError: params_all = params[None, :]
params_all = params_all[:, 2:].astype('float')
print(params_all.shape)

h5_load = h5py.File('h5_data/TP_wind2_spectra.h5', 'r')
data = h5_load['spectra'][:]
spec_all = data.reshape(412, 60, 1024, 54)
spec_all = spec_all[:, 50, wav_idxs, 0] # t ~ 10 days, 0 <= theta <= 17 deg, 0.4 mu <= lambda <= 10 mu

mu, sigma = np.mean(spec_all, axis=0), np.std(spec_all, axis=0)
spec_all -= mu
spec_all /= sigma

idx = np.random.choice(np.arange(spec_all.shape[0]))

test_params = params_all[idx]
test_spec = spec_all[idx]
params_all = np.delete(params_all, idx, axis=0)
spec_all = np.delete(spec_all, idx, axis=0)

print(params.shape)
print(spec_all.shape)

intp_rf = slick.Interpolator() # train with random forest which typically has better convergence to verify fidelity of NN output
intp_rf.train(params_all, spec_all) # actual training with RF
intp_spec = intp_rf.evaluate(test_params[None, :]) 

test_spec *= sigma
test_spec += mu

intp_spec *= sigma
intp_spec += mu

plt.figure(figsize=(19.2, 10.8))
plt.subplots_adjust(right=0.95)

filters = np.array(glob.glob('/home/marko.ristic/lanl/code_python3/lanl_data_conversions/filters/*'))
wavelengths = 'grizyJHKS'
colors = {"g": "blue", "r": "cyan", "i": "lime", "z": "green", "y": "greenyellow", "J": "gold",
         "H": "orange", "K": "red", "S": "darkred"}
text_loc_relative = [0.09, 0.165, 0.225, 0.26, 0.3, 0.35, 0.43, 0.51, 0.72]
for fltr in range(len(filters)):
	filter_wavs = np.loadtxt(filters[fltr])
	filter_wavs = filter_wavs[:, 0]*1e-4 # factor of 1e-4 to go from Angstroms to microns
	text_loc = np.mean(filter_wavs)
	text_loc -= text_loc/20
	wav_low, wav_upp = filter_wavs[0], filter_wavs[-1]
	fltr_band = filters[fltr].split('/')[-1][0]
	fltr_indx = wavelengths.find(fltr_band)
	plt.axvspan(wav_low, wav_upp, alpha=0.5, color=colors[wavelengths[fltr_indx]])
	plt.text(text_loc_relative[fltr_indx], 1.015, fltr_band, color=colors[wavelengths[fltr_indx]], transform=plt.gca().transAxes, path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="black")])

plt.title(r"TP   wind2   $M_d$={0:.4f}   $v_d$={1:.3f}   $M_w$={2:.4f}   $v_w$={3:.3f}".format(*test_params[0:4]), y=1.06)
plt.plot(wavs[wav_idxs], test_spec, label='true', color='k')
plt.plot(wavs[wav_idxs], intp_spec, label='intp', color='red')
plt.xscale('log')
plt.yscale('log')
plt.gca().set_xticks([0.5, 1, 2, 3, 5, 10])
plt.gca().get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
#plt.xlabel(r'$\lambda \ (\mu m)$')
plt.ylabel(r'$F_{\lambda} (erg \ s^{-1} \ cm^{-2} \ \mu m^{-1})$')
plt.xlabel(r'$\lambda$ ($\mu$m)')
plt.legend()
plt.savefig('rf_training_h5.pdf')

