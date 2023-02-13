import glob, sys
import numpy as np
import spectra_interpolator as si
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patheffects as PathEffects

at2017gfo_spectra = np.array(glob.glob('binned_at2017gfo_spectra/*.dat')) # [10, 1024] array
at2017gfo_spectra = natsorted(at2017gfo_spectra)
times_orig = [float(filename.split('/')[-1].split('+')[1].split('_')[0][:-1]) for filename in at2017gfo_spectra]
times_orig = np.array(times_orig).reshape(-1, 1)

#n_draws = 5
n_draws = 10000
param_mins = [-3, 0.05, -3, 0.05]
param_maxes = [-1, 0.3, -1, 0.3]

draws_orig = np.random.uniform(param_mins, param_maxes, size=(n_draws, len(param_mins)))
draws_orig[:, 0], draws_orig[:, 2] = 10**draws_orig[:, 0], 10**draws_orig[:, 2]

draws = np.repeat(draws_orig, len(times_orig), axis=0) # 100000x4 array, 10000x4 array repeated 10 times for 10 observations
times = np.tile(times_orig, (n_draws, 1)) # 100000x1 array, 10x1 array repeated 10000 times for 10000 draws
inputs = np.concatenate((draws, times), axis=1) # 100000x5 array, 4 ejecta parameters and 1 time parameter

intp = si.intp()
intp.load_data('/lustre/scratch4/turquoise/mristic/knsc1_active_learning/*spec*', '/lustre/scratch4/turquoise/mristic/h5_data/TP_wind2_spectra.h5', t_max=None, theta=0, short_wavs=False)
intp.append_input_parameter(intp.times, 1)
intp.preprocess()
intp.load('/lustre/scratch4/turquoise/mristic/rf_spec_intp_theta00deg.joblib', fixed_angle=True)
out = intp.evaluate(inputs, ret_out=True) # shape [n_draws*len(times), 1024]

wavs_supernu = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # from cm to microns (via 1e4 scaling factor)
out /= (4e6)**2 # scaling 40 Mpc source distance with source assumed emitting from 10 pc

for t in range(len(times_orig)):
	print('t = ', times_orig[t])
	pred = out[t::10] # [n_draws, 1024] shape array
	print(pred.shape)
	obs = np.loadtxt(at2017gfo_spectra[t])
	mask = np.where(obs[:, 1] > 0)[0] # this will be the first N arrays
	print(len(mask), '/%d wav bins used' % obs.shape[0])

	# SuperNu wavelengths are uniformly LOG-spaced
	# the data is linearly spaced, but we are binning it into the SuperNu wavelengths
	# therefore, more data points will be in the *RED* part of the spectrum than the *BLUE* part
	# below, I try weighting based on SuperNu wavelength bin spacing, giving more weight to shorter wavelengths

	# take the difference between adjacent wavelength bins
	#prefer_early_times = np.diff(wavs_supernu) # size 1023 array
	# this creates an N-1 length array, insert the first entry at the start for completeness without too much loss of generality
	#prefer_early_times = np.insert(prefer_early_times, 0, prefer_early_times[0]) # size 1024 array
	#prefer_early_times /= np.min(prefer_early_times) # scaling factor for reasonable residual values, in case we ever want likelihoods, also removes units
	# take the inverse, as we want the bins with smaller spacing to be given a higher weight (earlier bins have fewer data points)
	#prefer_early_times = 1/prefer_early_times
	#prefer_early_times /= np.min(prefer_early_times) # scaling factor for reasonable residual values, in case we ever want likelihoods, also removes units

	residuals = np.sum(((obs[mask, 1]-pred[:, mask])/obs[mask, 2])**2, axis=1)*1/len(mask)
	#print('residuals: ', residuals)
	#print('mean of obs: ', np.mean(np.log10(obs[mask, 1])))
	#print('mean of pred: ', np.mean(np.log10(pred[:, mask])))
	#print('mean of err: ', np.mean(np.log10(obs[mask, 2])))
	np.savetxt('/lustre/scratch4/turquoise/mristic/at2017gfo_likelihoods_t%g.dat' % times_orig[t], np.c_[draws_orig, residuals], fmt="%g", header="md vd mw vw residuals") 

	#residuals -= np.min(residuals)
	#residuals = np.exp(-1*residuals)
	#idx_max_L = np.argmax(residuals)
	idx_max_L = np.argmin(residuals)
	print('Lowest residual: ', residuals[idx_max_L])
	pred = pred[idx_max_L]

	print('Best parameters for time %g = ' % times_orig[t], draws_orig[idx_max_L])

	try:
		recov = np.c_[draws_orig[idx_max_L].reshape(1, 4), np.array(len(mask)/1024).reshape(1, 1)]
		recovered_parameters = np.concatenate((recovered_parameters, recov), axis=0)
	except NameError:
		recov = np.c_[draws_orig[idx_max_L].reshape(1, 4), np.array(len(mask)/1024).reshape(1, 1)]
		recovered_parameters = recov

	obs = obs[mask, :]

	# wavelengths are now in supernu bins, which are spaced evenly in log(wav) space
	# therefore we need to look at np.log10(obs[:, 0])
	# standard diff for np.diff(np.log10(obs[mask, 0])) = 0.002 
	
	cutoffs = np.where(np.diff(np.log10(obs[:, 0])) > 0.003)[0] + 1 # find where gaps are larger than 10 Angstroms. this is where discontinuities lie (from telluric/other effects)
	cutoffs = np.insert(cutoffs, [0, cutoffs.shape[0]], [0, obs.shape[0]-1])
	#print(cutoffs)

	obs[:, 0] *= 1e-4 # Angstroms to microns, for plotting purposes
	
	plt.rc('font', size=35)
	plt.rc('lines', lw=4)
	plt.figure(figsize=(19.2, 10.8))

	filters = np.array(glob.glob('filters/*'))
	wavelengths = 'grizyJHKS'
	colors = {"g": "blue", "r": "cyan", "i": "lime", "z": "green", "y": "greenyellow", "J": "gold",
		 "H": "orange", "K": "red", "S": "darkred"}
	text_locs = {"g": 0.2, "r": 0.35, "i": 0.44, "z": 0.51, "y": 0.57, "J": 0.65, "H": 0.78, "K": 0.91}
	for fltr in range(len(filters)):
		filter_wavs = np.loadtxt(filters[fltr])
		filter_wavs = filter_wavs[:, 0]*1e-4 # factor of 1e-4 to go from Angstroms to microns
		wav_low, wav_upp = filter_wavs[0], filter_wavs[-1]
		fltr_band = filters[fltr].split('/')[-1][0]
		if fltr_band == "S": continue
		text_loc = text_locs[fltr_band]
		fltr_indx = wavelengths.find(fltr_band)
		plt.axvspan(wav_low, wav_upp, alpha=0.5, color=colors[wavelengths[fltr_indx]])
		plt.text(text_loc, 1.015, fltr_band, color=colors[wavelengths[fltr_indx]], transform=plt.gca().transAxes, path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="black")])
	
	for i in range(len(cutoffs)-1):
		plt.plot(obs[cutoffs[i]:cutoffs[i+1], 0], obs[cutoffs[i]:cutoffs[i+1], 1], c='k')
	plt.plot(wavs_supernu, pred, c='r', label='Ristic fit')
	plt.title('%g days' % times_orig[t], loc="left")
	plt.plot([], [], c='k', label='AT2017gfo')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim([0.3, 2.4])
	#plt.ylim([1e-19, 5e-17])
	plt.ylim([1e-19, np.max(obs[:, 1])*5])
	plt.gca().set_xticks(np.array([0.5, 1, 2]))
	plt.gca().set_xticklabels(np.array([0.5, 1, 2]))
	plt.gca().minorticks_off()
	plt.ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$]')
	plt.xlabel(r'$\lambda$ [$\mu$m]')
	plt.legend()
	plt.savefig('figs_from_lowest_residuals/intp_t_%gd.pdf' % times_orig[t])
np.savetxt('recovered_parameters_from_intp.dat', recovered_parameters)
rcv_prm = np.loadtxt('recovered_parameters_from_intp.dat')
rcv_prm[:, 0] = np.log10(rcv_prm[:, 0]) # take log of dynamical mass
rcv_prm[:, 2] = np.log10(rcv_prm[:, 2]) # take log of wind mass
rcv_prm = np.average(rcv_prm[:, :4], weights=rcv_prm[:, 4], axis=0)
print('Recovered Parameters: md = %f, mw = %f, vd = %f, vw = %f' % (10**rcv_prm[0], 10**rcv_prm[2], rcv_prm[1], rcv_prm[3]))
