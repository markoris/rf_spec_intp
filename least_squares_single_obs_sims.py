def load_data(path_to_sims_dir, path_to_spectra_h5, short_prms=True, short_wavs=True, t_max=38.055, theta=0, downsample_theta=False, verbose=True, debugging=False):

	import h5py, sys, glob
	import numpy as np
		
	def get_params_from_filename(string):
		name_parse = string.split('/')[-1].split('_')
		params = np.array([name_parse[1], name_parse[5][-1], name_parse[7][2:], name_parse[8][2:], name_parse[9][2:], name_parse[10][2:]])
		return params

	t_max = t_max
	theta = theta

	### Prepare wavelength information	

	wavs_full = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # in microns (from 1e4 scaling factor)
	
	if short_wavs:
		#wav_idxs = np.where((wavs_full>0.4) & (wavs_full<10))[0] # above .4 micron and below 10 microns
		wav_idxs = np.where((wavs_full>0.4) & (wavs_full<2.5))[0] # above .4 micron and below 2.5 microns
	else: 
		wav_idxs = np.arange(len(wavs_full))
	
	wavs = wavs_full[wav_idxs]

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
	if verbose: print("Parameter array shape: ", params_all.shape)

	### Prepare spectra to serve as interpolation targets

	### First choose the cutoff time for spectra

	t_a = 0.125 # spectral start time
	#t_b = 49.351 # spectral end time
	t_b = 20.749 # spectral end time FIXME this is only temporary, need to remake .hdf5 to include time up to 66 iteration
	# FIXME re-run the .hdf5 creation using longest time available to all spectra. then re-write times to fit that number
	times = np.logspace(np.log10(t_a), np.log10(t_b), 60) # this should cover the full spectral possibilities range
	#times = np.logspace(np.log10(t_a), np.log10(t_b), 70) # this should cover the full spectral possibilities range FIXME
	angles = np.degrees([np.arccos((1 - (i-1)*2/54.)) for i in np.arange(1, 55, 1)])

	if verbose: print("Spectra times: ", times)

	if t_max is not None:
		t_idx = np.argmin(np.abs(times-t_max))
		times = times[:t_idx+1]
		if verbose: print("Time index of %d corresponds to %f days" % (t_idx, times[t_idx]))

	if theta is not None:
		angle_idx = np.argmin(np.abs(angles-theta))
		if verbose: print("Angle index of %d corresponds to %f degrees" % (angle_idx, angles[angle_idx]))

	if verbose: print("Loading h5 file, this may take a while on the first execution")

	h5_load = h5py.File(path_to_spectra_h5, 'r')
	data = h5_load['spectra'][:]
	spec_all = data.reshape(files.shape[0], times.shape[0], wavs_full.shape[0], angles.shape[0])
	
	if verbose: print('Data loaded and reshaped')

	params = params_all
	spectra = spec_all

	#spectra *= 54/(4*np.pi) # 4pi solid angle split over 54 angular bins, solid angle correction for one angular bin!
	spectra *= 54 # solid angle correction for one angular bin, spectra already divided by 4*pi*Robs^2 where Robs = 10 pc
	if debugging:
		params = params[:20, ...]
		spectra = spectra[:20, ...]

	if t_max is None and theta is None: # does not specify time or angle index
		spectra = spectra[:, :, wav_idxs, :]
		if downsample_theta: 
			spectra = spectra[:, :, :, np.arange(0, angles.shape[0]-1, 2)]
			angles = np.degrees([np.arccos((1 - (i-1)*2/27.)) for i in np.arange(1, 28, 1)])
		return times, params, spectra

	if t_max is None or theta is None: # one of angle or time unspecified

		if t_max is None:
			spectra = spectra[:, :, wav_idxs, angle_idx] # only angle specified, all times
			return times, params, spectra

		if theta is None:
			spectra = spectra[:, t_idx, wav_idxs, :] # only time specified, all angles
			if downsample_theta: 
				spectra = spectra[:, :, np.arange(0, angles.shape[0]-1, 2)]
				angles = np.degrees([np.arccos((1 - (i-1)*2/27.)) for i in np.arange(1, 28, 1)])
			return times, params, spectra

	spectra = spectra[:, t_idx, wav_idxs, angle_idx]	# time and angle specified
	return times, params, spectra

# ---------------------------------#

def least_squares(path_to_obs_dir, sim_params, sim_spectra, sim_times):

	import glob
	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib.patheffects as PathEffects
	from matplotlib import ticker
	from natsort import natsorted

	at2017gfo_spectra = np.array(glob.glob(path_to_obs_dir+'/*.dat')) # [10, 1024] array
	at2017gfo_spectra = natsorted(at2017gfo_spectra)
	times_orig = [float(filename.split('/')[-1].split('+')[1].split('_')[0][:-1]) for filename in at2017gfo_spectra]
	times_orig = np.array(times_orig).reshape(-1, 1)

	wavs_supernu = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # from cm to microns (via 1e4 scaling factor)
	sim_spectra /= (4e6)**2 # scaling 40 Mpc source distance with source assumed emitting from 10 pc

	for t in range(len(times_orig)):
		print('t = ', times_orig[t])
		t_idx = np.argmin(np.abs(times_orig[t]-sim_times))
		pred = sim_spectra[:, t_idx, :] # [n_draws, 1024] shape array
		print(pred.shape)
		obs = np.loadtxt(at2017gfo_spectra[t])
		mask = np.where(obs[:, 1] > 0)[0]
		print(len(mask), '/%d wav bins used' % obs.shape[0])
		residuals = np.sum(((obs[mask, 1]-pred[:, mask])/obs[mask, 2])**2, axis=1)
		#residuals = np.sum(((np.log10(obs[mask, 1])-np.log10(pred[:, mask]))/np.log10(obs[mask, 2]))**2, axis=1)
		#print('residuals: ', residuals)
		print('mean of obs: ', np.mean(obs[mask, 1]))
		print('mean of pred: ', np.mean(pred[:, mask]))
		print('mean of err: ', np.mean(obs[mask, 2]))
		np.savetxt('/lustre/scratch4/turquoise/mristic/at2017gfo_likelihoods_t%g.dat' % times_orig[t], np.c_[sim_params, residuals], fmt="%g", header="md vd mw vw residuals") 

		residuals -= np.min(residuals)
		residuals = np.exp(-1*residuals)
		idx_max_L = np.argmax(residuals)
		pred = pred[idx_max_L]

		print('Best parameters for time %g = ' % times_orig[t], sim_params[idx_max_L])

		try:
			recov = np.c_[sim_params[idx_max_L].reshape(1, 4), np.array(len(mask)/1024).reshape(1, 1)]
			recovered_parameters = np.concatenate((recovered_parameters, recov), axis=0)
		except NameError:
			recov = np.c_[sim_params[idx_max_L].reshape(1, 4), np.array(len(mask)/1024).reshape(1, 1)]
			recovered_parameters = recov

		obs = obs[mask, :]

		# wavelengths are now in supernu bins, which are spaced evenly in log(wav) space
		# therefore we need to look at np.log10(obs[:, 0])
		# standard diff for np.diff(np.log10(obs[mask, 0])) = 0.002 
		
		cutoffs = np.where(np.diff(np.log10(obs[:, 0])) > 0.003)[0] + 1 # find where gaps are larger than 10 Angstroms. this is where discontinuities lie (from telluric/other effects)
		cutoffs = np.insert(cutoffs, [0, cutoffs.shape[0]], [0, obs.shape[0]-1])

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
		plt.savefig('figs_from_lowest_residuals/sim_t_%gd.pdf' % times_orig[t])
	np.savetxt('recovered_parameters_from_sims.dat', recovered_parameters)
	rcv_prm = np.loadtxt('recovered_parameters_from_sims.dat')
	rcv_prm[:, 0] = np.log10(rcv_prm[:, 0]) # take log of dynamical mass
	rcv_prm[:, 2] = np.log10(rcv_prm[:, 2]) # take log of wind mass
	rcv_prm = np.average(rcv_prm[:, :4], weights=rcv_prm[:, 4], axis=0)
	print('Recovered Parameters: md = %f, mw = %f, vd = %f, vw = %f' % (10**rcv_prm[0], rcv_prm[1], 10**rcv_prm[2], rcv_prm[3]))

times, prm, spc = load_data('/lustre/scratch4/turquoise/mristic/knsc1_active_learning/*spec*', '/lustre/scratch4/turquoise/mristic/h5_data/TP_wind2_spectra.h5', t_max=None, theta=0, short_wavs=False)

print(prm.shape, spc.shape)

least_squares('binned_at2017gfo_spectra', prm, spc, times)
