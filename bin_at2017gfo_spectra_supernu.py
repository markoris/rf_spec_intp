import glob, sys
import numpy as np
import spectra_interpolator as si
from natsort import natsorted
import matplotlib.pyplot as plt

at2017gfo_spectra = np.array(glob.glob('/users/mristic/spec_intp_paper/AT2017gfo_spectra/flux_corrected_smoothed_joined_spectra/*.dat'))

# bin observations to match supernu wavelength bins
at2017gfo_spectra = natsorted(at2017gfo_spectra) # go through each spectrum, bin according to supernu wavelength binning
# binning should make all spectra have same length, thus can concatenate to common array
# output some obs array which is called in the likelihood interpolation 
# TODO: common array holding binned observations here

wavs_supernu = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e8 # from cm to Angstroms (via 1e8 scaling factor)

for obs_file in at2017gfo_spectra:
	binned_spectrum = np.ones_like(wavs_supernu)*(-1)
	binned_spec_err = np.ones_like(wavs_supernu)*(-1)
	obs = np.loadtxt(obs_file)
	#obs[:, 1] = np.where(np.isfinite(obs[:, 1]), obs[:, 1], 1e-18) # replacing NaNs
	#obs[:, 1] = np.where(obs[:, 1] > 0, obs[:, 1], 1e-18) # replacing negative values
	bad_idxs = np.where(obs[:, 1] < 0)[0] # replacing negative values
	obs = np.delete(obs, bad_idxs, axis=0) # delete bad data! 
	wavs = obs[:, 0]
	bad_wavs = np.where(((wavs >= 5300) & (wavs <= 5700)) | ((wavs >= 9800) & (wavs <= 10100)) | ((wavs >= 13100) & (wavs <= 14400)) | ((wavs >= 17900) & (wavs <= 19400)))[0]
	obs = np.delete(obs, bad_wavs, axis=0)

	for wavbin in range(len(wavs_supernu)-1):
		mask = np.where((obs[:, 0] > wavs_supernu[wavbin]) & (obs[:, 0] < wavs_supernu[wavbin+1]))[0]
		#n_okay = np.sum(mask)
		if len(mask) == 0: binned_spectrum[wavbin] = -1
		#if n_okay == 0: binned_spectrum[wavbin] = -1
		else:
			binned_spectrum[wavbin] = np.mean(obs[mask, 1])
			binned_spec_err[wavbin] = np.mean(obs[mask, 3])	

	negative_idxs = np.where(binned_spectrum<0)[0]
	binned_spectrum_plot = np.delete(binned_spectrum, negative_idxs, axis=0)
	binned_wavs = np.delete(wavs_supernu, negative_idxs, axis=0)

	cutoffs = np.where(np.diff(obs[:, 0]) > 10)[0] + 1 # find where gaps are larger than 10 microns. this is where discontinuities lie (from telluric/other effects)
	cutoffs = np.insert(cutoffs, [0, cutoffs.shape[0]], [0, obs.shape[0]-1])
	#
	plt.rc('font', size=35)
	plt.rc('lines', lw=4)
	plt.figure(figsize=(19.2, 10.8))
	
	for i in range(len(cutoffs)-1):
		plt.plot(obs[cutoffs[i]:cutoffs[i+1], 0], obs[cutoffs[i]:cutoffs[i+1], 1], c='k')
	#plt.plot(wavs_supernu[np.where(binned_spectrum>0)[0]], binned_spectrum[np.where(binned_spectrum>0)[0]], c='r')
	plt.scatter(binned_wavs, binned_spectrum_plot, c='r', s=300)
	#plt.ylim([0, 0.25e-16])
	plt.savefig('binned_at2017gfo_spectra/binned_spec_%s.pdf' % obs_file.split('/')[-1].split('+')[1].split('_')[0][:-5])
	
	np.savetxt('binned_at2017gfo_spectra/'+obs_file.split('/')[-1][:-4]+'_binned_supernu.dat', np.c_[wavs_supernu, binned_spectrum, binned_spec_err])
	print(binned_spectrum.shape)
