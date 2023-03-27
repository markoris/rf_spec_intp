# need to write separate loader script which loads the intp and evaluates the inputs at the respective times for the given angle
# then, save those outputs to file, and load them into this script which will plot multiple angles for a given time

import glob, sys
import numpy as np
import metzger2017 as m17
import spectra_interpolator as si
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patheffects as PathEffects

def color_rgb(theta):
	return (1/90*theta, 0, 1/90*(90-theta))

# -- loading SuperNu-binned AT2017gfo observational spectra

at2017gfo_spectra = np.array(glob.glob('binned_at2017gfo_spectra/*.dat')) # [10, 1024] array
at2017gfo_spectra = natsorted(at2017gfo_spectra)
times_orig = [float(filename.split('/')[-1].split('+')[1].split('_')[0][:-1]) for filename in at2017gfo_spectra]
times_orig = np.array(times_orig).reshape(-1, 1)

intpdeg00 = np.loadtxt('intpdeg00deg.dat') # 1 x 1024 array
intpdeg30 = np.loadtxt('intpdeg30deg.dat') # 1 x 1024 array
intpdeg45 = np.loadtxt('intpdeg45deg.dat') # 1 x 1024 array
intpdeg60 = np.loadtxt('intpdeg60deg.dat') # 1 x 1024 array
intpdeg75 = np.loadtxt('intpdeg75deg.dat') # 1 x 1024 array
intpdeg90 = np.loadtxt('intpdeg90deg.dat') # 1 x 1024 array

# -- loading SuperNu simulations with best-fit parameters
t_a = 0.125 # spectral start time
t_b = 20.749 # spectral end time FIXME this is only temporary, need to remake .hdf5 to include time up to 66 iteration
sim_times = np.logspace(np.log10(t_a), np.log10(t_b), 60) # this should cover the full spectral possibilities range

wavs_supernu = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # from cm to microns (via 1e4 scaling factor)

for t in range(len(times_orig)):
	print('t = ', times_orig[t])
	obs = np.loadtxt(at2017gfo_spectra[t])
	#intp00 = intpdeg00[t]
	#intp30 = intpdeg30[t]
	#intp45 = intpdeg45[t]
	#intp60 = intpdeg60[t]
	#intp75 = intpdeg75[t]
	#intp90 = intpdeg90[t]
	mask = np.where(obs[:, 1] > 0)[0] # this will be the first N arrays
	print(len(mask), '/%d wav bins used' % obs.shape[0])

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

	#for i in range(len(cutoffs)-1):
	#	plt.plot(obs[cutoffs[i]:cutoffs[i+1], 0], obs[cutoffs[i]:cutoffs[i+1], 1], c='k')
	plt.plot(wavs_supernu, intpdeg00 , '--', c=color_rgb(0), label=r'$\theta = 0^o$')
	plt.plot(wavs_supernu, intpdeg30 , '--', c=color_rgb(30), label=r'$\theta = 30^o$')
	plt.plot(wavs_supernu, intpdeg45 , '--', c=color_rgb(45), label=r'$\theta = 45^o$')
	plt.plot(wavs_supernu, intpdeg60 , '--', c=color_rgb(60), label=r'$\theta = 60^o$')
	plt.plot(wavs_supernu, intpdeg75 , '--', c=color_rgb(75), label=r'$\theta = 75^o$')
	plt.plot(wavs_supernu, intpdeg90 , '--', c=color_rgb(90), label=r'$\theta = 90^o$')
	plt.title('%g days' % times_orig[t], loc="left")
	#plt.plot([], [], c='k', label='AT2017gfo')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim([0.3, 2.4])
	#plt.ylim([1e-19, 5e-17])
	plt.ylim([1e-21, np.max(intpdeg00)*5])
	#plt.ylim([1e-19, np.max(intpdeg00)*5])
	plt.gca().set_xticks([0.5, 1, 2])
	#plt.gca().set_xticks([0.5, 1, 2, 3, 5, 10])
	plt.gca().get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
	#plt.xlabel(r'$\lambda \ (\mu m)$')
	plt.gca().minorticks_off()
	plt.ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')
	plt.xlabel(r'$\lambda$ [$\mu$m]')
	plt.legend(ncol=2, loc="lower right")
	plt.savefig('angle_variation/t_%g.pdf' % times_orig[t])
