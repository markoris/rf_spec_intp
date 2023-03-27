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

def color_rgb(time):
	return (1/10.4*time, 0, 1/10.4*(10.4-time))

# -- loading SuperNu-binned AT2017gfo observational spectra

at2017gfo_spectra = np.array(glob.glob('binned_at2017gfo_spectra/*.dat')) # [10, 1024] array
at2017gfo_spectra = natsorted(at2017gfo_spectra)
times_orig = [float(filename.split('/')[-1].split('+')[1].split('_')[0][:-1]) for filename in at2017gfo_spectra]
times_orig = np.array(times_orig).reshape(-1, 1)

intptime01 = np.loadtxt('intptime1p43days.dat') # 1 x 1024 array
intptime04 = np.loadtxt('intptime4p4days.dat') # 1 x 1024 array
intptime07 = np.loadtxt('intptime7p4days.dat') # 1 x 1024 array
intptime10 = np.loadtxt('intptime10p4days.dat') # 1 x 1024 array

# -- loading SuperNu simulations with best-fit parameters
wavs_supernu = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # from cm to microns (via 1e4 scaling factor)

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

plt.plot(wavs_supernu, intptime01 , '--', c=color_rgb(1.43), label=r'$t = 1.43 \rm d$')
plt.plot(wavs_supernu, intptime04 , '--', c=color_rgb(4.4), label=r'$t = 4.4 \rm d$')
plt.plot(wavs_supernu, intptime07 , '--', c=color_rgb(7.4), label=r'$t = 7.4 \rm d$')
plt.plot(wavs_supernu, intptime10 , '--', c=color_rgb(10.4), label=r'$t = 10.4 \rm d$')
plt.title('%g deg' % float(0), loc="left")
plt.xscale('log')
plt.yscale('log')
plt.xlim([0.3, 2.4])
#plt.ylim([1e-19, 5e-17])
plt.ylim([1e-21, np.max(intptime01)*5])
#plt.ylim([1e-19, np.max(intpdeg00)*5])
plt.gca().set_xticks([0.5, 1, 2])
#plt.gca().set_xticks([0.5, 1, 2, 3, 5, 10])
plt.gca().get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
#plt.xlabel(r'$\lambda \ (\mu m)$')
plt.gca().minorticks_off()
plt.ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')
plt.xlabel(r'$\lambda$ [$\mu$m]')
plt.legend(ncol=2, loc="lower right")
plt.savefig('angle_variation/theta_%g.pdf' % float(0))
