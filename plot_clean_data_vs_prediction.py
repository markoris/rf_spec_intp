import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
from matplotlib import ticker
import matplotlib.patheffects as PathEffects

# FLUX CORRECTED SMOOTHED SPECTRA DATA 
obs_dir = '/users/mristic/spec_intp_paper/AT2017gfo_spectra/flux_corrected_smoothed_joined_spectra/'
pred_dir = '/users/mristic/nn_spec_intp/at2017gfo_predictions/'
obs_files = natsorted(glob.glob(obs_dir+'*.dat'))
pred_files = natsorted(glob.glob(pred_dir+'*.dat'))

for obs, pred in zip(obs_files, pred_files):
	time = float(obs.split('/')[-1].split('+')[-1][:-5])
	obs = np.loadtxt(obs)
	pred = np.loadtxt(pred)

	obs[:, 1] = np.where(np.isfinite(obs[:, 1]), obs[:, 1], 1e-18)
	obs[:, 1] = np.where(obs[:, 1] > 0, obs[:, 1], 1e-18)
	wavs = obs[:, 0]
	bad_wavs = np.where(((wavs >= 5300) & (wavs <= 5700)) | ((wavs >= 9800) & (wavs <= 10100)) | ((wavs >= 13100) & (wavs <= 14400)) | ((wavs >= 17900) & (wavs <= 19400)))[0]
	obs = np.delete(obs, bad_wavs, axis=0)

	cutoffs = np.where(np.diff(obs[:, 0]) > 10)[0] + 1 # find where gaps are larger than 10 microns. this is where discontinuities lie (from telluric/other effects)
	cutoffs = np.insert(cutoffs, [0, cutoffs.shape[0]], [0, obs.shape[0]-1])
	
	pred[:, 0] *= 1e4 # angstrom scaling
	pred[:, 1] /= (4e6)**2 # 40 Mpc scaling

	plt.rc('font', size=35)
	plt.rc('lines', lw=4)
	plt.figure(figsize=(19.2, 10.8))

	for i in range(len(cutoffs)-1):
		plt.plot(obs[cutoffs[i]:cutoffs[i+1], 0], obs[cutoffs[i]:cutoffs[i+1], 1], c='k')

	filters = np.array(glob.glob('filters/*'))
	wavelengths = 'grizyJHKS'
	colors = {"g": "blue", "r": "cyan", "i": "lime", "z": "green", "y": "greenyellow", "J": "gold",
		 "H": "orange", "K": "red", "S": "darkred"}
	text_locs = {"g": 0.2, "r": 0.32, "i": 0.41, "z": 0.48, "y": 0.54, "J": 0.62, "H": 0.75, "K": 0.88}
	for fltr in range(len(filters)):
		filter_wavs = np.loadtxt(filters[fltr])
		#filter_wavs = filter_wavs[:, 0]*1e-4 # factor of 1e-4 to go from Angstroms to microns
		#text_loc = np.mean(filter_wavs*1e-4)
		#text_loc = (np.log10(text_loc)-np.log10(pred[:, 0].min()*1e-4))/(np.log10(pred[:, 0].max()*1e-4)-np.log10(pred[:, 0].min()*1e-4))
		wav_low, wav_upp = filter_wavs[0, 0], filter_wavs[-1, 0]
		fltr_band = filters[fltr].split('/')[-1][0]
		if fltr_band == "S": continue
		text_loc = text_locs[fltr_band]
		fltr_indx = wavelengths.find(fltr_band)
		plt.axvspan(wav_low, wav_upp, alpha=0.5, color=colors[wavelengths[fltr_indx]])
		plt.text(text_loc, 1.015, fltr_band, color=colors[wavelengths[fltr_indx]], transform=plt.gca().transAxes, path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="black")])

	plt.plot([], [], ls='-', color='black', label="AT2017gfo")
	plt.plot(pred[:, 0], pred[:, 1], c='r', label="prediction")
	ylims = np.array([np.mean(obs[:, 1])/20, np.mean(obs[:, 1])*20])
	plt.ylim(ylims)
	plt.xscale('log')
	plt.yscale('log')
	#plt.title(r"t = %gd" % time, y=1.06)
	plt.gca().set_xticks(np.array([0.5, 1, 2])*1e4)
	plt.gca().set_xticklabels(np.array([0.5, 1, 2])*1e4)
	plt.xticks(np.array([0.5, 1, 2])*1e4)
	plt.gca().get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
	plt.ylabel(r'$F_{\lambda} (erg \ s^{-1} \ cm^{-2} \ \AA^{-1})$')
	plt.xlabel(r'$\lambda$ ($\AA$)')
	plt.legend()
	figname = "AT2017gfo_comparison_figures/"
	figname += "t_%gd.pdf" % time
	print(figname)
	plt.gca().minorticks_off()
	plt.savefig(figname)

#data_1micron = np.argmin(np.abs(data[:, 0]-10000))
#pred_1micron = np.argmin(np.abs(prediction[:, 0]-10000))

#print(data[data_1micron, 1]/prediction[pred_1micron, 1])
