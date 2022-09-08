import joblib, glob
import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
import spectra_interpolator as si
import matplotlib.patheffects as PathEffects

at2017gfo_spectra = np.array(glob.glob('/users/mristic/spec_intp_paper/AT2017gfo_spectra/dereddened+deredshifted_spectra/*.dat'))

times = [float(filename.split('/')[-1].split('+')[1].split('_')[0][:-1]) for filename in at2017gfo_spectra]
times = np.array(times).reshape(-1, 1)

at2017gfo_spectra = [np.loadtxt(fname, usecols=(0, 2, 3)) for fname in at2017gfo_spectra]
for idx in range(len(at2017gfo_spectra)):
	at2017gfo_spectra[idx][:, 2] = np.where(np.isfinite(at2017gfo_spectra[idx][:, 2]), at2017gfo_spectra[idx][:, 2], 1e-18)
#	at2017gfo_spectra[idx] = np.where(at2017gfo_spectra[idx] >= 0, at2017gfo_spectra[idx], 1e-18)
	at2017gfo_spectra[idx][:, 0] *= 1e-4

inputs = np.array([10**(-1.6), 0.16, 10**(-1.4), 0.25])
inputs = np.tile(inputs, (len(at2017gfo_spectra), 1))

inputs = np.concatenate((inputs, times), axis=1)

intp = si.intp()
intp.load_data('/net/scratch4/mristic/knsc1_active_learning/*spec*', '/net/scratch4/mristic/h5_data/TP_wind2_spectra.h5', t_max=None, theta=0, short_wavs=False)
intp.append_input_parameter(intp.times, 1)
intp.preprocess()
intp.load('/net/scratch4/mristic/rf_spec_intp_theta00deg.joblib', fixed_angle=True)
intp.evaluate(inputs)

plt.rc('font', size=30)

for idx in range(len(at2017gfo_spectra)):

	plt.figure(figsize=(19.2, 10.8))
	plt.subplots_adjust(right=0.95)
	
	filters = np.array(glob.glob('filters/*'))
	wavelengths = 'grizyJHKS'
	colors = {"g": "blue", "r": "cyan", "i": "lime", "z": "green", "y": "greenyellow", "J": "gold",
		 "H": "orange", "K": "red", "S": "darkred"}
	for fltr in range(len(filters)):
		filter_wavs = np.loadtxt(filters[fltr])
		filter_wavs = filter_wavs[:, 0]*1e-4 # factor of 1e-4 to go from Angstroms to microns
		text_loc = np.mean(filter_wavs)
		text_loc = (np.log10(text_loc)-np.log10(intp.wavs.min()))/(np.log10(intp.wavs.max())-np.log10(intp.wavs.min()))
		wav_low, wav_upp = filter_wavs[0], filter_wavs[-1]
		fltr_band = filters[fltr].split('/')[-1][0]
		fltr_indx = wavelengths.find(fltr_band)
		plt.axvspan(wav_low, wav_upp, alpha=0.5, color=colors[wavelengths[fltr_indx]])
		#plt.text(text_loc_relative[fltr_indx], 1.015, fltr_band, color=colors[wavelengths[fltr_indx]], transform=plt.gca().transAxes, path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="black")])
		plt.text(text_loc, 1.015, fltr_band, color=colors[wavelengths[fltr_indx]], transform=plt.gca().transAxes, path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="black")])
	
	plt.title(r"TP   wind2   $M_d$={0:.4f}   $v_d$={1:.3f}   $M_w$={2:.4f}   $v_w$={3:.3f}   $t$={4:.3f}".format(*inputs[idx][0:5]), y=1.06)
	plt.plot(at2017gfo_spectra[idx][:, 0], at2017gfo_spectra[idx][:, 2]*(np.max(intp.prediction[idx])/np.max(at2017gfo_spectra[idx][:, 2])), label='true', color='k')
	plt.plot(intp.wavs, intp.prediction[idx], label='intp', color='red')
	plt.xscale('log')
	plt.yscale('log')
	plt.gca().set_xticks([0.5, 1, 2, 3, 5, 10])
	plt.gca().get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
##	plt.gca().set_ylim(bottom=1e-15)
	plt.ylabel(r'$F_{\lambda} (erg \ s^{-1} \ cm^{-2} \ \mu m^{-1})$')
	plt.xlabel(r'$\lambda$ ($\mu$m)')
	plt.legend()
	figname = "intp_figures/"
	figname += "rf_TP_wind2_md{0:.4f}_vd{1:.3f}_mw{2:.4f}_vw{3:.3f}_t{4:.3f}.pdf".format(*inputs[idx][0:5])
	print(figname)
	#plt.savefig("intp_figures/rf_TP_wind2_md{0:.4f}_vd{1:.3f}_mw{2:.4f}_vw{3:.3f}_t{4: .3f}.pdf".format(*inputs[idx][0:5]))
	plt.savefig(figname)
