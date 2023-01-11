import glob, sys
import numpy as np
import metzger2017 as m17
import spectra_interpolator as si
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patheffects as PathEffects

# -- loading SuperNu-binned AT2017gfo observational spectra

at2017gfo_spectra = np.array(glob.glob('binned_at2017gfo_spectra/*.dat')) # [10, 1024] array
at2017gfo_spectra = natsorted(at2017gfo_spectra)
times_orig = [float(filename.split('/')[-1].split('+')[1].split('_')[0][:-1]) for filename in at2017gfo_spectra]
times_orig = np.array(times_orig).reshape(-1, 1)

# -- loading SuperNu simulations with best-fit parameters

data_2c = np.loadtxt('Run_TP_dyn_all_lanth_wind2_all_md0.030643_vd0.175067_mw0.015214_vw0.163226_spec_2023-01-09.dat')
data_3c = np.loadtxt('Run_TPS_dyn_all_lanth_wind2_all_wind1_all_md0.030643_vd0.175067_mw0.015214_vw0.163226_mth0.003_vth0.005_spec_2023-01-09.dat')
data_3c_metzger = np.loadtxt('Run_TP_dyn_all_lanth_wind2_all_md0.030643_vd0.175067_mw0.015214_vw0.163226_spec_2023-01-09.dat')
data_3c_Ye0p5 = np.loadtxt('Run_TPS_dyn_all_lanth_wind2_all_v0.05_Ye0.50_md0.030643_vd0.175067_mw0.015214_vw0.163226_mth0.003_vth0.005_spec_2023-01-11.dat')
data_3c_Ye0p5_peanut = np.loadtxt('Run_TPP_dyn_all_lanth_wind2_all_v0.05_Ye0.50_md0.030643_vd0.175067_mw0.015214_vw0.163226_mth0.003_vth0.005_spec_2023-01-11.dat')
data_3c_Ye0p5_peanut_fast = np.loadtxt('Run_TPP_dyn_all_lanth_wind2_all_v0.05_Ye0.50_md0.030643_vd0.175067_mw0.015214_vw0.163226_mth0.003_vth0.025_spec_2023-01-11.dat')
data_2c = np.array(np.split(data_2c, data_2c.shape[0]/1024))
data_3c = np.array(np.split(data_3c, data_3c.shape[0]/1024))
data_3c_metzger = np.array(np.split(data_3c_metzger, data_3c_metzger.shape[0]/1024))
data_3c_Ye0p5 = np.array(np.split(data_3c_Ye0p5, data_3c_Ye0p5.shape[0]/1024))
data_3c_Ye0p5_peanut = np.array(np.split(data_3c_Ye0p5_peanut, data_3c_Ye0p5_peanut.shape[0]/1024))
data_3c_Ye0p5_peanut_fast = np.array(np.split(data_3c_Ye0p5_peanut_fast, data_3c_Ye0p5_peanut_fast.shape[0]/1024))

# -- Metzger model parameters for post-facto direct flux addition (no re-processing, no radiative transfer)

dt = 0.01 # days
t_ini = 0.01 # days
beta = 3
m = 3e-3
v = 0.005
k = 1

t_a = 0.125 # spectral start time
t_b = 20.749 # spectral end time FIXME this is only temporary, need to remake .hdf5 to include time up to 66 iteration
sim_times = np.logspace(np.log10(t_a), np.log10(t_b), 60) # this should cover the full spectral possibilities range

wavs_supernu = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # from cm to microns (via 1e4 scaling factor)

for t in range(len(times_orig)):
	print('t = ', times_orig[t])
	obs = np.loadtxt(at2017gfo_spectra[t])
	t_idx = np.argmin(np.abs(times_orig[t]-sim_times))
	sim_2c = data_2c[t_idx, :, 2]
	sim_3c = data_3c[t_idx, :, 2]
	sim_3c_metzger = data_3c_metzger[t_idx, :, 2]
	sim_3c_Ye0p5 = data_3c_Ye0p5[t_idx, :, 2]
	sim_3c_Ye0p5_peanut = data_3c_Ye0p5_peanut[t_idx, :, 2]
	sim_3c_Ye0p5_peanut_fast = data_3c_Ye0p5_peanut_fast[t_idx, :, 2]
	sim_2c *= 54/(4e6)**2
	sim_3c *= 54/(4e6)**2
	sim_3c_metzger *= 54/(4e6)**2
	tdays, Ltot, flux, _ = m17.calc_lc(t_ini, times_orig[t]+dt, dt, m, v, beta, k)
	flux = flux[:, -2]*1e-8
	sim_3c_metzger += flux
	sim_3c_Ye0p5 *= 54/(4e6)**2
	sim_3c_Ye0p5_peanut *= 54/(4e6)**2
	sim_3c_Ye0p5_peanut_fast *= 54/(4e6)**2
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

	for i in range(len(cutoffs)-1):
		plt.plot(obs[cutoffs[i]:cutoffs[i+1], 0], obs[cutoffs[i]:cutoffs[i+1], 1], c='k')
	plt.plot(wavs_supernu, sim_2c, '--', c='red', label='2c')
	plt.plot(wavs_supernu, sim_3c_metzger, '--', c='brown', label='3cMetzger')
	plt.plot(wavs_supernu, sim_3c, '--', c='blue', label='3cw1Sslow')
	plt.plot(wavs_supernu, sim_3c_Ye0p5, '--', c='orange', label='3cNiSslow')
	plt.plot(wavs_supernu, sim_3c_Ye0p5_peanut, '--', c='green', label='3cNiPslow')
	plt.plot(wavs_supernu, sim_3c_Ye0p5_peanut_fast, '--', c='purple', label='3cNiPfast')
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
	plt.legend(ncol=2, loc="lower right")
	plt.savefig('best_fit_sim_vs_obs/t_%g.pdf' % times_orig[t])
