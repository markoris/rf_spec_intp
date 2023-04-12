import corner
import glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import metzger2017 as m17
import spectra_interpolator as si
from natsort import natsorted
from matplotlib import ticker
import matplotlib.patheffects as PathEffects

def load_obs_data(path_to_binned_obs):
    spectra = np.array(glob.glob(path_to_binned_obs)) # [10, 1024] array
    spectra = natsorted(spectra)
    times = [float(filename.split('/')[-1].split('+')[1].split('_')[0][:-1]) for filename in spectra]
    times = np.array(times).reshape(-1, 1)
    return spectra, times

def load_interpolator(path_to_sims, path_to_h5, path_to_model):
    intp = si.intp(rf=True)
    intp.load_data(path_to_sims, path_to_h5, t_max=None, theta=0, trim_dataset=True)
    intp.append_input_parameter(intp.times, 1)
    intp.preprocess()
    intp.load(path_to_model, fixed_angle=True)
    return intp 

def least_squares(t, times_orig, spectra, prediction, trim_wavs=False, metz_model=False, save_L=True):
#def least_squares(t, times_orig, spectra, prediction, err_prediction, trim_wavs=False, metz_model=False, save_L=True):
    # take every 10th index starting with t, such that 0 = 1.43 days, 1 = 2.42 days, ..., 9 = 10.4 days
    obs = np.loadtxt(spectra[t])
    if trim_wavs:
        wav_trim = np.where((obs[:, 0] > 0.39*1e4) & (obs[:, 0] < 2.4*1e4))[0]
        obs = obs[wav_trim, :]
    mask = np.where(obs[:, 1] > 0)[0]
    print(len(mask), '/%d wav bins used' % obs.shape[0])
    chi2 = np.sum(((obs[mask, 1]-prediction[mask])/obs[mask, 2])**2)*1/len(mask)
    obs = obs[mask, :]
    print(chi2) 
    return obs, chi2, mask

def make_plots(obs, best_spec, wavs_supernu, times_orig):
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
    text_locs = {"g": 0.09, "r": 0.25, "i": 0.35, "z": 0.44, "y": 0.51, "J": 0.61, "H": 0.75, "K": 0.92}
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
    plt.plot(wavs_supernu, best_spec, c='r', label=r'$F_{\lambda, \rm intp}$')
    #plt.fill_between(wavs_supernu, best_spec-best_spec_err, best_spec+best_spec_err, alpha=0.3, c='r')
    plt.title('%g d' % times_orig[t], loc="left", x=0.02, y=1.0, pad=-40)
    plt.plot([], [], c='k', label=r'$F_{\lambda, \rm AT2017gfo}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([0.39, 2.4])
    #plt.ylim([1e-19, 5e-17])
    plt.ylim([1e-19, np.max(obs[:, 1])*5])
    plt.gca().set_xticks(np.array([0.5, 1, 2]))
    plt.gca().set_xticklabels(np.array([0.5, 1, 2]))
    plt.gca().minorticks_off()
    plt.ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')
    plt.xlabel(r'$\lambda$ [$\mu$m]')
    plt.legend(loc='lower right')
    plt.savefig('figs_from_lowest_residuals/intp_t_%gd.pdf' % times_orig[t])
    plt.close()
    return

at2017gfo_spectra, times_orig = load_obs_data('binned_at2017gfo_spectra/*.dat')

best_fit_params =  [[10**(-1.14), 0.12, 10**(-1.61), 0.06],
                    [10**(-1.34), 0.14, 10**(-1.99), 0.08],
                    [10**(-1.08), 0.13, 10**(-1.63), 0.16],
                    [10**(-1.53), 0.13, 10**(-1.58), 0.22],
                    [10**(-1.65), 0.21, 10**(-1.73), 0.07],
                    [10**(-1.64), 0.30, 10**(-1.79), 0.08],
                    [10**(-1.70), 0.30, 10**(-1.78), 0.05],
                    [10**(-1.92), 0.15, 10**(-1.58), 0.08],
                    [10**(-1.79), 0.26, 10**(-1.84), 0.24],
                    [10**(-1.58), 0.25, 10**(-2.07), 0.25]]
# these parameters come from running plot_corner.py on the RIFT samples in $SCRATCH/rift_runs  
best_fit_params = np.c_[best_fit_params, times_orig]

intp = load_interpolator('/lustre/scratch4/turquoise/mristic/knsc1_active_learning/*spec*', \
                         '/lustre/scratch4/turquoise/mristic/h5_data/TP_wind2_spectra.h5', \
                         '/lustre/scratch4/turquoise/mristic/rf_spec_intp_optim_trimdata_theta00deg.joblib')
out = intp.evaluate(best_fit_params, ret_out=True) # shape [n_draws*len(times), 1024]
out /= (4e6)**2 # scaling 40 Mpc source distance with source assumed emitting from 10 pc

print(out.shape)

trim_wavs=True
metz_model=False
wavs_supernu = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # from cm to microns (via 1e4 scaling factor)
if trim_wavs: wavs_supernu = wavs_supernu[np.where((wavs_supernu > 0.39) & (wavs_supernu < 2.4))[0]]

masks = []

for t in range(len(times_orig)):
    print('t = ', times_orig[t])
    obs, chi2, mask = least_squares(t, times_orig, at2017gfo_spectra, out[t, :], trim_wavs=trim_wavs, metz_model=metz_model)

    masks.append(len(mask))

    best_spec = out[t, :]

    make_plots(obs, best_spec, wavs_supernu, times_orig)

best_fit_params[:, 0] = np.log10(best_fit_params[:, 0]) # take log of dynamical mass
best_fit_params[:, 2] = np.log10(best_fit_params[:, 2]) # take log of wind mass
best_fit_params = np.average(best_fit_params[:, :4], weights=masks, axis=0)
print('Recovered Parameters: md = %f, mw = %f, vd = %f, vw = %f' % (10**best_fit_params[0], 10**best_fit_params[2], best_fit_params[1], best_fit_params[3]))
