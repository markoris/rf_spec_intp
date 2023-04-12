import glob, sys
import numpy as np
import metzger2017 as m17
import spectra_interpolator as si
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patheffects as PathEffects
import corner

# -- Metzger model parameters for post-facto direct flux addition (no re-processing, no radiative transfer)

dt = 0.01 # days
t_ini = 0.01 # days
beta = 3
m = 3e-3
v = 0.005
k = 1

def load_obs_data(path_to_binned_obs):
    spectra = np.array(glob.glob(path_to_binned_obs)) # [10, 1024] array
    spectra = natsorted(spectra)
    times = [float(filename.split('/')[-1].split('+')[1].split('_')[0][:-1]) for filename in spectra]
    times = np.array(times).reshape(-1, 1)
    return spectra, times

def generate_samples(n_samples, times_orig):
    param_mins = [-3, 0.05, -3, 0.05]
    param_maxes = [-1, 0.3, -1, 0.3]
    
    draws_orig = np.random.uniform(param_mins, param_maxes, size=(n_samples, len(param_mins)))
    draws_orig[:, 0], draws_orig[:, 2] = 10**draws_orig[:, 0], 10**draws_orig[:, 2]

    # save base samples

    np.savetxt('samples.dat', draws_orig)
    
    draws = np.repeat(draws_orig, len(times_orig), axis=0) # 100000x4 array, 10000x4 array repeated 10 times for 10 observations
    times = np.tile(times_orig, (n_samples, 1)) # 100000x1 array, 10x1 array repeated 10000 times for 10000 draws
    inputs = np.concatenate((draws, times), axis=1) # 100000x5 array, 4 ejecta parameters and 1 time parameter
    return draws_orig, inputs 

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
    pred = prediction[t::10]
    #pred_err = err_prediction[t::10]
    if metz_model:
        tdays, Ltot, flux, _ = m17.calc_lc(t_ini, times_orig[t]+dt, dt, m, v, beta, k, wav_trim=trim_wavs)
        flux = flux[:, -2]*1e-8 # scales metzger model to our units per Angstrom instead of per cm
        pred += flux
    obs = np.loadtxt(spectra[t])
    if trim_wavs:
        wav_trim = np.where((obs[:, 0] > 0.39*1e4) & (obs[:, 0] < 2.4*1e4))[0]
        obs = obs[wav_trim, :]
    mask = np.where(obs[:, 1] > 0)[0]
    print(len(mask), '/%d wav bins used' % obs.shape[0])
    residuals = np.sum(((obs[mask, 1]-pred[:, mask])/obs[mask, 2])**2, axis=1)*1/len(mask)
    #residuals = np.sum((0.5*(obs[mask, 1]-pred[:, mask])/np.sqrt(obs[mask, 2]**2+pred_err**2))**2, axis=1)*1/len(mask)
    if save_L:
        np.savetxt('/lustre/scratch4/turquoise/mristic/at2017gfo_likelihoods_t%g.dat' % times_orig[t], residuals, fmt="%g", header="Likelihood for AT2017gfo spectrum at %g days" % times_orig[t])
    obs = obs[mask, :] 
    return obs, pred, residuals, mask 
    #return obs, pred, pred_err, residuals, mask 

def get_best_match(samples, prediction, residuals):
#def get_best_match(samples, prediction, err_prediction, residuals):
    idx_max_L = np.argmin(residuals)
    print('Lowest chi^2 residual value: ', residuals[idx_max_L])
    best_params = samples[idx_max_L]
    best_spec = prediction[idx_max_L]
    #best_spec_err = err_prediction[idx_max_L]
    return best_params, best_spec
    #return best_params, best_spec, best_spec_err 

def rift_parameter_uncertainty(prediction, spectra, t, times_orig, trim_wavs=False, metz_model=False):

    import RIFT.integrators.MonteCarloEnsemble as monte_carlo_integrator

    pred = prediction[t::10]
    if metz_model:
        tdays, Ltot, flux, _ = m17.calc_lc(t_ini, times_orig[t]+dt, dt, m, v, beta, k, wav_trim=trim_wavs)
        flux = flux[:, -2]*1e-8 # scales metzger model to our units per Angstrom instead of per cm
        pred += flux
    obs = np.loadtxt(spectra[t])
    if trim_wavs:
        wav_trim = np.where((obs[:, 0] > 0.39*1e4) & (obs[:, 0] < 2.4*1e4))[0]
        obs = obs[wav_trim, :]
    mask = np.where(obs[:, 1] > 0)[0]
    
    dim = 4
    bounds = {(0,): np.array([[0.001, 0.1]]), (1,): np.array([[0.05, 0.3]]), (2,): np.array([[0.001, 0.1]]), (3,): np.array([[0.05, 0.3]])}
    gmm_dict = {(0,): None, (1,):None, (2,): None, (3,): None}
    ncomp = 1

    def prior(x):
        return np.ones_like(x[:,0])

    def residual_function(x):
        #val =100 -0.5*(np.sum(x,axis=-1)**2)
        #val = -0.5*(np.sum(x**2,axis=-1))
        out = intp.evaluate(np.c_[x, np.ones(len(x))*times_orig[t]], ret_out=True)
        out /= (4e6)**2 # scaling 40 Mpc source distance with source assumed emitting from 10 pc
        val = -0.5*np.sum(((obs[mask, 1]-out[:, mask])/obs[mask, 2])**2, axis=1)
        val += 40*len(mask)
        #val -= np.max(val)
        print(val)
        print(len(np.where(val>0)[0]))
        return val

    #residuals = residual_function(pred)

    integrator = monte_carlo_integrator.integrator(dim, bounds, gmm_dict, ncomp, proc_count=None, use_lnL=True, prior=prior, return_lnI=True, temper_log=True)
    integrator.integrate(residual_function, min_iter=20, max_iter=20, progress=False, epoch=2, use_lnL=True, return_lnI=True, temper_log=True, verbose=True)
    
    int_samples = integrator.cumulative_samples
    lnL = residual_function(int_samples)
    lnL -= np.max(lnL)
    p = integrator.cumulative_p
    p_s = integrator.cumulative_p_s
    my_random_number = np.random.randint(50)
    np.savetxt('rift_runs/rift_samples_%d_t%g.dat' % (my_random_number, times_orig[t]), np.c_[int_samples, lnL, p, p_s])
    
    weights = np.exp(lnL)*p/p_s
    
    corner.corner(int_samples, weights=weights)
    plt.savefig('rift_runs/rift_samples_%d_t%g.pdf' % (my_random_number, times_orig[t]))
    

def parameter_uncertainty(samples, residuals, mask, t, times_orig):
    # undo len(mask) division from least squares calculation
    residuals *= len(mask)
    # fix such that largest value of L is 1
    print(residuals.min(), residuals.mean(), residuals.max())
    residuals -= np.min(residuals)
    print(residuals.min(), residuals.mean(), residuals.max())
    # calculate L
    L = np.exp(-0.5*residuals)
    # find sorting indices for ascending L such that L_N is largest L value
    idx_sort_L = np.argsort(L)
    # sort x_k and L by increasing L
    samples = samples[idx_sort_L, :]
    L = L[idx_sort_L]
    # calculate cumulative sum S
    S = np.sum(L)
    # define p_k which finds index of simulation which matches the p_y confidence interval bound 
    def p_k(p_y, L, S):
        p = 0
        for i in range(L.shape[0]):
            p += L[i]/S
            if p > p_y: return i
    
    ## identify p_k(x) = 0.05 location
    #p_k0p05 = p_k(0.05, L, S)
    # identify p_k(x) = 0.05 location
    p_k0p16 = p_k(0.16, L, S)
    ## identify p_k(x) = 0.95 location
    #p_k0p95 = p_k(0.95, L, S)
    # identify p_k(x) = 0.84 location
    p_k0p84 = p_k(0.84, L, S)
    # find the ejecta parameters which fall within this range
    CI_68 = samples[p_k0p16:p_k0p84, :4]
    # find lower limits
    x_mins = np.min(CI_68, axis=0)
    # find upper limits
    x_maxs = np.max(CI_68, axis=0)

    print('lower limits for md, vd, mw, vw: ', x_mins)
    print('upper limits for md, vd, md, vw: ', x_maxs)

    CI_68[:, 0] = np.log10(CI_68[:, 0])
    CI_68[:, 2] = np.log10(CI_68[:, 2])

    print(CI_68.shape)
 
    # plot posterior of 90% CI
    plt.rc('font', size=12)
    figure = corner.corner(
    CI_68,
    labels=[
        r"$M_d$",
        r"$V_d$",
        r"$M_w$",
        r"$V_w$",
    ],
    quantiles=[0, 1],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)
    plt.savefig('t_%gd_posteriors.pdf' % times_orig[t])
    plt.close()
    return

def make_plots(obs, best_spec, wavs_supernu, times_orig):
#def make_plots(obs, best_spec, best_spec_err, wavs_supernu, times_orig):
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
    
trim_wavs = True
metz_model = False
use_rf_err = False
wavs_supernu = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # from cm to microns (via 1e4 scaling factor)
if trim_wavs: wavs_supernu = wavs_supernu[np.where((wavs_supernu > 0.39) & (wavs_supernu < 2.4))[0]]
n_samples = 100000
#n_samples = 100
at2017gfo_spectra, times_orig = load_obs_data('binned_at2017gfo_spectra/*.dat')
samples, inputs = generate_samples(n_samples, times_orig)
intp = load_interpolator('/lustre/scratch4/turquoise/mristic/knsc1_active_learning/*spec*', \
                         '/lustre/scratch4/turquoise/mristic/h5_data/TP_wind2_spectra.h5', \
                         '/lustre/scratch4/turquoise/mristic/rf_spec_intp_optim_trimdata_theta00deg.joblib')
out = intp.evaluate(inputs, ret_out=True) # shape [n_draws*len(times), 1024]
out /= (4e6)**2 # scaling 40 Mpc source distance with source assumed emitting from 10 pc

# for uncertainty calculation, inputs are RF object, training inputs, test inputs
# for us, that is intp.rfr, intp.params, inputs 
#out_err = fci.random_forest_error(intp.intp.rfr, intp.params, inputs) # reports variance
#out_err = np.sqrt(out_err) # std dev

for t in range(len(times_orig)):
    print('t = ', times_orig[t])
    obs, pred, residuals, mask = least_squares(t, times_orig, at2017gfo_spectra, out, trim_wavs=trim_wavs, metz_model=metz_model)
    #obs, pred, pred_err, residuals, mask = least_squares(t, times_orig, at2017gfo_spectra, out, out_err, trim_wavs=trim_wavs, metz_model=metz_model)
  
    best_params, best_spec = get_best_match(samples, pred, residuals)
    #best_params, best_spec, best_spec_err = get_best_match(samples, pred, pred_err, residuals)
   
    print('Best recovered parameters for time %g: ' % times_orig[t], best_params)
    
    try:
        recov = np.c_[best_params.reshape(1, 4), np.array(len(mask)/1024).reshape(1, 1)]
        recovered_parameters = np.concatenate((recovered_parameters, recov), axis=0)
    except NameError:
        recov = np.c_[best_params.reshape(1, 4), np.array(len(mask)/1024).reshape(1, 1)]
        recovered_parameters = recov

    #rift_parameter_uncertainty(pred, at2017gfo_spectra, t, times_orig=times_orig, trim_wavs=True, metz_model=False)
    parameter_uncertainty(samples, residuals, mask, t, times_orig)

    make_plots(obs, best_spec, wavs_supernu, times_orig)
    #make_plots(obs, best_spec, best_spec_err, wavs_supernu, times_orig)

np.savetxt('recovered_parameters_from_intp.dat', recovered_parameters)
rcv_prm = np.loadtxt('recovered_parameters_from_intp.dat')
rcv_prm[:, 0] = np.log10(rcv_prm[:, 0]) # take log of dynamical mass
rcv_prm[:, 2] = np.log10(rcv_prm[:, 2]) # take log of wind mass
rcv_prm = np.average(rcv_prm[:, :4], weights=rcv_prm[:, 4], axis=0)
print('Recovered Parameters: md = %f, mw = %f, vd = %f, vw = %f' % (10**rcv_prm[0], 10**rcv_prm[2], rcv_prm[1], rcv_prm[3]))
