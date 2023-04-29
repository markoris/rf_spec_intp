import glob, sys
import numpy as np
import metzger2017 as m17
import spectra_interpolator as si
from natsort import natsorted
import corner
import matplotlib.pyplot as plt

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

def rift_parameter_uncertainty(spectra, t, times_orig, trim_wavs=False, metz_model=False):

    import RIFT.integrators.MonteCarloEnsemble as monte_carlo_integrator

    if metz_model:
        tdays, Ltot, flux, _ = m17.calc_lc(t_ini, times_orig[t]+dt, dt, m, v, beta, k, wav_trim=trim_wavs)
        flux = flux[:, -2]*1e-8 # scales metzger model to our units per Angstrom instead of per cm
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
        import datetime
        import numexpr as ne
        
        #val =100 -0.5*(np.sum(x,axis=-1)**2)
        #val = -0.5*(np.sum(x**2,axis=-1))
        inputs = np.c_[x, np.ones(len(x))*times_orig[t]]
#        print('inputs shape = ', inputs.shape)
        out = intp.evaluate(inputs, ret_out=True)
#        print('out shape = ', out.shape)
        #out = intp.evaluate(np.c_[x, np.ones(len(x))*times_orig[t]], ret_out=True)
        if metz_model: out += flux
        out /= (4e6)**2 # scaling 40 Mpc source distance with source assumed emitting from 10 pc

        # take log of md, mw, and time
#        start = datetime.datetime.now()
#        inputs[:, [0, 2, 4]] = np.log10(inputs[:, [0, 2, 4]])
#        pred = np.array([tree.predict(inputs) for tree in intp.intp.rfr])
#
#        std_spec = intp.std_spec
#        mu_spec = intp.mu_spec
#	
#        pred = ne.evaluate('pred*std_spec')
#        pred = ne.evaluate('pred+mu_spec')
#        
#        pred = ne.evaluate('10**pred')
#        pred = ne.evaluate('pred / (4e6)**2')
#        
#        pred_err = pred.std(axis=0)
#        print(pred_err.mean())
#        print('STD DEV END-TO-END TAKES ', datetime.datetime.now()-start)

        obs_error_factor = 3
        val = -0.5*np.sum(((obs[mask, 1]-out[:, mask])**2/(pred_err[:, mask]**2 + (obs_error_factor*obs[mask, 2])**2)), axis=1)
        del pred, pred_err
        val += 40*len(mask)
        #val -= np.max(val)
        print(val)
        print(len(np.where(val>0)[0]))
        return val

    #residuals = residual_function(pred)

    integrator = monte_carlo_integrator.integrator(dim, bounds, gmm_dict, ncomp, proc_count=None, use_lnL=True, prior=prior, return_lnI=True, temper_log=True)
    integrator.integrate(residual_function, min_iter=100, max_iter=250, progress=False, epoch=2, use_lnL=True, return_lnI=True, temper_log=True, verbose=True)
    
    int_samples = integrator.cumulative_samples
    lnL = residual_function(int_samples)
    lnL -= np.max(lnL)
    p = integrator.cumulative_p
    p_s = integrator.cumulative_p_s
    my_random_number = np.random.randint(999)
    savedir = 'theta00deg_obs_plus_model_err'
    np.savetxt('/lustre/scratch4/turquoise/mristic/rift_runs/%s/rift_samples_%d_t%g.dat' % (savedir, my_random_number, times_orig[t]), np.c_[int_samples, lnL, p, p_s])
    
    weights = np.exp(lnL)*p/p_s
    
    corner.corner(int_samples, weights=weights)
    plt.savefig('/lustre/scratch4/turquoise/mristic/rift_runs/%s/rift_samples_%d_t%g.pdf' % (savedir, my_random_number, times_orig[t]))
    
trim_wavs = True
metz_model = False
use_rf_err = False
wavs_supernu = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # from cm to microns (via 1e4 scaling factor)
if trim_wavs: wavs_supernu = wavs_supernu[np.where((wavs_supernu > 0.39) & (wavs_supernu < 2.4))[0]]

# give time input rounded to nearest day
# i.e. 1.43 days -> 1, 2.42 days -> 2, ..., 10.4 days -> 10
t = int(sys.argv[1]) - 1

at2017gfo_spectra, times_orig = load_obs_data('binned_at2017gfo_spectra/*.dat')
intp = load_interpolator('/lustre/scratch4/turquoise/mristic/knsc1_active_learning/*spec*', \
                         '/lustre/scratch4/turquoise/mristic/h5_data/TP_wind2_spectra.h5', \
                         '/lustre/scratch4/turquoise/mristic/rf_spec_intp_optim_trimdata_theta00deg.joblib')
                         #'/lustre/scratch4/turquoise/mristic/rf_spec_intp_theta30deg.joblib')

print('t = ', times_orig[t])
rift_parameter_uncertainty(at2017gfo_spectra, t, times_orig=times_orig, trim_wavs=trim_wavs, metz_model=False)
