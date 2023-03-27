import glob, sys
import numpy as np
import spectra_interpolator as si
from natsort import natsorted
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('theta')

args = parser.parse_args()

at2017gfo_spectra = np.array(glob.glob('binned_at2017gfo_spectra/*.dat')) # [10, 1024] array
at2017gfo_spectra = natsorted(at2017gfo_spectra)
times_orig = [float(filename.split('/')[-1].split('+')[1].split('_')[0][:-1]) for filename in at2017gfo_spectra]
times_orig = np.array(times_orig).reshape(-1, 1)

intp = si.intp(seed=12)
intp.load_data('/lustre/scratch4/turquoise/mristic/knsc1_active_learning/*spec*', '/lustre/scratch4/turquoise/mristic/h5_data/TP_wind2_spectra.h5', t_max=None, theta=float(args.theta), short_wavs=False)
intp.create_test_set(size=5)
intp.append_input_parameter(intp.times, 1)
intp.preprocess()
intp.load('/lustre/scratch4/turquoise/mristic/rf_spec_intp_theta%sdeg.joblib' % args.theta)

param = intp.params_test[-2][:4]
param = np.concatenate((param, intp.times[51].reshape(1))).reshape(1, -1) # 10-day time input
print(param)

sys.exit()

out = intp.evaluate(param, ret_out=True) # shape [n_draws*len(times), 1024]

out /= (4e6)**2 # scaling 40 Mpc source distance with source assumed emitting from 10 pc
np.savetxt('intpdeg%sdeg.dat' % args.theta, out.flatten())
