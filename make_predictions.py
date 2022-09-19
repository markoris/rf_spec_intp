import glob
import numpy as np
import spectra_interpolator as si

at2017gfo_spectra = np.array(glob.glob('/users/mristic/spec_intp_paper/AT2017gfo_spectra/flux_corrected_smoothed_joined_spectra/*.dat'))

times = [float(filename.split('/')[-1].split('+')[1].split('_')[0][:-5]) for filename in at2017gfo_spectra]
times = np.array(times).reshape(-1, 1)
#inputs = np.array([10**(-1.6), 0.17, 10**(-1), 0.18]) # EM paper
inputs = np.array([10**(-2), 0.25, 10**(-1.6), 0.23])  # r-process paper
inputs = np.tile(inputs, (len(at2017gfo_spectra), 1))
inputs = np.concatenate((inputs, times), axis=1)

intp = si.intp()
intp.load_data('/net/scratch4/mristic/knsc1_active_learning/*spec*', '/net/scratch4/mristic/h5_data/TP_wind2_spectra.h5', t_max=None, theta=0, short_wavs=True)
intp.append_input_parameter(intp.times, 1)
intp.preprocess()
intp.load('/net/scratch4/mristic/rf_spec_intp_theta00deg_shortwavs.joblib', fixed_angle=True)
intp.evaluate(inputs)

for idx in range(len(at2017gfo_spectra)):
	np.savetxt("at2017gfo_predictions/rf_out_TP_wind2_md{0:.4f}_vd{1:.3f}_mw{2:.4f}_vw{3:.3f}_t{4:.3f}.dat".format(*inputs[idx][0:5]), np.c_[intp.wavs, intp.prediction[idx]])
