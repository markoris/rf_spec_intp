import os
import numpy as np
import spectra_interpolator as si

# fixed time, fixed angle interpolation
##intp = si.intp(seed=100, rf=True, n_estimators=500)
##intp.load_data('/net/scratch4/mristic/knsc1_active_learning/*spec*', '/net/scratch4/mristic/h5_data/TP_wind2_spectra.h5', t_max=21, theta=0, short_wavs=False)
##intp.create_test_set(size=5)
##intp.preprocess()
##intp.train()
##intp.evaluate()
##intp.make_plots()
##intp.save(compress=True)

# free time, fixed angle interpolation
intp = si.intp(rf=True)
intp.load_data('/net/scratch4/mristic/knsc1_active_learning/*spec*', '/net/scratch4/mristic/h5_data/TP_wind2_spectra.h5', t_max=None, theta=0, short_wavs=True)
#intp.load_data('/net/scratch4/mristic/knsc1_active_learning/*spec*', '/net/scratch4/mristic/h5_data/TP_wind2_spectra_t10p4d.h5', t_max=None, theta=0, short_wavs=True)
intp.append_input_parameter(intp.times, 1)
intp.create_test_set(size=5)
intp.preprocess()
intp.train()
intp.save(name='/net/scratch4/mristic/rf_spec_intp_theta00deg_shortwavs.joblib')
intp.evaluate()
intp.make_plots()

# fixed time, free angle interpolation
##intp = si.intp(seed=100, rf=True)
##intp.load_data('/home/marko.ristic/lanl/knsc1_active_learning/*spec*', './h5_data/TP_wind2_spectra.h5', t_max=21, theta=None, short_wavs=False, downsample_theta=True)
##intp.append_input_parameter(intp.angles, 2) # axis 1 = time, axis 2 = angle (originally angle = axis 3, but for fixed time we have -1 dimension)
##intp.create_test_set(size=5)
##intp.preprocess()
##intp.train()
##intp.evaluate()
##intp.make_plots()

# free time, free angle interpolation # CURRENTLY TIMES OUT ON LANL HPC CLUSTER
##intp = si.intp(seed=100, rf=True)
##intp.load_data('/net/scratch4/mristic/knsc1_active_learning/*spec*', '/net/scratch4/mristic/h5_data/TP_wind2_spectra.h5', t_max=None, theta=None, short_wavs=True, downsample_theta=True)
##intp.append_input_parameter(intp.times, 1) # [N, time, wav, angle] -> [N, wav, angle]
##intp.append_input_parameter(intp.angles, 2) # axis 1 = time, axis 2 = angle (originally angle = axis 3, but for fixed time we have -1 dimension)
##intp.create_test_set(size=10)
##intp.preprocess()
##intp.train()
##intp.save(name='/net/scratch4/mristic/rf_spec_intp_5d.joblib')

### free time, fixed angle interpolation
##intp = si.intp(seed=100, nn=True, learning_rate_init=0.1, max_iter=500)
##intp.load_data('/home/marko.ristic/lanl/knsc1_active_learning/*spec*', './h5_data/TP_wind2_spectra.h5', t_max=None, theta=None, short_wavs=False, downsample_theta=True)
##intp.append_input_parameter(intp.times, 1)
##intp.append_input_parameter(intp.angles, 2) # axis 1 = time, axis 2 = angle (originally angle = axis 3, but for fixed time we have -1 dimension)
##intp.create_test_set(size=5)
##intp.preprocess()
##intp.train()
##intp.evaluate()
##intp.make_plots()
