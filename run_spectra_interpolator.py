import os
import numpy as np
import spectra_interpolator as si

# fixed time, fixed angle interpolation
##intp = si.intp_rf(seed=100)
##intp.load_data('/home/marko.ristic/lanl/knsc1_active_learning/*spec*', './h5_data/TP_wind2_spectra.h5', t_max=21, short_wavs=False)
##intp.create_test_set(size=5)
##intp.preprocess()
##intp.train(rf=True)
##intp.evaluate()
##intp.make_plots()

# free time, fixed angle interpolation
##intp = si.intp_rf(seed=100)
##intp.load_data('/net/scratch4/mristic/knsc1_active_learning/*spec*', '/net/scratch4/mristic/h5_data/TP_wind2_spectra.h5', t_max=None, theta=0, short_wavs=False)
##intp.append_input_parameter(intp.times, 1)
##intp.create_test_set(size=5)
##intp.preprocess()
##intp.train(rf=True)
##intp.evaluate()
##intp.make_plots()

# fixed time, free angle interpolation
##intp = si.intp_rf(seed=100)
##intp.load_data('/net/scratch4/mristic/knsc1_active_learning/*spec*', '/net/scratch4/mristic/h5_data/TP_wind2_spectra.h5', t_max=21, theta=None, short_wavs=False, downsample_theta=True)
##intp.append_input_parameter(intp.angles, 2) # axis 1 = time, axis 2 = angle (originally angle = axis 3, but for fixed time we have -1 dimension)
##intp.create_test_set(size=5)
##intp.preprocess()
##intp.train(rf=True)
##intp.evaluate()
##intp.make_plots()

# free time, free angle interpolation # CURRENTLY TIMES OUT ON LANL HPC CLUSTER
intp = si.intp_rf(seed=100, verbose=True, debugging=False)
intp.load_data('/net/scratch4/mristic/knsc1_active_learning/*spec*', '/net/scratch4/mristic/h5_data/TP_wind2_spectra.h5', t_max=None, theta=None, short_wavs=False, downsample_theta=True)
intp.append_input_parameter(intp.times, 1, var='time') # [N, time, wav, angle] -> [N, wav, angle]
intp.append_input_parameter(intp.angles, 2, var='angle') # axis 1 = time, axis 2 = angle (originally angle = axis 3, but for fixed time we have -1 dimension)
intp.create_test_set(size=10)
intp.preprocess()
intp.train(rf=True)
intp.evaluate()
intp.make_plots()

