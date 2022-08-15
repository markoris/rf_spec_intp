import os
import numpy as np
import rf_interpolator as rfi

os.environ['OMP_NUM_THREADS'] = "12"

# fixed time, fixed angle interpolation
##intp = rfi.intp_rf(seed=100)
##intp.load_data('/home/marko.ristic/lanl/knsc1_active_learning/*spec*', './h5_data/TP_wind2_spectra.h5', t_max=21, short_wavs=False)
##intp.create_test_set(size=5)
##intp.preprocess()
##intp.train(rf=True)
##intp.evaluate()
##intp.make_plots()

# free time, fixed angle interpolation
intp = rfi.intp_rf(seed=100)
intp.load_data('/home/marko.ristic/lanl/knsc1_active_learning/*spec*', './h5_data/TP_wind2_spectra.h5', t_max=None, theta=0, short_wavs=False)
intp.append_input_parameter(intp.times, 1)
intp.create_test_set(size=5)
intp.preprocess()
intp.train(rf=True)
intp.evaluate()
intp.make_plots()

