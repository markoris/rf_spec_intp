import numpy as np
import rf_interpolator as rfi

intp = rfi.intp_rf()
intp.load_data('/home/marko.ristic/lanl/knsc1_active_learning/*spec*', './h5_data/TP_wind2_spectra.h5', t_max=21)
intp.create_test_set(size=5)
