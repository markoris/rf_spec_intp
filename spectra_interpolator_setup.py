import os, sys
import argparse
import numpy as np
import spectra_interpolator as si

# APPEND INPUT PARAMETER: AXIS = 1 FOR TIME, AXIS = 2 FOR ANGLE
# FOR JUST TIME: intp.append_input_parameter(intp.times, 1) # [N, time, wav, angle] -> [N, wav, angle]
# FOR JUST ANGLE: intp.append_input_parameter(intp.angles, 3) # [N, time, wav, angle] -> [N, time, wav]
# FOR BOTH TIME AND ANGLE, IN THAT ORDER: intp.append_input_parameter(intp.angles, 2) # axis 1 = time, axis 2 = angle (originally angle = axis 3, but for fixed time we have -1 dimension)

parser = argparse.ArgumentParser(description='set up random forest interpolator')
parser.add_argument('intp_params', help='which parameters to use for interpolation. "ejecta" assumes only 4D interpolation for all times at theta=0. "time" assumes 5D ejecta+time interpolation, requires a value for fixed angle between 0-180 degrees. "angle" assumes 5D ejecta+angle interpolation, requires a value for fixed time between 0.125 and 21.75 days. "all" assumes ejecta+time+angle.')
parser.add_argument('--fixed_value', default=None, help='if using 5D interpolation, specify value for the parameter (time or angle) NOT being considered as an input. e.g., if time is being used for interpolation, specify fixed angle value, or if angle being used, specify fixed tiem value at which spectra are evaluated')
parser.add_argument('--trim_dataset', action='store_true', help='if True, restricts times to between 1.4 < t < 10.4 days and wavelengths to 0.39 < lambda < 2.39 microns')
parser.add_argument('--test_set_size', type=int, default=5)

args = parser.parse_args()

print(args.intp_params)

scratch = '/lustre/scratch4/turquoise/mristic/'

if args.intp_params == "ejecta":
    intp = si.intp(seed=12, rf=True)
    intp.load_data(scratch+'knsc1_active_learning/*spec*', scratch+'h5_data/TP_wind2_spectra.h5', t_max=21, theta=0, trim_dataset=args.trim_dataset)
    intp.create_test_set(size=args.test_set_size)
    intp.preprocess()
    intp.train()
    intp.save(name=scratch+'rf_spec_intp_optim_trimdata.joblib')
    intp.evaluate()
    intp.make_plots()

elif args.intp_params == "time":
    # free time, fixed angle interpolation
    if args.fixed_value is None:
        print('please provide a fixed angle value, 0-180, at which the spectra will be evaluated')
        sys.exit()
    model_name = 'rf_spec_intp_optim'
    if args.trim_dataset: model_name += '_trimdata'
    fixed_value = str(args.fixed_value)
    if len(fixed_value) < 2: fixed_value = '0'+fixed_value
    model_name += '_theta%sdeg.joblib' % (fixed_value)
    intp = si.intp(rf=True, seed=12)
    intp.load_data(scratch+'knsc1_active_learning/*spec*', scratch+'h5_data/TP_wind2_spectra.h5', t_max=None, theta=int(args.fixed_value), trim_dataset=args.trim_dataset)
    intp.create_test_set(size=args.test_set_size)
    intp.append_input_parameter(intp.times, 1)
    intp.preprocess()
    intp.train()
    intp.save(name=scratch+model_name)
    intp.evaluate()
    intp.make_plots()
    
elif args.intp_params == "angle":
    # free angle, fixed time interpolation
    if args.fixed_value is None:
        print('please provide a fixed time value, 0.125-21.75, at which the spectra will be evaluated')
        sys.exit()
    model_name = 'rf_spec_intp_optim'
    if args.trim_dataset: model_name += '_trimdata'
    model_name += '_time%sdays.joblib' % (args.fixed_value.replace('.', 'p'))
    intp = si.intp(rf=True, seed=12)
    intp.load_data(scratch+'knsc1_active_learning/*spec*', scratch+'h5_data/TP_wind2_spectra.h5', t_max=float(args.fixed_value), theta=None, trim_dataset=args.trim_dataset)
    intp.create_test_set(size=args.test_set_size)
    intp.append_input_parameter(intp.angles, 2)
    intp.preprocess()
    intp.train()
    intp.save(scratch+model_name)
    intp.evaluate()
    intp.make_plots()

elif args.intp_params == "all":
    # free time, free angle interpolation # CURRENTLY TIMES OUT ON LANL HPC CLUSTER
    model_name = 'rf_spec_intp_optim'
    if args.trim_dataset: model_name += '_trimdata'
    model_name += '_6d.joblib'
    intp = si.intp(seed=12, rf=True)
    intp.load_data(scratch+'knsc1_active_learning/*spec*', scratch+'h5_data/TP_wind2_spectra.h5', t_max=None, theta=None, trim_dataset=args.trim_dataset)
    intp.create_test_set(size=10)
    intp.append_input_parameter(intp.times, 1) # [N, time, wav, angle] -> [N, wav, angle]
    intp.append_input_parameter(intp.angles, 2) # axis 1 = time, axis 2 = angle (originally angle = axis 3, but for fixed time we have -1 dimension)
    intp.preprocess()
    intp.train()
    intp.save(scratch+model_name)
    intp.evaluate()
    intp.make_plots()
