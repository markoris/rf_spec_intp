import sys, glob
import numpy as np
import matplotlib.pyplot as plt
import senni, slick

plt.rc('font', size=40)
plt.rc('lines', lw=4)

def get_params(string):
	name_parse = string.split('/')[-1].split('_')
	params = np.array([name_parse[1], name_parse[5][-1], name_parse[7][2:], name_parse[8][2:], name_parse[9][2:], name_parse[10][2:]])
	return params

files = np.array(glob.glob('/net/scratch5/mristic/knsc1_active_learning/*spec*.dat')) # majority of data files from Zenodo
#files2 = np.array(glob.glob('/usr/projects/w20_knspectra/*TP*wind2*spec*.dat')) # not sure what I was trying to do here... 
#files = np.append(files, files2)
files.sort()

#intp = senni.Interpolator(np.random.rand(100, 4), np.random.rand(100, 1024), np.random.rand(100, 1024), hlayer_size=8, no_pad=True)

#intp.load('models/bestmodel.pt')

cutoff = 60 # takes only the first ~60 time steps in each spectrum which represents the first 22.6 days 

for idx in range(files.shape[0]):
	params = get_params(files[idx]) # parse filenames into parameters
	f = open(files[idx], 'r')
	lines = f.readlines()
	if idx == 0:
		for line in lines:
			if line[0] == '#':
				try:
					times_spec = np.append(times_spec, float(line.split()[-1])) # obtaining time values for each spectrum
				except NameError:
					times_spec = np.array(float(line.split()[-1])).reshape(1)
				if len(times_spec) == cutoff: break
	f.close()

	spec_data = np.loadtxt(files[idx])
	spec_data = np.array(np.split(spec_data, spec_data.shape[0]/1024)) # splitting the spectra into a shape that corresponds to [simulation index, wavelength index, angular bin index (0-53) or angular bin uncertainty index (54-107)]
	wav_lower = spec_data[0, :, 0] # lower limits for each wavelength bin
	wav_upper = spec_data[0, :, 1] # upper limits for each wavelength bin
	spec = spec_data[:cutoff, :, 2:] # taking only the first [cutoff] time samples and ignoring the iteration numbers and time values (taken care of in code block above)
	if (spec.shape[0] != 60) or (spec.shape[1] != 1024) or (spec.shape[2] != 108):
		print('skipped ', params)
		continue

	try:
		params_all = np.append(params_all, params[None, :], axis=0) # storing all parameter values
		spec_all = np.append(spec_all, spec[None, :], axis=0) # storing all spectra values
	except NameError:
		params_all = params[None, :]
		spec_all = spec[None, :]

	print(idx, times_spec.shape[0], times_spec[-1], spec.shape)

#	if idx == 10: break

params_all = params_all[:, 2:].astype('float')

time_idx = 50 # trains the NN for *ONE* time point to reduce dimensionality for initial test, free this constraint when we get a good interpolation working

wavs = np.logspace(np.log10(wav_lower[0]), np.log10(wav_lower[-1]), num=1024) # wavelengths are binned into 1024 bins
spectrum = spec_all[:, time_idx, :, 0] # taking all the samples, at the specified time index, considering all wavelengths, taking the first angular bin (0-15.64 degrees)
spectrum[np.where(spectrum <= 0)] = np.min(spectrum[np.nonzero(spectrum)])/100 # remove negative/zero values for well behaved log
spectrum = np.log10(spectrum)

test_idx = 258 # removing one spectrum from the dataset to use as a test case
test_params = params_all[test_idx]
test_spectrum = spectrum[test_idx]
params_all = np.delete(params_all, test_idx, axis=0)
spectrum = np.delete(spectrum, test_idx, axis=0)
spec_err = np.ones_like(spectrum)*0.01 # need some uncertainty to use the chi2 likelihood function, assume low error for now, apply more realistic value later

print(params_all.shape, spectrum.shape)

intp_nn = senni.Interpolator(params_all, spectrum, spec_err, frac=0.1, test_frac=0.1, hlayer_size=8, p_drop=0.05,
                   epochs=1000, learning_rate=1e-1, betas=(0.9, 0.99), eps=1e-2, weight_decay=1e-6,
                   epochs_per_lr=200, lr_divisions=5, lr_frac=1./3., batch_size=128, shuffle=True, 
                   working_dir='.', loss_func='mape', no_pad=True) # set up NN interpolator, most of these will not be adjusted

intp_nn.train() # train NN

intp_nn.load('models/bestmodel.pt') # load best model achieved during training

for i in range(1000): # we want many evaluations to see how the effects of dropout will quantify uncertainty
	prediction = intp_nn.evaluate(test_params[None, :], eval_mode=False)
	try:
		predictions = np.append(predictions, prediction, axis=0)
	except NameError:
		predictions = prediction
	del prediction

print(predictions.shape)

intp_spectrum_nn = np.mean(predictions, axis=0) # take mean of 1000 evaluations to get mean spectrum
intp_spectrum_nn_err = np.std(predictions, axis=0) # std of 1000 evaluations to get uncertainty from dropout
#intp_spectrum_nn = intp_nn.evaluate(test_params[None, :])

intp_rf = slick.Interpolator() # train with random forest which typically has better convergence to verify fidelity of NN output

intp_rf.train(params_all, spectrum) # actual training with RF

intp_spectrum_rf = intp_rf.evaluate(test_params[None, :]) # prediction from RF

print(times_spec[time_idx])

np.savetxt('spectrum_intp_nn.dat', np.c_[wavs, test_spectrum, intp_spectrum_nn.flatten(), intp_spectrum_nn_err.flatten()]) # save NN outputs along with uncertainty
np.savetxt('spectrum_intp_rf.dat', np.c_[wavs, test_spectrum, intp_spectrum_rf.flatten()]) # RF outputs with no uncertainty, only used for NN fidelity

