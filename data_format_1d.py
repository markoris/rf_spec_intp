import numpy as np

def format(inputs, targets, inputs_to_append, target_axis_to_append, check_subzero=False, log=False):

	'''
	Returns a 1-d array of targets corresponding to the N-d array of inputs
	'''

	for i in range(targets.shape[target_axis_to_append]):
		find_axis = [slice(None)]*(targets.ndim-1) # take all the values in the non-appended dimensions
		find_axis = np.insert(find_axis, target_axis_to_append, i) # take the index in the relevant dimension
		tgt = targets[tuple(find_axis)]
		try:
			new_targets = np.concatenate((new_targets, tgt), axis=0)
		except NameError:
			new_targets = tgt # this will put all N simulations for the 0th new input, then all N simulations for the 1st, etc...

	if check_subzero: new_targets[np.where(new_targets<=0)] = 1e-9
	if log: new_targets = np.log10(new_targets)

	long_inputs = np.tile(inputs, (inputs_to_append.shape[0], 1))
	long_inputs_to_append = np.repeat(inputs_to_append, inputs.shape[0], axis=0)

	new_inputs = np.hstack((long_inputs, long_inputs_to_append[:, None]))

	return new_inputs, new_targets
