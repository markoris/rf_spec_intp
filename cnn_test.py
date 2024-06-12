import numpy as np
import h5py, glob, torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
	# in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1
	# in_channels = dimensionality at axis=1
	# out_channels = output dimensionality at axis=1
	# kernel_size = size of convolutional kernel (larger = wider kernel = larger covariance neighborhood)
	# stride = how many "pixels"/bins the kernel moves per evaluation
	# padding = what values to include at the edges of the data (i.e. to preserve dimensionality)
	# dilation = how much space is between kernel points
	# 	dilation=1 means that a kernel of size 3 is size 3x3, with each pixel convolved
	# 	dilation=1: [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
	#	dilation=2 means that a kernel of size 3 is size 5x5, with every other pixel convolved
	#	dilation=2: [[1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1]] 
	# kernel, padding, etc. can all be 2-D and should probably be different!
	# that is to say, a convolution across 5 wavelength bins is 5/1024 ~ 0.5%, but across 5 time bins is 5/60 ~ 8%
	# https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md  < great animations!

        self.conv1 = nn.Conv2d(54, 108, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(108, 216, 5)
        self.fc1 = nn.Linear(216 * 12 * 253, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data(path_to_sims_dir, path_to_spectra_h5):

	t_a = 0.125 # spectral start time
	t_b = 20.749 # spectral end time 

	times = np.logspace(np.log10(t_a), np.log10(t_b), 60) # this should cover the full spectral possibilities range
	angles = np.degrees([np.arccos((1 - (i-1)*2/54.)) for i in np.arange(1, 55, 1)])
	wavs_full = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # in microns (from 1e4 scaling factor)

	files = np.array(glob.glob(path_to_sims_dir))
	files.sort()

	h5_load = h5py.File(path_to_spectra_h5, 'r')
	data = h5_load['spectra'][:]
	spec_all = data.reshape(files.shape[0], times.shape[0], wavs_full.shape[0], angles.shape[0])

	# move the angles axis to coincide with "image feature" location for convolutions
	spec_all = np.moveaxis(spec_all, -1, 1)

	img_vol = 256**2 * 3
	spec_vol = times.shape[0]*wavs_full.shape[0]*angles.shape[0]

	print('Typical data volume for a single 256x256 RGB image: ', img_vol)
	print('Spectra data volume assuming all times/angles/wavs: ', spec_vol)
	print('Ratio of spectra volume to image volume: ', spec_vol/img_vol)

	return spec_all

if __name__== "__main__":
	
	# Load sims to get parameter values + h5 to get spectra
	path_to_sims_dir = '/home/marko.ristic/knsc1_active_learning/*spec*'
	path_to_spectra_h5 = '/home/marko.ristic/h5_data/TP_wind2_spectra.h5'
	spec = load_data(path_to_sims_dir, path_to_spectra_h5)
	
	# Take log of data, whiten later
	spec[np.where(spec <= 0)] = np.min(spec[np.nonzero(spec)])/10
	spec = np.log10(spec) 
	
	# Torch is picky, work in float32 for less memory impact
	spec = spec.astype(np.float32)

	# Using torch.from_numpy does NOT create a copy (array WILL be modified)
	# This means verification needs to reload spectra, but saves LOTS of memory
	spec = torch.from_numpy(spec)

	net = Net()
	print(net(spec))

