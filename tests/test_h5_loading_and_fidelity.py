import h5py
import glob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

start = datetime.now()

files = np.array(glob.glob('/home/marko.ristic/lanl/knsc1_active_learning/*spec*'))
files.sort()

for idx in range(files.shape[0]):
	if files[idx].split('/')[-1].split('_')[7] == 'md0.011759': break

print(idx)

h5_load = h5py.File('h5_data/TP_wind2_spectra.h5', 'r')
data = h5_load['spectra'][:]
print(data.shape)
print(data.shape[0]/(1024*60))
print('finished loading h5')
#spec_all = np.array(np.split(data, 412))
spec_all = data.reshape(412, 60, 1024, 54)
print(spec_all.shape)

print((datetime.now()-start).total_seconds(), "seconds to load all data")

sim_spec = np.loadtxt('Run_TP_dyn_all_lanth_wind2_all_md0.011759_vd0.296779_mw0.067509_vw0.296739_spec_2020-07-23.dat')
sim_spec = np.array(np.split(sim_spec, sim_spec.shape[0]/1024))
print(sim_spec.shape)
sim_spec = sim_spec[50, :, 2]

plt.figure()
plt.plot(np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024), spec_all[idx, 50, :, 0])
plt.plot(np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024), sim_spec)
plt.xscale('log')
plt.yscale('log')
plt.savefig('test_h5_fidelity.pdf')

print(np.allclose(sim_spec, spec_all[idx, 50, :, 0]))
