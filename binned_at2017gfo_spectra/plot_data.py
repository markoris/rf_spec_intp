import numpy as np
import matplotlib.pyplot as plt
import glob

plt.rc('font', size=30)
plt.rc('lines', lw=2)

data = glob.glob('AT*')
times = np.linspace(1.4, 10.4, 10)

for obs in range(len(data)):
    plt.figure(figsize=(19.2, 10.8))
    dat = np.loadtxt(data[obs])
    dat = dat[np.where(dat[:, 1] > 0)[0]]
    plt.plot(dat[:, 0]*1e-4, dat[:, 1]) # factor of 1e-4 on wavelengths to convert to microns
    plt.fill_between(dat[:, 0]*1e-4, dat[:, 1]-dat[:, 2], dat[:, 1]+dat[:, 2], alpha=0.3)
    plt.xscale('log')
    plt.savefig('t_%g.pdf' % times[obs])
    plt.close()
