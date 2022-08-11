import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=40)
plt.rc('lines', lw=3)

#nn = np.loadtxt('spectrum_intp_nn.dat')
rf = np.loadtxt('spectrum_intp_rf.dat')

wavs = rf[:, 0]
test_spectrum = rf[:, 1]
#intp_spectrum_nn = nn[:, 2]
#intp_spectrum_nn_err = nn[:, 3]
intp_spectrum_rf = rf[:, 2]

#plt.figure(figsize=(19.2, 10.8))
#plt.plot(wavs, test_spectrum, 'k')
#plt.plot(wavs, intp_spectrum_nn, 'r')
#plt.fill_between(wavs, intp_spectrum_nn-intp_spectrum_nn_err, intp_spectrum_nn+intp_spectrum_nn_err, alpha=0.3, color='red')
#plt.xscale('log')
#plt.xlim(wavs[0], wavs[-1])
#plt.title('neural network interpolation')
#plt.savefig('sample_spectrum_nn.png') 
#plt.close()

plt.figure(figsize=(19.2, 10.8))
plt.plot(wavs, test_spectrum, 'k')
plt.plot(wavs, intp_spectrum_rf.flatten(), 'r')
plt.xscale('log')
plt.xlim(wavs[0], wavs[-1])
plt.title('random forest interpolation')
plt.savefig('sample_spectrum_rf.pdf') 
