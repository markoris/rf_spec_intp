import numpy as np
import matplotlib.pyplot as plt
import sys

def mass(m):
    return (m-1)/2 + 2

def vel(v):
    return (v-0.05)/0.25

mds = np.array([-1.14, -1.34, -1.08, -1.53, -1.65, -1.64, -1.70, -1.92, -1.79, -1.58])
md_err_upp = np.array([0.09, 0.20, 0.07, 0.02, 0.09, 0.10, 0.14, 0.13, 0.22, 0.20])
md_err_low = np.array([-0.15, -0.09, -0.39, -0.02, -0.07, -0.13, -0.11, -0.12, -0.16, -0.28])

vds = np.array([0.12, 0.14, 0.13, 0.13, 0.21, 0.30, 0.30, 0.15, 0.26, 0.25])
vd_err_upp = np.array([0.00, 0.01, 0.01, 0.01, 0.06, 0.00, 0.00, 0.10, 0.03, 0.05])
vd_err_low = np.array([-0.01, -0.01, -0.02, -0.02, -0.10, -0.14, -0.09, -0.07, -0.05, -0.06])

mws = np.array([-1.61, -1.99, -1.63, -1.58, -1.73, -1.79, -1.78, -1.58, -1.84, -2.07])
mw_err_upp = np.array([0.01, 0.06, 0.04, 0.01, 0.03, 0.03, 0.03, 0.05, 0.03, 0.12])
mw_err_low = np.array([-0.01, -0.08, -0.10, -0.01, -0.03, -0.04, -0.04, -0.08, -0.04, -0.16])

vws = np.array([0.06, 0.08, 0.16, 0.22, 0.07, 0.08, 0.05, 0.08, 0.24, 0.25])
vw_err_upp = np.array([0.02, 0.05, 0.07, 0.01, 0.01, 0.00, 0.03, 0.01, 0.01, 0.04])
vw_err_low = np.array([-0.01, -0.02, -0.00, -0.00, -0.02, -0.03, -0.00, -0.02, -0.02, -0.07])

#md_err_upp = mass(mds+md_err_upp)
#md_err_low = mass(mds+md_err_low)
#mds = mass(mds)

#vd_err_upp = vel(vds+vd_err_upp)
#vd_err_low = vel(vds+vd_err_low)
#vds = vel(vds)

#mw_err_upp = mass(mws+mw_err_upp)
#mw_err_low = mass(mws+mw_err_low)
#mws = mass(mws)

#vw_err_upp = vel(vws+vw_err_upp)
#vw_err_low = vel(vws+vw_err_low)
#vws = vel(vws)

times = np.linspace(1.4, 10.4, 10)

plt.rc('font', size=30)

fig = plt.figure(figsize=(19.2, 19.2))
gs = fig.add_gridspec(4, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
axs[0].scatter(times, mds, label='md')
axs[0].fill_between(times, mds+md_err_low, mds+md_err_upp, alpha=0.3, color='k')
axs[0].text(0.05, 0.1, r'$m_d$', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
axs[0].set_xticks(times)
axs[0].set_yticks([-1, -2, -3])
axs[0].set_ylim([-0.85, -3.15])
axs[0].invert_yaxis()
#axs[0].legend(loc='lower left')
axs[1].scatter(times, mws, label='mw')
axs[1].fill_between(times, mws+mw_err_low, mws+mw_err_upp, alpha=0.3, color='k')
axs[1].text(0.05, 0.1, r'$m_w$', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
axs[1].set_xticks(times)
axs[1].set_yticks([-1, -2, -3])
axs[1].set_ylim([-0.85, -3.15])
axs[1].invert_yaxis()
#axs[1].legend(loc='lower left')
axs[2].scatter(times, vds, label='vd')
axs[2].fill_between(times, vds+vd_err_low, vds+vd_err_upp, alpha=0.3, color='k')
axs[2].text(0.05, 0.1, r'$v_d$', horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
axs[2].set_xticks(times)
axs[2].set_yticks([0.05, 0.175, 0.3])
axs[2].set_ylim([0.02, 0.33])
#axs[2].legend(loc='upper left')
axs[3].scatter(times, vws, label='vw')
axs[3].fill_between(times, vws+vw_err_low, vws+vw_err_upp, alpha=0.3, color='k')
axs[3].text(0.05, 0.1, r'$v_w$', horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes)
axs[3].set_xticks(times)
axs[3].set_yticks([0.05, 0.175, 0.3])
axs[3].set_ylim([0.02, 0.33])
#axs[3].legend(loc='upper left')

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

#plt.scatter(times, mds, label='md')
#plt.fill_between(times, mds-md_err_low, mds+md_err_upp, alpha=0.3, color='k')
#
#plt.scatter(times, vds, label='vd+1')
#plt.fill_between(times, vd_err_low, vd_err_upp, alpha=0.3, color='k')
#
#plt.scatter(times, mws+2, label='mw+2')
#plt.fill_between(times, mw_err_low+2, mw_err_upp+2, alpha=0.3, color='k')
#
#plt.scatter(times, vws+3, label='vw+3')
#plt.fill_between(times, vw_err_low+3, vw_err_upp+3, alpha=0.3, color='k')
#
#plt.axhline(y=1, linestyle='--', color='gray')
#plt.axhline(y=2, linestyle='--', color='gray')
#plt.axhline(y=3, linestyle='--', color='gray')

plt.savefig('uncertainties.pdf')
