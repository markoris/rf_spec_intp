import corner
import glob, os
import numpy as np
import matplotlib.pyplot as plt

times = [1.43, 2.42, 3.41, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4]

for t in times:
    all_samples = glob.glob('rift*t%g.dat' % t)
    all_samples.sort(key=os.path.getmtime)
    all_samples = all_samples[:50] # first set of 5x10 runs
    for i in range(len(all_samples)):
        print(i)
        samples = np.loadtxt(all_samples[i])
        try:
            total_samples = np.concatenate((total_samples, samples), axis=0)
        except NameError:
            total_samples = samples
    lnL = total_samples[:, 4]
    p = total_samples[:, 5]
    p_s = total_samples[:, 6]
    weights = np.exp(lnL)*p/p_s
    weights /= np.sum(weights)
    total_samples[:, 0] = np.log10(total_samples[:, 0])
    total_samples[:, 2] = np.log10(total_samples[:, 2])
    corner.corner(total_samples[:, :4],
    weights=weights, 
    labels=[
        r"$\log M_d$",
        r"$V_d$",
        r"$\log M_w$",
        r"$V_w$",
    ],
    quantiles=[0.16, 0.84],
    show_titles=True,
    plot_datapoints=False,
    plot_density=False,
    contours=True,
    smooth1d=0.1,
    smooth=0.1,
    title_kwargs={"fontsize": 15},
)
    plt.savefig('t_%gd_posteriors.pdf' % t)
    plt.close()
    del all_samples, total_samples, weights
