def third_component(M, v, kappa):
	import glob
	import numpy as np
	from matplotlib import ticker
	import matplotlib.patheffects as PathEffects
	import matplotlib.pyplot as plt

	# Tunable parameters describing the low-opacity (Ye ~ 0.5) third component
	M = 0.01 				# Solar masses
	v = 0.1 				# Speed of light
	kappa = 0.5  				# cm^2 / g
	v_array = np.linspace(0.1, 0.3, 21)	# Speed of light
		
	print('Mass = ', M)
	print('Velocity = ', v)
	print('Opacity = ', kappa)
	print('Velocity shells = \n', v_array)

	c = 3e10				# cm / s
	v_array *= c				# cm / s

	# wavelengths below binned as in SuperNu, reported in cm
	wavs = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)

	# times below in days from AT2017gfo observations
	times = np.array([1.43, 2.42, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4])

	M *= 1.989e33 	# g
	v *= c 		# cm / s
	times *= 86400  # seconds

	# Equation 10
	def M_v(v_array, M, v_0):
		return M*np.power((v_array/v_0), -3)

	def E_v(dM_v, v_array):
		# assume E_v = 0.5M_v*v^2 for the mass, velocity of each shell
		return 0.5*dM_v*v_array**2

	# text under Equation 18
	def L_tot(dE_v, dM_v, kappa, M, v_0, v_array, times):
		# L_v = dE_v / (t_d_v + t_lc_v)
		# L_v = dE_v / (M_v^(4/3) * kappa / (4 * pi * M^1/3 * v_0 * t * c) + (v * t / c)) for t in times
		# L_tot = sum_v dM_v * L_v
		
		return np.array([np.sum(dM_v*(dE_v/((dM_v**(4/3)*kappa/(4*np.pi*M**(1/3)*v_0*time*c) + v_array*time/c)))) for time in times])

	def T_eff(L_tot, R_ph):
		sigma = 5.6704e-5 # erg cm^-2 s^-1 K^-4
		return np.power((L_tot/(4*np.pi*sigma*R_ph**2)), 1/4)

	def planck(wav, T_eff, R_ph):
		# wavs must be supplied in cm!!!
		h = 6.626e-27 	# cm^2 g / s
		k = 1.381e-16 	# cm^2 g s^-2 K^-1
		D = 40 		# Mpc
		D *= 3.086e24 	# cm
		return 2*h*c**2/(wav**5)*(1/(np.exp(h*c/(wav*k*T_eff))-1))*R_ph**2/(D)**2

	dM_v = M_v(v_array, M, v)
	dE_v = E_v(dM_v, v_array)

	R_ph = kappa/(4*np.pi*times**2)*np.sum(dM_v/v_array**2)
	print(R_ph.shape)
	L_total = L_tot(dE_v, dM_v, kappa, M, v, v_array, times)
	print(L_total.shape)
	T_eff_t = T_eff(L_total, R_ph)
	print(T_eff_t.shape)
	F_lambda_t = np.array([planck(wavs, T_eff_t[i], R_ph[i]) for i in range(len(R_ph))])
	print(F_lambda_t.shape)

	plt.rc('font', size=40)
	plt.rc('lines', lw=3)

	times /= 86400 # for plot naming purposes

	for i in range(F_lambda_t.shape[0]):
		plt.figure(figsize=(19.2, 10.8))
		plt.plot(wavs*1e3, F_lambda_t[i, :], c='k')
		plt.xlabel(r'$\lambda$ ($\mu$m)')
		plt.ylabel(r'$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\mu$m^${-1}$)')
		filters = np.array(glob.glob('filters/*'))
		wavelengths = 'grizyJHKS'
		colors = {"g": "blue", "r": "cyan", "i": "lime", "z": "green", "y": "greenyellow", "J": "gold",
			 "H": "orange", "K": "red", "S": "darkred"}
		text_locs = {"g": 0.2, "r": 0.35, "i": 0.44, "z": 0.51, "y": 0.57, "J": 0.65, "H": 0.78, "K": 0.91}
		for fltr in range(len(filters)):
			filter_wavs = np.loadtxt(filters[fltr])
			filter_wavs = filter_wavs[:, 0]*1e-4 # factor of 1e-4 to go from Angstroms to microns
			wav_low, wav_upp = filter_wavs[0], filter_wavs[-1]
			fltr_band = filters[fltr].split('/')[-1][0]
			if fltr_band == "S": continue
			text_loc = text_locs[fltr_band]
			fltr_indx = wavelengths.find(fltr_band)
			plt.axvspan(wav_low, wav_upp, alpha=0.5, color=colors[wavelengths[fltr_indx]])
			plt.text(text_loc, 1.015, fltr_band, color=colors[wavelengths[fltr_indx]], transform=plt.gca().transAxes, path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="black")])
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig('third_component_spectra/t_%g.pdf' % times[i])	

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Spectrum from third component to boost blue emission.")
	parser.add_argument('-M', default=0.01, type=float, help='Ejecta mass in solar masses.')
	parser.add_argument('-v', default=0.1, type=float, help='Ejecta velocity as a fraction of the speed of light.')
	parser.add_argument('-kappa', default=0.5, type=float, help='Ejecta grey opacity in units of cm^2 g^-1.') 

	args = parser.parse_args()

	third_component(args.M, args.v, args.kappa)
