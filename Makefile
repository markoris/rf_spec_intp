define PREFIX
#!/bin/bash
#SBATCH --job-name spec_intp
#SBATCH --account=w22_knspectra
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --mem=0	

module load python

endef

export PREFIX

prefix:
	@echo "$$PREFIX" > runjob.sh

train_rf: prefix
	@echo "python -u run_spectra_interpolator.py" >> runjob.sh
	@sbatch runjob.sh

least_squares_single_obs: prefix
	@echo "python -u least_squares_single_obs.py" >> runjob.sh
	@sbatch runjob.sh

hyperparameters: prefix
	@echo "python -u rf_hyperparameters.py" >> runjob.sh
	@sbatch runjob.sh

parameter_uncertainty: prefix
	@echo "python -u parameter_uncertainty.py" >> runjob.sh
	@sbatch runjob.sh

least_squares_single_obs_sims: prefix
	@echo "python -u least_squares_single_obs_sims.py" >> runjob.sh
	@sbatch runjob.sh

least_squares_all_times: prefix
	@echo "pythoni -u least_squares.py" >> runjob.sh
	@sbatch runjob.sh

sim_obs_overlap: prefix
	@echo "python -u plot_sims_obs.py" >> runjob.sh
	@sbatch runjob.sh
