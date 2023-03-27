import spectra_interpolator as si
from sklearn.model_selection import RandomizedSearchCV as RSCV

def load_interpolator(path_to_sims, path_to_h5, path_to_model):
    intp = si.intp()
    intp.load_data(path_to_sims, path_to_h5, t_max=None, theta=0, short_wavs=False)
    intp.append_input_parameter(intp.times, 1)
    intp.preprocess()
    intp.load(path_to_model, fixed_angle=True)
    return intp

intp = load_interpolator('/lustre/scratch4/turquoise/mristic/knsc1_active_learning/*spec*', \
                         '/lustre/scratch4/turquoise/mristic/h5_data/TP_wind2_spectra.h5', \
                         '/lustre/scratch4/turquoise/mristic/rf_spec_intp_theta00deg.joblib')

#['bootstrap', 'ccp_alpha', 'class_weight', 'criterion', 'decision_path', 'estimator_params', 'estimators_', 'feature_importances_', 'fit', 'get_params', 'max_depth', 'max_features', 'max_leaf_nodes'    , 'max_samples', 'min_impurity_decrease', 'min_impurity_split', 'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 'n_estimators', 'n_features_', 'n_features_in_', 'n_jobs', 'n_outputs_', 'oob    _score', 'predict', 'random_state', 'score', 'set_params', 'verbose', 'warm_start']

rf = intp.intp.rfr

print('Loaded rf, optimizing parameters')

param_grid = {
        'n_estimators': [100, 250, 500, 750, 1000],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [500, 1000, 2000, 3000, 5000],
        'max_leaf_nodes': [250, 500, 750, 1000, 2000]
}

random_search = RSCV(rf, param_grid)
random_search.fit(intp.params, intp.spectra)

print(random_search.best_estimator_)
