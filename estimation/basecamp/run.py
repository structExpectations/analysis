import pickle

import numpy as np

from estimagic.optimization.optimize import minimize
from smm_estimagic.auxiliary import prepare_estimation
from smm_estimagic.auxiliary import get_moments
from smm_estimagic.SimulationBasedEstimation import SimulationBasedEstimationCls


lower = np.tile(
    (
        1.000,
        1.000,
        1.000,
        0.050,
        0.050,
        0.050,
        0.005,
        0.005,
        0.005,
        0.001,
        0.001,
        0.001,
        1.00,
        1.00,
        -0.400,
        -0.800,
        0.05,
        0.001,
        0.001,
        0.001,
    ),
    1,
)

upper = np.tile(
    (
        3.000,
        3.000,
        3.000,
        0.400,
        0.400,
        0.400,
        0.600,
        0.600,
        0.600,
        0.150,
        0.150,
        0.150,
        4.00,
        4.00,
        -0.050,
        -0.150,
        0.500,
        0.800,
        0.800,
        0.800,
    ),
    1,
)

model_params_init_file_name = "init_files/toy_model_init_file_03_2types.pkl"
model_spec_init_file_name = "init_files/model_spec_init.yml"
log_file_name_extension = "data_v2"


model_params_df = prepare_estimation(
    model_params_init_file_name, lower, upper
)

with open('init_files/moments_obs.pkl', 'rb') as f:
    moments_obs = pickle.load(f)

with open('init_files/weighting_matrix.pkl', 'rb') as f:
    weighting_matrix = pickle.load(f)

max_evals = 2000

adapter_smm = SimulationBasedEstimationCls(
    params=model_params_df,
    model_spec_init_file_name=model_spec_init_file_name,
    moments_obs=moments_obs,
    weighting_matrix=weighting_matrix,
    get_moments=get_moments,
    log_file_name_extension=log_file_name_extension,
    max_evals=max_evals,
)

algo_options = {"stopeval": 1e-14}


result = minimize(
    criterion=adapter_smm.get_objective,
    params=adapter_smm.params,
    algorithm="nlopt_bobyqa",
    algo_options=algo_options,
)
