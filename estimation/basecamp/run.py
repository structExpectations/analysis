import numpy as np

from estimagic.optimization.optimize import minimize

from auxiliary import prepare_estimation
from auxiliary import get_moments
from SimulationBasedEstimation import SimulationBasedEstimationCls


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
        -0.400,
        -0.800,
        0.05,
        0.10,
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
        -0.050,
        -0.150,
        0.400,
        0.500,
        0.800,
        0.800,
        0.800,
    ),
    1,
)

model_params_init_file_name = "resources/toy_model_init_file_03_3types.pkl"
model_spec_init_file_name = "resources/model_spec_init.yml"
data_file_name = "resources/data_obs_3types_9000.pkl"
log_file_name_extension = "test"

# TODO: We need the weighing matrix computed outside this function. We do not need to change it
#  during estimations. The same is with the observed moments. We need all things related to the
#  observed data outside of this repository. I created a data repo, please move data and creation
#  of observed moments and weighting matrix there.
moments_obs, weighting_matrix, model_params_df = prepare_estimation(
     model_params_init_file_name, model_spec_init_file_name, data_file_name, lower, upper
)

max_evals = 1

adapter_smm = SimulationBasedEstimationCls(
     params=model_params_df,
     model_spec_init_file_name=model_spec_init_file_name,
     moments_obs=moments_obs,
     weighting_matrix=weighting_matrix,
     get_moments=get_moments,
     log_file_name_extension=log_file_name_extension,
     max_evals=max_evals,
 )

algo_options = {"stopeval": 1e-9}


result = minimize(
    criterion=adapter_smm.get_objective,
    params=adapter_smm.params,
    algorithm="nlopt_bobyqa",
    algo_options=algo_options,
)

# TODO: We need the ability to create an estimation report based on a simulated sample at the
#  best parameter vectors. See as an example,
#  https://github.com/OptionValueHumanCapital/analysis/blob/master/python/create_report.py
