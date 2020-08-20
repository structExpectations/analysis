#!usr/bin/env python
import pickle

import pandas as pd
import numpy as np

from estimagic.optimization.optimize import minimize
from estimation.basecamp.moments_calculation import get_moments
from python.SimulationBasedEstimation import SimulationBasedEstimationCls

model_params_init_file_name = "resources/init_values_06_07_3types.pkl"
model_spec_init_file_name = "resources/model_spec_init_08.yml"
log_file_name_extension = "run_6"


model_params_df = pd.read_pickle(model_params_init_file_name)

with open("resources/moments_obs.pkl", "rb") as f:
    moments_obs = pickle.load(f)

with open("resources/weighting_matrix.pkl", "rb") as f:
    weighting_matrix = pickle.load(f)

adapter_smm = SimulationBasedEstimationCls(
    params=model_params_df,
    model_spec_init_file_name=model_spec_init_file_name,
    moments_obs=moments_obs,
    weighting_matrix=weighting_matrix,
    get_moments=get_moments,
    log_file_name_extension=log_file_name_extension,
)

algo_options = {"stopeval": 1e-14, "maxeval": 2}


result = minimize(
    criterion=adapter_smm.get_objective,
    params=adapter_smm.params,
    algorithm="nlopt_bobyqa",
    algo_options=algo_options,
)

np.testing.assert_almost_equal(
    16042.085765533773, result[0]["fitness"], 6, "Criterion value mismatch"
)
