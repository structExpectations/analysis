#!usr/bin/env python
import pickle

import pandas as pd
import numpy as np

from estimagic.optimization.optimize import minimize
from estimation.basecamp.moments_calculation import get_moments
from python.SimulationBasedEstimation import SimulationBasedEstimationCls

model_params_init_file_name = "resources/toy_model_init_file_03_2types.pkl"
model_spec_init_file_name = "resources/model_spec_init.yml"
log_file_name_extension = "test"


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

algo_options = {"stopeval": 1e-14, "maxeval": 1}


result = minimize(
    criterion=adapter_smm.get_objective,
    params=adapter_smm.params,
    algorithm="nlopt_bobyqa",
    algo_options=algo_options,
)

np.testing.assert_almost_equal(
    11356.971245152485, result[0]["fitness"], 6, "Criterion value mismatch"
)
