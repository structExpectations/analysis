#!usr/bin/env python
import pickle

import pandas as pd

from estimagic.optimization.optimize import minimize
from estimation.basecamp.moments_calculation import get_moments
from python.SimulationBasedEstimation import SimulationBasedEstimationCls
from configurations.analysis_soepy_config import LOGGING_DIR

model_params_init_file_name = "resources/model_params.pkl"
model_spec_init_file_name = "resources/model_spec_init.yml"


model_params_df = pd.read_pickle(model_params_init_file_name)

with open("resources/moments_obs.pkl", "rb") as f:
    moments_obs = pickle.load(f)

with open("resources/weighting_matrix.pkl", "rb") as f:
    weighting_matrix = pickle.load(f)

constraints = [
    {"loc": "exp_accm", "type": "fixed"},
    {"loc": "exp_deprec", "type": "fixed"},
    {"loc": "hetrg_unobs", "type": "fixed"},
    {"loc": "shares", "type": "fixed"},
]

adapter_smm = SimulationBasedEstimationCls(
    params=model_params_df,
    model_spec_init_file_name=model_spec_init_file_name,
    moments_obs=moments_obs,
    weighting_matrix=weighting_matrix,
    get_moments=get_moments,
    logging_dir=str(LOGGING_DIR),
)

algo_options = {"max_iterations": 10000}


result = minimize(
    criterion=adapter_smm.get_objective,
    params=adapter_smm.params,
    algorithm="scipy_powell",
    algo_options=algo_options,
    constraints=constraints,
)

with open("logging/result.pkl", "wb") as f:
    pickle.dump(result, f)

# np.testing.assert_almost_equal(
#     4642.07887028819, result[0]["fitness"], 4, "Criterion value mismatch"
# )
