#!usr/bin/env python
from functools import partial
import pickle

from scipy import optimize as opt
import pandas as pd

from SimulationBasedEstimation import SimulationBasedEstimationCls

from global_auxiliary import prepare_bounds
from global_auxiliary import wrapper_numpy
from global_moments import get_moments

model_spec_fname = "resources/model_spec_init.yml"
model_para_fname = "resources/model_params.pkl"

weighting_matrix = pickle.load(open("resources/weighting_matrix_ones.pkl", "rb"))
moments_obs = pickle.load(open("resources/moments_obs.pkl", "rb"))
model_params = pd.read_pickle(model_para_fname)

# We need to add the information which parameters are fixed, which are free. We fix all
# parameters as the default and then free the relevant set.
model_params["fixed"] = True
model_params.loc[("const_wage_eq", "gamma_0s1"), "fixed"] = False

# We set the tuning parameters of the optimizer so that it runs forever.
opt_kwargs = dict()
opt_kwargs["maxiter"] = 1000000
opt_kwargs["tol"] = 0.0

# We need to set up our criterion function.
adapter_kwargs = dict()
adapter_kwargs["model_spec_init_file_name"] = model_spec_fname
adapter_kwargs["weighting_matrix"] = weighting_matrix
adapter_kwargs["moments_obs"] = moments_obs
adapter_kwargs["get_moments"] = get_moments

adapter_kwargs["params"] = model_params

adapter_smm = SimulationBasedEstimationCls(**adapter_kwargs)

# Ready for the optimization.
model_params, bounds = prepare_bounds(model_params)
p_wrapper_numpy = partial(wrapper_numpy, model_params, adapter_smm)
opt.differential_evolution(p_wrapper_numpy, bounds, **opt_kwargs)
