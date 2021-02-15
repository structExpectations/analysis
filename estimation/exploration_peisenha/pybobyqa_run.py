#!usr/bin/env python
from functools import partial
import pickle as pkl

import pybobyqa as pybob
import pandas as pd
import numpy as np

from SimulationBasedEstimation import SimulationBasedEstimationCls

from pybobyqa_auxiliary import prepare_optimizer_interface
from pybobyqa_auxiliary import wrapper_numpy
from moments import get_moments

model_spec_fname = "resources/model_spec_init.yml"
model_para_fname = "start.soepy.pkl"

weighting_matrix = pkl.load(open("weighting_matrix.pkl", "rb"))
moments_obs = pkl.load(open("observed_moments.pkl", "rb"))
model_params = pd.read_pickle(model_para_fname)

# We extend the model parameters to also include the replacement rate as the last element.
model_params.loc["benefits_base", :] = [200, 1000, 100, True]
model_params.loc["delta", :] = [0.98, 0.99, 0.90, True]
model_params.loc["mu", :] = [-0.56, -0.99, -0.01, True]

# We need to add the information which parameters are fixed, which are free. We fix all
# parameters as the default and then free the relevant set.
model_params["fixed"] = True
model_params.loc[("disutil_work", "no_kids_f"), "fixed"] = False
model_params.loc[("disutil_work", "no_kids_p"), "fixed"] = False
model_params.loc[("disutil_work", "yes_kids_f"), "fixed"] = False
model_params.loc[("disutil_work", "yes_kids_p"), "fixed"] = False

model_params.loc[("const_wage_eq", "gamma_0s1"), "fixed"] = False
model_params.loc[("const_wage_eq", "gamma_0s2"), "fixed"] = False
model_params.loc[("const_wage_eq", "gamma_0s3"), "fixed"] = False

model_params.loc[("exp_returns", "gamma_1s1"), "fixed"] = False
model_params.loc[("exp_returns", "gamma_1s2"), "fixed"] = False
model_params.loc[("exp_returns", "gamma_1s3"), "fixed"] = False

model_params = model_params.astype({"fixed": bool})

# We set the tuning parameters of the optimizer so that it runs forever.
opt_kwargs = dict()
opt_kwargs["scaling_within_bounds"] = True
opt_kwargs["seek_global_minimum"] = True
opt_kwargs["objfun_has_noise"] = True
opt_kwargs["maxfun"] = 1

# We need to set up our criterion function.
adapter_kwargs = dict()
adapter_kwargs["model_spec_init_file_name"] = model_spec_fname
adapter_kwargs["weighting_matrix"] = weighting_matrix
adapter_kwargs["moments_obs"] = moments_obs
adapter_kwargs["get_moments"] = get_moments
adapter_kwargs["params"] = model_params

adapter_smm = SimulationBasedEstimationCls(**adapter_kwargs)

# Ready for the optimization.
x0, bounds = prepare_optimizer_interface(model_params)
p_wrapper_numpy = partial(wrapper_numpy, model_params, adapter_smm)
rslt = pybob.solve(p_wrapper_numpy, x0, bounds=bounds, **opt_kwargs)
np.testing.assert_almost_equal(rslt.f, 1367.110008640319)
