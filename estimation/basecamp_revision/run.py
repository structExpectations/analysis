#!usr/bin/env python
import pickle as pkl

import pandas as pd
import numpy as np

from SimulationBasedEstimation import SimulationBasedEstimationCls
from moments import get_moments

weighting_matrix = pkl.load(open("weighting_matrix.pkl", "rb"))
moments_obs = pkl.load(open("observed_moments.pkl", "rb"))
model_params = pd.read_pickle("start.soepy.pkl")

# We need to set up our criterion function.
adapter_kwargs = dict()
adapter_kwargs["model_spec_init_file_name"] = "resources/model_spec_init.yml"
adapter_kwargs["weighting_matrix"] = weighting_matrix
adapter_kwargs["moments_obs"] = moments_obs
adapter_kwargs["get_moments"] = get_moments
adapter_kwargs["params"] = model_params

adapter_smm = SimulationBasedEstimationCls(**adapter_kwargs)
np.testing.assert_almost_equal(adapter_smm.fval, 551.830950963115)
