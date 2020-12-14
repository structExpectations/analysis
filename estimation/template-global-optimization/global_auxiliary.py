import numpy as np


def wrapper_numpy(model_params, adapter_smm, cand_values):
    cand_df = model_params.copy()
    cand_df.loc[~cand_df["fixed"], "value"] = cand_values
    return adapter_smm.get_objective(cand_df)


def prepare_bounds(model_params):
    is_free = ~model_params["fixed"]
    bounds = np.tile(np.nan, (is_free.sum(), 2))
    bounds[:, 0] = model_params["lower"][is_free].values
    bounds[:, 1] = model_params["upper"][is_free].values
    return model_params, bounds
