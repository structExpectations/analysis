def wrapper_numpy(model_params, adapter_smm, cand_values):
    cand_df = model_params.copy()
    cand_df.loc[~cand_df["fixed"], "value"] = cand_values
    return adapter_smm.get_objective(cand_df)


def prepare_optimizer_interface(model_params):
    is_free = ~model_params["fixed"]

    x0 = model_params.loc[is_free, "value"].values

    lower = model_params.loc[is_free, "lower"].values
    upper = model_params.loc[is_free, "upper"].values
    bounds = (lower, upper)

    return x0, bounds
