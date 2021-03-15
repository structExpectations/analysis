import yaml

import pandas as pd
import numpy as np

import soepy

HUGE_FLOAT = 100000000000.0
HUGE_INT = 100000000000


class SimulationBasedEstimationCls:
    """This class facilitates estimation of the free parameter vector in a life-cycle model of
    labor supply based on the soepy package and smm_estimagic."""

    def __init__(
        self,
        params,
        model_spec_init_file_name,
        moments_obs,
        weighting_matrix,
        get_moments,
        max_evals=HUGE_INT,
    ):

        self.model_spec_init_file_name = model_spec_init_file_name
        self.weighting_matrix = weighting_matrix
        self.moments_obs = moments_obs
        self.get_moments = get_moments
        self.max_evals = max_evals
        self.fval = HUGE_FLOAT
        self.df_steps = None
        self.params = params
        self.num_evals = 0

        self.fval = self.get_objective(params)

    def get_objective(self, params_cand):

        self.params = params_cand

        fval = self._calculate_criterion_func_value(params_cand)

        if fval < self.fval:
            self._logging_smm(params_cand, fval)
            self.fval = fval

        if self.num_evals >= self.max_evals:
            raise RuntimeError("maximum number of evaluations reached")

        return fval

    def _calculate_criterion_func_value(self, params_cand):

        self.params = params_cand

        # Extract elements from configuration file.
        benefits_base = float(params_cand.loc["benefits_base", "value"].values[0])
        delta = float(params_cand.loc["delta", "value"].values[0])
        mu = float(params_cand.loc["mu", "value"].values[0])

        model_spec_init_dict = yaml.load(
            open(self.model_spec_init_file_name), Loader=yaml.Loader
        )
        model_spec_init_dict["TAXES_TRANSFERS"]["benefits_base"] = benefits_base
        model_spec_init_dict["CONSTANTS"]["delta"] = delta
        model_spec_init_dict["CONSTANTS"]["mu"] = mu

        fname_modified = "resources/model_spec_init.modified.yml"
        yaml.dump(model_spec_init_dict, open(fname_modified, "w"))

        data_frame_sim = df_alignment(soepy.simulate(params_cand, fname_modified))
        moments_sim = self.get_moments(data_frame_sim)

        stats_dif = np.array(self.moments_obs) - np.array(moments_sim)
        fval = float(np.dot(np.dot(stats_dif, self.weighting_matrix), stats_dif))

        return fval

    def _logging_smm(self, params_cand, fval):
        """This method contains logging capabilities
        that are just relevant for the SMM routine."""
        df_step = pd.concat([params_cand], keys=[self.num_evals], names=["Step"])
        df_step.loc[(self.num_evals, "fval", "fval"), "value"] = fval
        self.df_steps = pd.concat([self.df_steps, df_step])
        self.df_steps.to_pickle("steps.respy.pkl")


def df_alignment(df):
    df_int = df.copy()
    rename = dict()
    rename["Choice"] = {0: "Home", 1: "Part", 2: "Full"}
    rename["Education_Level"] = {0: "Low", 1: "Medium", 2: "High"}

    df_int.replace(rename, inplace=True)

    df_int.set_index(["Identifier", "Period"], inplace=True)

    return df_int
