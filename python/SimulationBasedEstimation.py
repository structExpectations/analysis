import os

import pandas as pd
import numpy as np

import soepy

HUGE_INT = 100000000000


class SimulationBasedEstimationCls:
    """This class facilitates estimation of the free parameter vector
    in a life-cycle model of labor supply based on the soepy package
    and smm_estimagic."""

    def __init__(
        self,
        params,
        model_spec_init_file_name,
        moments_obs,
        weighting_matrix,
        get_moments,
        log_file_name_extension,
        max_evals=HUGE_INT,
    ):

        self.params = params
        self.model_spec_init_file_name = model_spec_init_file_name
        self.moments_obs = moments_obs
        self.weighting_matrix = weighting_matrix
        self.get_moments = get_moments
        self.max_evals = max_evals
        self.log_file_name_extension = log_file_name_extension

        self.num_evals = 0
        self.fval = None

        self._calculate_criterion_func_value(self.params)

    def get_objective(self, params_cand):

        self.params = params_cand
        self.params.drop(columns=["_fixed"], inplace=True, errors="ignore")

        # Obtain criterion function value
        fval, stats_obs, stats_sim = self._calculate_criterion_func_value(params_cand)

        # print(params_cand)
        print(fval)

        # Save params and function value as pickle object.
        is_start = self.fval is None

        if is_start:
            data = {"current": fval, "start": fval, "step": fval}
            self.fval = pd.DataFrame(
                data, columns=["current", "start", "step"], index=[0]
            )
            self.params.to_pickle("logging/step.soepy.pkl")
        else:
            is_step = self.fval["step"].iloc[-1] > fval
            step = self.fval["step"].iloc[-1]
            start = self.fval["start"].loc[0]

            if is_step:
                data = {"current": fval, "start": start, "step": fval}
                self.params.to_pickle("logging/step.soepy.pkl")
            else:
                data = {"current": fval, "start": start, "step": step}

            self.fval = self.fval.append(data, ignore_index=True)

        self._logging_smm(stats_obs, stats_sim, fval)

        self.num_evals = self.num_evals + 1
        if self.num_evals >= self.max_evals:
            raise RuntimeError("maximum number of evaluations reached")

        return fval

    def _calculate_criterion_func_value(self, params_cand):

        self.params = params_cand

        # Generate simulated data set
        data_frame_sim = soepy.simulate(self.params, self.model_spec_init_file_name)

        # Calculate simulated moments
        moments_sim = self.get_moments(data_frame_sim)

        # Move all moments from a dictionary to an array
        stats_obs, stats_sim = [], []

        for group in ["Wage_Distribution", "Choice_Probability"]:
            for key_ in self.moments_obs[group].keys():
                stats_obs.extend(self.moments_obs[group][key_])
                stats_sim.extend(moments_sim[group][key_])

        # Construct criterion value
        stats_dif = np.array(stats_obs) - np.array(stats_sim)

        fval = float(np.dot(np.dot(stats_dif, self.weighting_matrix), stats_dif))

        return fval, stats_obs, stats_sim

    def _logging_smm(self, stats_obs, stats_sim, fval):
        """This method contains logging capabilities that are just relevant for the SMM routine."""

        # Save log files in a seperate directory
        if not os.path.exists("logging/"):
            os.makedirs("logging/")

        fname = "logging/monitoring.smm_estimagic." + self.log_file_name_extension + ".info"
        fname2 = (
            "logging/monitoring_compact.smm_estimagic."
            + self.log_file_name_extension
            + ".info"
        )

        if self.num_evals == 1 and os.path.exists(fname):
            os.unlink(fname)
        if self.num_evals == 1 and os.path.exists(fname2):
            os.unlink(fname2)

        with open(fname, "a+") as outfile:
            fmt_ = "\n\n{:>8}{:>15}\n\n"
            outfile.write(fmt_.format("EVALUATION", self.num_evals))
            fmt_ = "\n\n{:>8}{:>15}\n\n"
            outfile.write(fmt_.format("fval", round(fval, 5)))
            for x in self.params.index:
                info = [x[0], x[1], round(self.params.loc[x, "value"], 5)]
                fmt_ = "{:>8}" + "{:>15}" * 2 + "\n\n"
                outfile.write(fmt_.format(*info))

            fmt_ = "{:>8}" + "{:>15}" * 4 + "\n\n"
            info = ["Moment", "Observed", "Simulated", "Difference", "Weight"]
            outfile.write(fmt_.format(*info))
            for x in enumerate(stats_obs):
                stat_obs, stat_sim = stats_obs[x[0]], stats_sim[x[0]]
                info = [
                    x[0],
                    stat_obs,
                    stat_sim,
                    abs(stat_obs - stat_sim),
                    self.weighting_matrix[x[0], x[0]],
                ]

                fmt_ = "{:>8}" + "{:15.5f}" * 4 + "\n"
                outfile.write(fmt_.format(*info))

        with open(fname2, "a+") as outfile:
            fmt_ = "\n\n{:>8}{:>15}\n\n"
            outfile.write(fmt_.format("EVALUATION", self.num_evals))
            fmt_ = "\n\n{:>8}{:>15}\n\n"
            outfile.write(fmt_.format("fval", round(fval, 5)))
            for x in self.params.index:
                info = [x[0], x[1], round(self.params.loc[x, "value"], 5)]
                fmt_ = "{:>8}" + "{:>15}" * 2 + "\n\n"
                outfile.write(fmt_.format(*info))
