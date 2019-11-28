import pandas as pd
import numpy as np

# TODO: There are sevaral unused arguments and violations of flake8. Please set up automatic test
#  and configuration.

def get_moments(data):
    # Pre_process data frame

    # Determine the education level given years of experience
    data["Educ_Level"] = 0
    data.loc[data["Years_of_Education"] == 11, "Educ_Level"] = 1
    data.loc[data["Years_of_Education"] == 12, "Educ_Level"] = 2

    # Determine the observed wage given period choice
    data["Wage_Observed"] = 0
    data.loc[data["Choice"] == 0, "Wage_Observed"] = data.loc[
        data["Choice"] == 0, "Period_Wage_N"
    ]
    data.loc[data["Choice"] == 1, "Wage_Observed"] = data.loc[
        data["Choice"] == 1, "Period_Wage_P"
    ]
    data.loc[data["Choice"] == 2, "Wage_Observed"] = data.loc[
        data["Choice"] == 2, "Period_Wage_F"
    ]

    # Calculate moments

    # Initialize moments dictionary
    moments = dict()

    # Store moments in groups as nested dictionary
    for group in ["Wage_Distribution", "Choice_Probability"]:
        moments[group] = dict()

    # Compute unconditional moments of the wage distribution
    info = data.groupby(["Period"])["Wage_Observed"].describe().to_dict()

    # Save mean and standard deviation of wages for each period
    # to Wage Distribution section of the moments dictionary
    for period in range(40):  ## TODO: Remove hard coded number
        moments["Wage_Distribution"][period] = []
        try:
            for label in ["mean", "std"]:
                moments["Wage_Distribution"][period].append(info[label][period])
        except KeyError:
            for i in range(2):
                moments["Wage_Distribution"][period].append(0.0)

    # Compute unconditional moments of the choice probabilities
    info = data.groupby(["Period"])["Choice"].value_counts(normalize=True).to_dict()

    for period in range(40):  ## TODO: Remove hard coded number
        moments["Choice_Probability"][period] = []
        for choice in range(3):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments["Choice_Probability"][period].append(stat)

    return moments


def get_weighting_matrix(data, num_agents_smm, num_samples):
    """Calculates the weighting matrix based on the
    moments of the observed data"""

    moments_sample = []

    # Collect n samples of moments
    for k in range(num_samples):
        data_sample = data.sample(n=num_agents_smm)

        moments_sample_k = get_moments(data_sample)

        moments_sample.append(moments_sample_k)

        k = +1

    # Append samples to a list of size num_samples
    # containing number of moments values each
    stats = []

    for moments_sample_k in moments_sample:
        stats.append(moments_dict_to_list(moments_sample_k))

    # Calculate sample variances for each moment
    moments_var = np.array(stats).var(axis=0)

    # Handling of nan
    moments_var[np.isnan(moments_var)] = np.nanmax(moments_var)

    # Handling of zero variances
    is_zero = moments_var <= 1e-10
    moments_var[is_zero] = 0.1

    # Construct weighting matrix
    weighting_matrix = np.diag(moments_var ** (-1))

    return weighting_matrix


def moments_dict_to_list(moments_dict):
    """This function constructs a list of available moments based on the moment dictionary."""
    moments_list = []
    for group in ["Wage_Distribution", "Choice_Probability"]:
        for period in sorted(moments_dict[group].keys()):
            moments_list.extend(moments_dict[group][period])
    return moments_list


def prepare_estimation(
    model_params_init_file_name, model_spec_init_file_name, data_file_name, lower, upper
):
    """Prepares objects for SMM estimation."""

    # Read in data and init file sources
    model_params_df = pd.read_pickle(model_params_init_file_name)
    data = pd.read_pickle(data_file_name)
    model_params_df["lower"] = lower
    model_params_df["upper"] = upper

    # Get moments from observed data
    moments_obs = get_moments(data)

    # Calculate weighting matrix based on bootstrap variances of observed moments
    weighting_matrix = get_weighting_matrix(data, num_agents_smm=500, num_samples=200)

    return moments_obs, weighting_matrix, model_params_df
