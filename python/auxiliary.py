import pickle

import pandas as pd
import numpy as np


def pre_process_soep_data(file_name):
    data_full = pd.read_stata(file_name)

    # Restrict sample to age 50
    data_30periods = data_full[data_full["age"] < 47]

    # Restirct sample to west Germany
    data = data_30periods[data_30periods["east"] == 0]

    # Drop observations with missing values in hdegree
    data = data[data["hdegree"].isna() == False]

    # Generate period variable
    def get_period(row):
        return row["age"] - 17

    data["Period"] = data.apply(
        lambda row: get_period(row), axis=1
    )

    # Determine the observed wage given period choice
    def recode_educ_level(row):
        if row["hdegree"] == 'Primary/basic vocational':
            return 0
        elif row["hdegree"] == 'Abi/intermediate voc.':
            return 1
        elif row["hdegree"] == 'University':
            return 2
        else:
            return np.nan

    data["Educ Level"] = data.apply(
        lambda row: recode_educ_level(row), axis=1
    )

    # Recode choice
    # Determine the observed wage given period choice
    def recode_choice(row):
        if row["empchoice"] == 'Full-Time':
            return 2
        elif row["empchoice"] == 'Part-Time':
            return 1
        elif row["empchoice"] == 'Non-Working':
            return 0
        else:
            return np.nan

    data["Choice"] = data.apply(
        lambda row: recode_choice(row), axis=1
    )

    # Generate wage for Non-Employment choice
    data["wage_nw_imp"] = 4.00

    # Determine the observed wage given period choice
    def get_observed_wage(row):
        if row["empchoice"] == 'Full-Time':
            return row["wage_ft"]
        elif row["empchoice"] == 'Part-Time':
            return row["wage_pt"]
        elif row["empchoice"] == 'Non-Working':
            return row["wage_nw_imp"]
        else:
            return np.nan

    data["Wage Observed"] = data.apply(
        lambda row: get_observed_wage(row), axis=1
    )

    return data

def get_moments_obs(data):
    # Initialize moments dictionary
    moments = dict()

    # Store moments in groups as nested dictionary
    for group in [
        "Wage_Distribution",
        "Choice_Probability",
    ]:
        moments[group] = dict()

    # Compute unconditional moments of the wage distribution
    info = data.groupby(["Period"])["Wage Observed"].describe().to_dict()

    # Save mean and standard deviation of wages for each period
    # to Wage Distribution section of the moments dictionary
    for period in range(30):
        moments["Wage_Distribution"][period] = []
        try:
            for label in ["mean", "std"]:
                moments["Wage_Distribution"][period].append(info[label][period])
        except KeyError:
            for i in range(2):
                moments["Wage_Distribution"][period].append(
                    0.0
                )

    # Compute unconditional moments of the choice probabilities
    info = data.groupby(["Period"])["Choice"].value_counts(normalize=True).to_dict()

    for period in range(30):
        moments["Choice_Probability"][period] = []
        for choice in range(3):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments["Choice_Probability"][period].append(stat)

    return moments

def get_weighting_matrix(data_frame, num_agents_smm, num_samples):
    """Calculates the weighting matrix based on the
    moments of the observed data"""

    moments_sample = []

    # Collect n samples of moments
    for k in range(num_samples):
        data_frame_sample = data_frame.sample(n=num_agents_smm)

        moments_sample_k = get_moments_obs(data_frame_sample)

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
    for group in [
        "Wage_Distribution",
        "Choice_Probability",
    ]:
        for period in sorted(moments_dict[group].keys()):
            moments_list.extend(moments_dict[group][period])
    return moments_list

def get_moments(data):
    # Pre_process data frame

    # Determine the education level given years of experience
    data["Educ_Level"] = 0
    data.loc[(data["Years_of_Education"] >= 10) & (data["Years_of_Education"] < 12), "Educ_Level"] = 0
    data.loc[(data["Years_of_Education"] >= 12) & (data["Years_of_Education"] < 16), "Educ_Level"] = 1
    data.loc[data["Years_of_Education"] >= 16, "Educ_Level"] = 2

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
    for period in range(30):  ## TO DO: Remove hard coded number
        moments["Wage_Distribution"][period] = []
        try:
            for label in ["mean", "std"]:
                moments["Wage_Distribution"][period].append(info[label][period])
        except KeyError:
            for i in range(2):
                moments["Wage_Distribution"][period].append(0.0)

    # Compute unconditional moments of the choice probabilities
    info = data.groupby(["Period"])["Choice"].value_counts(normalize=True).to_dict()

    for period in range(30):  ## TO DO: Remove hard coded number
        moments["Choice_Probability"][period] = []
        for choice in range(3):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments["Choice_Probability"][period].append(stat)

    return moments

def prepare_estimation(
    model_params_init_file_name, lower, upper
):
    """Prepares objects for SMM estimation."""

    # Read in data and init file sources
    model_params_df = pd.read_pickle(model_params_init_file_name)
    model_params_df["lower"] = lower
    model_params_df["upper"] = upper

    return model_params_df

def get_observed_data_moments_and_weighting(data_file_name):
    """This function extracts the observed moments and the associated weighting matrix
    and saves these as pickle files."""

    data_frame_observed = pre_process_soep_data(data_file_name)
    moments_obs = get_moments_obs(data_frame_observed)
    weighting_matrix = get_weighting_matrix(
        data_frame_observed, num_agents_smm=6000, num_samples=200
    )

    with open('init_files/moments_obs.pkl', 'wb') as f:
        pickle.dump(moments_obs, f)

    with open('init_files/weighting_matrix.pkl', 'wb') as f:
        pickle.dump(weighting_matrix, f)

    return moments_obs, weighting_matrix
