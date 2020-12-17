import pickle

import pandas as pd
import numpy as np

from configurations.analysis_soepy_config import RESOURCES_DIR

NUM_PERIODS = 30


def pre_process_soep_data(file_name):
    data_full = pd.read_stata(file_name)

    # Restrict sample to age 50
    data_30periods = data_full[data_full["age"] < 47]

    # Restrict sample to west Germany
    data = data_30periods[data_30periods["east"] == 0]

    # Drop observations with missing values in hdegree
    data = data[data["hdegree"].notna()]

    # Generate period variable
    def get_period(row):
        return row["age"] - 17

    data["Period"] = data.apply(lambda row: get_period(row), axis=1)

    # Determine the level of education
    def recode_educ_level(row):
        if row["hdegree"] == "Primary/basic vocational":
            return 0
        elif row["hdegree"] == "Abi/intermediate voc.":
            return 1
        elif row["hdegree"] == "University":
            return 2
        else:
            return np.nan

    data["Educ_Level"] = data.apply(lambda row: recode_educ_level(row), axis=1)

    # Recode choice
    def recode_choice(row):
        if row["empchoice"] == "Full-Time":
            return 2
        elif row["empchoice"] == "Part-Time":
            return 1
        elif row["empchoice"] == "Non-Working":
            return 0
        else:
            return np.nan

    data["Choice"] = data.apply(lambda row: recode_choice(row), axis=1)

    # Determine the observed wage given period choice
    def get_observed_wage(row):
        if row["empchoice"] == "Full-Time":
            return row["wage_ft"] * 4.5 * 38
        elif row["empchoice"] == "Part-Time":
            return row["wage_pt"] * 4.5 * 18
        else:
            return 700

    data["Wage_Observed"] = data.apply(lambda row: get_observed_wage(row), axis=1)
    data.loc[(data["Choice"] == 0) & (data["age"] > 27), "Wage_Observed"] = 1000

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
    info = data.groupby(["Period"])["Wage_Observed"].describe().to_dict()

    # Save mean and standard deviation of wages for each period
    # to Wage Distribution section of the moments dictionary
    for period in range(30):
        moments["Wage_Distribution"][period] = []
        try:
            for label in ["mean", "std"]:
                moments["Wage_Distribution"][period].append(info[label][period])
        except KeyError:
            for i in range(2):
                moments["Wage_Distribution"][period].append(0.0)

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


def prepare_estimation(model_params_init_file_name, lower, upper):
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

    with open(str(RESOURCES_DIR) + "/moments_obs.pkl", "wb") as f:
        pickle.dump(moments_obs, f)

    with open(str(RESOURCES_DIR) + "/weighting_matrix.pkl", "wb") as f:
        pickle.dump(weighting_matrix, f)

    return moments_obs, weighting_matrix


def transitions_out_to_in(data_subset, num_periods):
    counts_list = []
    for period in np.arange(1, num_periods):
        # get period IDs:
        period_employed_ids = data_subset[
            (data_subset["Period"] == period) & (data_subset["Choice"] != 0)
        ]["Identifier"].to_list()
        transition_ids = data_subset[
            (data_subset["Period"] == period - 1)
            & (data_subset["Identifier"].isin(period_employed_ids))
            & (data_subset["Choice"] == 0)
        ]["Identifier"].to_list()
        period_counts = (
            data_subset[
                (data_subset["Period"] == period)
                & (data_subset["Identifier"].isin(transition_ids))
            ]["Identifier"].count()
            / data_subset[(data_subset["Period"] == period)]["Identifier"].count()
        )
        counts_list += [period_counts]
    avg = np.mean(counts_list)
    return avg


def transitions_out_to_in_mothers(data_subset, num_periods):
    counts_list = []
    for period in np.arange(1, min(28, num_periods)):
        # get period IDs:
        period_employed_ids = data_subset[
            (data_subset["Period"] == period) & (data_subset["Choice"] != 0)
        ]["Identifier"].to_list()
        transition_ids = data_subset[
            (data_subset["Period"] == period - 1)
            & (data_subset["Identifier"].isin(period_employed_ids))
            & (data_subset["Choice"] == 0)
        ]["Identifier"].to_list()
        period_counts = (
            data_subset[
                (data_subset["Period"] == period)
                & (data_subset["Identifier"].isin(transition_ids))
            ]["Identifier"].count()
            / data_subset[(data_subset["Period"] == period)]["Identifier"].count()
        )
        counts_list += [period_counts]
    avg = np.mean(counts_list)
    return avg


def transitions_in_to_out(data_subset, num_periods):
    counts_list = []
    for period in np.arange(1, num_periods):
        # get period IDs:
        period_unemployed_ids = data_subset[
            (data_subset["Period"] == period) & (data_subset["Choice"] == 0)
        ]["Identifier"].to_list()
        transition_ids = data_subset[
            (data_subset["Period"] == period - 1)
            & (data_subset["Identifier"].isin(period_unemployed_ids))
            & (data_subset["Choice"] != 0)
        ]["Identifier"].to_list()
        period_counts = (
            data_subset[
                (data_subset["Period"] == period)
                & (data_subset["Identifier"].isin(transition_ids))
            ]["Identifier"].count()
            / data_subset[(data_subset["Period"] == period)]["Identifier"].count()
        )
        counts_list += [period_counts]
    avg = np.mean(counts_list)
    return avg


def transitions_in_to_out_mothers(data_subset, num_periods):
    counts_list = []
    for period in np.arange(1, min(28, num_periods)):
        # get period IDs:
        period_unemployed_ids = data_subset[
            (data_subset["Period"] == period) & (data_subset["Choice"] == 0)
        ]["Identifier"].to_list()
        transition_ids = data_subset[
            (data_subset["Period"] == period - 1)
            & (data_subset["Identifier"].isin(period_unemployed_ids))
            & (data_subset["Choice"] != 0)
        ]["Identifier"].to_list()
        period_counts = (
            data_subset[
                (data_subset["Period"] == period)
                & (data_subset["Identifier"].isin(transition_ids))
            ]["Identifier"].count()
            / data_subset[(data_subset["Period"] == period)]["Identifier"].count()
        )
        counts_list += [period_counts]
    avg = np.mean(counts_list)
    return avg


def transitions_in_to_out_deciles(data, decile, num_periods):
    counts_list = []
    for period in np.arange(1, num_periods):
        # get period IDs:
        period_unemployed_ids = data[
            (data["Period"] == period) & (data["Choice"] == 0)
        ]["Identifier"].to_list()
        transition_ids = data[
            (data["Period"] == period - 1)
            & (data["Identifier"].isin(period_unemployed_ids))
            & (data["Wage_Observed"] < data["Wage_Observed"].quantile(decile))
        ]["Identifier"].to_list()
        period_counts = (
            data[
                (data["Period"] == period) & (data["Identifier"].isin(transition_ids))
            ]["Identifier"].count()
            / data[(data["Period"] == period)]["Identifier"].count()
        )
        counts_list += [period_counts]
    avg = np.mean(counts_list)

    return avg


def get_moments(data):
    num_periods = data["Period"].max() + 1
    moments = dict()

    # Store moments in groups as nested dictionary
    for group in [
        "Wage_Distribution_Educ_Low",
        "Wage_Distribution_Educ_Middle",
        "Wage_Distribution_Educ_High",
        "Wage_Distribution",
        "Wage_By_Full_Time_Experience_Educ_Low",
        "Wage_By_Full_Time_Experience_Educ_Middle",
        "Wage_By_Full_Time_Experience_Educ_High",
        "Choice_Probability",
        "Employment_Shares",
        "Transitions",
    ]:
        moments[group] = dict()

    # Wage distribution by education
    names = [
        "Wage_Distribution_Educ_Low",
        "Wage_Distribution_Educ_Middle",
        "Wage_Distribution_Educ_High",
    ]
    for educ_level in range(3):
        data_subset = data[data["Education_Level"] == educ_level]

        info = (
            data_subset[data_subset["Choice"] != 0]
            .groupby(["Period"])["Wage_Observed"]
            .describe()
            .to_dict()
        )

        for period in range(educ_level, 39):
            moments[names[educ_level]][period] = []
            try:
                for label in ["mean", "std"]:
                    moments[names[educ_level]][period].append(info[label][period])
            except KeyError:
                for i in range(2):
                    moments[names[educ_level]][period].append(np.nan)

    # Wages at entrace of working life
    moments["Wage_Distribution"]["First_Wage"] = []
    moments["Wage_Distribution"]["First_Wage"].extend(
        data[(data["Period"].isin(range(4))) & (data["Choice"] != 0)]
        .groupby(["Education_Level"])["Wage_Observed"]
        .mean()
    )
    moments["Wage_Distribution"]["First_Wage"].extend(
        data[(data["Period"].isin(range(4))) & (data["Choice"] != 0)]
        .groupby(["Education_Level"])["Wage_Observed"]
        .std()
    )
    moments["Wage_Distribution"]["First_Wage"].extend(
        data[(data["Period"].isin(range(4))) & (data["Choice"] != 0)]
        .groupby(["Education_Level"])["Wage_Observed"]
        .quantile(0.25)
    )
    moments["Wage_Distribution"]["First_Wage"].extend(
        data[(data["Period"].isin(range(4))) & (data["Choice"] != 0)]
        .groupby(["Education_Level"])["Wage_Observed"]
        .quantile(0.5)
    )
    moments["Wage_Distribution"]["First_Wage"].extend(
        data[(data["Period"].isin(range(4))) & (data["Choice"] != 0)]
        .groupby(["Education_Level"])["Wage_Observed"]
        .quantile(0.75)
    )

    # Wages by full time experience
    names = [
        "Wage_By_Full_Time_Experience_Educ_Low",
        "Wage_By_Full_Time_Experience_Educ_Middle",
        "Wage_By_Full_Time_Experience_Educ_High",
    ]

    for educ_level in range(3):
        data_subset = data[data["Education_Level"] == educ_level]
        info = (
            data_subset[data_subset["Choice"] != 0]
            .groupby(["Experience_Full_Time"])["Wage_Observed"]
            .describe()
            .to_dict()
        )
        for exp in range(20):
            moments[names[educ_level]][exp] = []
            try:
                moments[names[educ_level]][exp].append(info["mean"][exp])
            except KeyError:
                moments[names[educ_level]][exp].append(np.nan)

    # Wage distribution during working life
    data_subset = data[data["Choice"] == 2]
    moments["Wage_Distribution"]["Full_Time"] = []
    moments["Wage_Distribution"]["Full_Time"].extend(
        data_subset.groupby(["Education_Level"])["Wage_Observed"].mean()
    )
    moments["Wage_Distribution"]["Full_Time"].extend(
        data_subset.groupby(["Education_Level"])["Wage_Observed"].quantile(0.1)
    )
    moments["Wage_Distribution"]["Full_Time"].extend(
        data_subset.groupby(["Education_Level"])["Wage_Observed"].quantile(0.25)
    )
    moments["Wage_Distribution"]["Full_Time"].extend(
        data_subset.groupby(["Education_Level"])["Wage_Observed"].quantile(0.5)
    )
    moments["Wage_Distribution"]["Full_Time"].extend(
        data_subset.groupby(["Education_Level"])["Wage_Observed"].quantile(0.75)
    )
    moments["Wage_Distribution"]["Full_Time"].extend(
        data_subset.groupby(["Education_Level"])["Wage_Observed"].quantile(0.9)
    )

    # Unconditional moments of the choice probabilities
    info = data.groupby(["Period"])["Choice"].value_counts(normalize=True).to_dict()

    for period in range(39):  # TODO: Remove hard coded number
        moments["Choice_Probability"][period] = []
        for choice in range(3):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = np.nan
            moments["Choice_Probability"][period].append(stat)

    # Employment
    # All employment
    moments["Employment_Shares"]["All_Employment"] = []
    # Single no child
    moments["Employment_Shares"]["All_Employment"].extend(
        (
            data[
                (data["Partner_Indicator"] == 0)
                & (data["Age_Youngest_Child"] == -1)
                & (data["Choice"] != 0)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[
                (data["Partner_Indicator"] == 0) & (data["Age_Youngest_Child"] == -1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )

    # Married no child
    moments["Employment_Shares"]["All_Employment"].extend(
        (
            data[
                (data["Partner_Indicator"] == 1)
                & (data["Age_Youngest_Child"] == -1)
                & (data["Choice"] != 0)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[
                (data["Partner_Indicator"] == 1) & (data["Age_Youngest_Child"] == -1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )
    # Lone mothers
    moments["Employment_Shares"]["All_Employment"].extend(
        (
            data[
                (data["Partner_Indicator"] == 0)
                & (data["Age_Youngest_Child"] != -1)
                & (data["Choice"] != 0)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[
                (data["Partner_Indicator"] == 0) & (data["Age_Youngest_Child"] != -1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )
    # Married mothers
    moments["Employment_Shares"]["All_Employment"].extend(
        (
            data[
                (data["Partner_Indicator"] == 1)
                & (data["Age_Youngest_Child"] != -1)
                & (data["Choice"] != 0)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[
                (data["Partner_Indicator"] == 1) & (data["Age_Youngest_Child"] != -1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )
    # Child 0-2
    moments["Employment_Shares"]["All_Employment"].extend(
        (
            data[(data["Age_Youngest_Child"].isin([0, 1, 2])) & (data["Choice"] != 0)]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[(data["Age_Youngest_Child"].isin([0, 1, 2]))]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )
    # Child 3-5
    moments["Employment_Shares"]["All_Employment"].extend(
        (
            data[(data["Age_Youngest_Child"].isin([3, 4, 5])) & (data["Choice"] != 0)]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[(data["Age_Youngest_Child"].isin([3, 4, 5]))]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )
    # Child 6-10
    moments["Employment_Shares"]["All_Employment"].extend(
        (
            data[
                (data["Age_Youngest_Child"].isin([6, 7, 8, 9, 10]))
                & (data["Choice"] != 0)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[(data["Age_Youngest_Child"].isin([6, 7, 8, 9, 10]))]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )

    # Part-time employment
    moments["Employment_Shares"]["Part_Time_Employment"] = []
    # Single no child
    moments["Employment_Shares"]["Part_Time_Employment"].extend(
        (
            data[
                (data["Partner_Indicator"] == 0)
                & (data["Age_Youngest_Child"] == -1)
                & (data["Choice"] == 1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[
                (data["Partner_Indicator"] == 0) & (data["Age_Youngest_Child"] == -1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )
    # Married no child
    moments["Employment_Shares"]["Part_Time_Employment"].extend(
        (
            data[
                (data["Partner_Indicator"] == 1)
                & (data["Age_Youngest_Child"] == -1)
                & (data["Choice"] == 1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[
                (data["Partner_Indicator"] == 1) & (data["Age_Youngest_Child"] == -1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )
    # Lone mothers
    moments["Employment_Shares"]["Part_Time_Employment"].extend(
        (
            data[
                (data["Partner_Indicator"] == 0)
                & (data["Age_Youngest_Child"] != -1)
                & (data["Choice"] == 1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[
                (data["Partner_Indicator"] == 0) & (data["Age_Youngest_Child"] != -1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )
    # Married mothers
    moments["Employment_Shares"]["Part_Time_Employment"].extend(
        (
            data[
                (data["Partner_Indicator"] == 1)
                & (data["Age_Youngest_Child"] != -1)
                & (data["Choice"] == 1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[
                (data["Partner_Indicator"] == 1) & (data["Age_Youngest_Child"] != -1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )
    # Child 0-2
    moments["Employment_Shares"]["Part_Time_Employment"].extend(
        (
            data[(data["Age_Youngest_Child"].isin([0, 1, 2])) & (data["Choice"] == 1)]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[(data["Age_Youngest_Child"].isin([0, 1, 2]))]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )
    # Child 3-5
    moments["Employment_Shares"]["Part_Time_Employment"].extend(
        (
            data[(data["Age_Youngest_Child"].isin([3, 4, 5])) & (data["Choice"] == 1)]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[(data["Age_Youngest_Child"].isin([3, 4, 5]))]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )
    # Child 6-10
    moments["Employment_Shares"]["Part_Time_Employment"].extend(
        (
            data[
                (data["Age_Youngest_Child"].isin([6, 7, 8, 9, 10]))
                & (data["Choice"] == 1)
            ]
            .groupby(["Education_Level"])["Identifier"]
            .count()
            / data[(data["Age_Youngest_Child"].isin([6, 7, 8, 9, 10]))]
            .groupby(["Education_Level"])["Identifier"]
            .count()
        ).to_list()
    )

    # Transitions
    # Transition rates from out of work into work by education
    moments["Transitions"]["Out_To_In"] = []

    # All
    avg = transitions_out_to_in(data, num_periods)
    moments["Transitions"]["Out_To_In"].extend([avg])

    # Women with no children
    data_subset = data[data["Age_Youngest_Child"] == -1]
    avg = transitions_out_to_in(data_subset, num_periods)
    moments["Transitions"]["Out_To_In"].extend([avg])

    # Lone mothers
    data_subset = data[
        (data["Age_Youngest_Child"] != -1) & (data["Partner_Indicator"] == 0)
    ]
    avg = transitions_out_to_in_mothers(data_subset, num_periods)
    moments["Transitions"]["Out_To_In"].extend([avg])

    # Married mothers
    data_subset = data[
        (data["Age_Youngest_Child"] != -1) & (data["Partner_Indicator"] == 1)
    ]
    avg = transitions_out_to_in_mothers(data_subset, num_periods)
    moments["Transitions"]["Out_To_In"].extend([avg])

    # Transition rates from employment to out of work by education
    moments["Transitions"]["In_To_Out"] = []

    # All
    avg = transitions_in_to_out(data, num_periods)
    moments["Transitions"]["In_To_Out"].extend([avg])

    # Women with no children
    data_subset = data[data["Age_Youngest_Child"] == -1]
    avg = transitions_in_to_out(data_subset, num_periods)
    moments["Transitions"]["In_To_Out"].extend([avg])

    # Lone mothers
    data_subset = data[
        (data["Age_Youngest_Child"] != -1) & (data["Partner_Indicator"] == 0)
    ]
    avg = transitions_in_to_out_mothers(data_subset, num_periods)
    moments["Transitions"]["In_To_Out"].extend([avg])

    # Married mothers
    data_subset = data[
        (data["Age_Youngest_Child"] != -1) & (data["Partner_Indicator"] == 1)
    ]
    avg = transitions_in_to_out_mothers(data_subset, num_periods)
    moments["Transitions"]["In_To_Out"].extend([avg])

    # Past wage in bottom decile
    avg = transitions_in_to_out_deciles(data, 0.1, num_periods)
    moments["Transitions"]["In_To_Out"].extend([avg])

    # Past wage below median
    avg = transitions_in_to_out_deciles(data, 0.5, num_periods)
    moments["Transitions"]["In_To_Out"].extend([avg])

    # Past wage below 90th percentile
    avg = transitions_in_to_out_deciles(data, 0.9, num_periods)
    moments["Transitions"]["In_To_Out"].extend([avg])

    return moments
