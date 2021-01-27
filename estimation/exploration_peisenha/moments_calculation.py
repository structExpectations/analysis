"""This module contains functions to calculate the moments based on the simulated
data."""

import numpy as np

"""This module contains functions to calculate the moments based on the simulated
data."""


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
