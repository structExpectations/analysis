"""This module contains functions to calculate the moments based on the simulated
data."""

import numpy as np


def get_moments(data):

    # Initialize moments dictionary
    moments = dict()

    # Store moments in groups as nested dictionary
    for group in [
        "Wage_Distribution",
        "Choice_Probability",
    ]:
        moments[group] = dict()

    # Compute unconditional moments of the wage distribution
    info = (
        data[data["Choice"] != 0]
        .groupby(["Period"])["Wage_Observed"]
        .describe()
        .to_dict()
    )

    # Save mean and standard deviation of wages for each period
    # to Wage Distribution section of the moments dictionary
    for period in range(39):
        moments["Wage_Distribution"][period] = []
        try:
            for label in ["mean", "std"]:
                moments["Wage_Distribution"][period].append(info[label][period])
        except KeyError:
            for i in range(2):
                moments["Wage_Distribution"][period].append(0.0)

    # Compute unconditional moments of the choice probabilities
    info = data.groupby(["Period"])["Choice"].value_counts(normalize=True).to_dict()

    for period in range(39):  # TODO: Remove hard coded number
        moments["Choice_Probability"][period] = []
        for choice in range(3):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments["Choice_Probability"][period].append(stat)

    return moments


def transitions_out_to_in(data_subset):
    counts_list = []
    for period in np.arange(1, 39):
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
            (
                data_subset[
                    (data_subset["Period"] == period)
                    & (data_subset["Identifier"].isin(transition_ids))
                ]
                .groupby(["Education_Level"])["Identifier"]
                .count()
                / data_subset[(data_subset["Period"] == period)]
                .groupby(["Education_Level"])["Identifier"]
                .count()
            )
            .fillna(0)
            .to_list()
        )
        counts_list += [period_counts]
    avg = [float(sum(col)) / len(col) for col in zip(*counts_list)]
    return avg


def transitions_out_to_in_mothers(data_subset):
    counts_list = []
    for period in np.arange(1, 28):
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
            (
                data_subset[
                    (data_subset["Period"] == period)
                    & (data_subset["Identifier"].isin(transition_ids))
                ]
                .groupby(["Education_Level"])["Identifier"]
                .count()
                / data_subset[(data_subset["Period"] == period)]
                .groupby(["Education_Level"])["Identifier"]
                .count()
            )
            .fillna(0)
            .to_list()
        )
        counts_list += [period_counts]
    avg = [float(sum(col)) / len(col) for col in zip(*counts_list)]
    return avg


def transitions_in_to_out(data_subset):
    counts_list = []
    for period in np.arange(1, 39):
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
            (
                data_subset[
                    (data_subset["Period"] == period)
                    & (data_subset["Identifier"].isin(transition_ids))
                ]
                .groupby(["Education_Level"])["Identifier"]
                .count()
                / data_subset[(data_subset["Period"] == period)]
                .groupby(["Education_Level"])["Identifier"]
                .count()
            )
            .fillna(0)
            .to_list()
        )
        counts_list += [period_counts]
    avg = [float(sum(col)) / len(col) for col in zip(*counts_list)]
    return avg


def transitions_in_to_out_mothers(data_subset):
    counts_list = []
    for period in np.arange(1, 28):
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
            (
                data_subset[
                    (data_subset["Period"] == period)
                    & (data_subset["Identifier"].isin(transition_ids))
                ]
                .groupby(["Education_Level"])["Identifier"]
                .count()
                / data_subset[(data_subset["Period"] == period)]
                .groupby(["Education_Level"])["Identifier"]
                .count()
            )
            .fillna(0)
            .to_list()
        )
        counts_list += [period_counts]
    avg = [float(sum(col)) / len(col) for col in zip(*counts_list)]
    return avg


def transitions_in_to_out_deciles(data, decile):
    counts_list = []
    for period in np.arange(1, 39):
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
            (
                data[
                    (data["Period"] == period)
                    & (data["Identifier"].isin(transition_ids))
                ]
                .groupby(["Education_Level"])["Identifier"]
                .count()
                / data[(data["Period"] == period)]
                .groupby(["Education_Level"])["Identifier"]
                .count()
            )
            .fillna(0)
            .to_list()
        )
        counts_list += [period_counts]
    avg = [float(sum(col)) / len(col) for col in zip(*counts_list)]

    return avg


def get_moments_disutility(data):
    moments = dict()

    # Store moments in groups as nested dictionary
    for group in [
        "Employment_Shares",
        "Transitions",
    ]:
        moments[group] = dict()

    # All employment
    moments["Employment_Shares"]["All_Employment"] = []
    # Single no child
    moments["Employment_Shares"]["All_Employment"].append(
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
        ).values
    )
    # Married no child
    moments["Employment_Shares"]["All_Employment"].append(
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
    moments["Employment_Shares"]["All_Employment"].append(
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
    moments["Employment_Shares"]["All_Employment"].append(
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
    moments["Employment_Shares"]["All_Employment"].append(
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
    moments["Employment_Shares"]["All_Employment"].append(
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
    moments["Employment_Shares"]["All_Employment"].append(
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
    moments["Employment_Shares"]["Part_Time_Employment"].append(
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
    moments["Employment_Shares"]["Part_Time_Employment"].append(
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
    moments["Employment_Shares"]["Part_Time_Employment"].append(
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
    moments["Employment_Shares"]["Part_Time_Employment"].append(
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
    moments["Employment_Shares"]["Part_Time_Employment"].append(
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
    moments["Employment_Shares"]["Part_Time_Employment"].append(
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
    moments["Employment_Shares"]["Part_Time_Employment"].append(
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

    # Transition rates from out of work into work by education
    moments["Transitions"]["Out_To_In"] = []

    # All
    avg = transitions_out_to_in(data)
    moments["Transitions"]["Out_To_In"].append(avg)

    # Women with no children
    data_subset = data[data["Age_Youngest_Child"] == -1]
    avg = transitions_out_to_in(data_subset)
    moments["Transitions"]["Out_To_In"].append(avg)

    # Lone mothers
    data_subset = data[
        (data["Age_Youngest_Child"] != -1) & (data["Partner_Indicator"] == 0)
    ]
    avg = transitions_out_to_in_mothers(data_subset)
    moments["Transitions"]["Out_To_In"].append(avg)

    # Married mothers
    data_subset = data[
        (data["Age_Youngest_Child"] != -1) & (data["Partner_Indicator"] == 1)
    ]
    avg = transitions_out_to_in_mothers(data_subset)
    moments["Transitions"]["Out_To_In"].append(avg)

    # Transition rates from employment to out of work by education
    moments["Transitions"]["In_To_Out"] = []

    # All
    avg = transitions_in_to_out(data)
    moments["Transitions"]["In_To_Out"].append(avg)

    # Women with no children
    data_subset = data[data["Age_Youngest_Child"] == -1]
    avg = transitions_in_to_out(data_subset)
    moments["Transitions"]["In_To_Out"].append(avg)

    # Lone mothers
    data_subset = data[
        (data["Age_Youngest_Child"] != -1) & (data["Partner_Indicator"] == 0)
    ]
    avg = transitions_in_to_out_mothers(data_subset)
    moments["Transitions"]["In_To_Out"].append(avg)

    # Married mothers
    data_subset = data[
        (data["Age_Youngest_Child"] != -1) & (data["Partner_Indicator"] == 1)
    ]
    avg = transitions_in_to_out_mothers(data_subset)
    moments["Transitions"]["In_To_Out"].append(avg)

    # Past wage in bottom decile
    avg = transitions_in_to_out_deciles(data, 0.1)
    moments["Transitions"]["In_To_Out"].append(avg)

    # Past wage below median
    avg = transitions_in_to_out_deciles(data, 0.5)
    moments["Transitions"]["In_To_Out"].append(avg)

    # Past wage below 90th percentile
    avg = transitions_in_to_out_deciles(data, 0.9)
    moments["Transitions"]["In_To_Out"].append(avg)

    return moments
