"""This module contains functions to calculate the moments based on the simulated
data."""

NUM_PERIODS = 30


def get_moments(data):
    # Pre_process data frame

    # Determine the observed wage given period choice
    data["Wage_Observed"] = 0
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
    for period in range(30):
        moments["Wage_Distribution"][period] = []
        try:
            for label in ["mean"]:
                moments["Wage_Distribution"][period].append(info[label][period])
        except KeyError:
            for i in range(2):
                moments["Wage_Distribution"][period].append(0.0)

    # Compute unconditional moments of the choice probabilities
    info = data.groupby(["Period"])["Choice"].value_counts(normalize=True).to_dict()

    for period in range(30):  # TODO: Remove hard coded number
        moments["Choice_Probability"][period] = []
        for choice in range(3):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments["Choice_Probability"][period].append(stat)

    return moments


def get_moments_one_educ_level(data):
    # Pre_process data frame

    # Determine the observed wage given period choice
    data["Wage_Observed"] = 0
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
    for group in ["Wage_Distribution"]:
        moments[group] = dict()

    data_by_educ_level = data[data["Education_Level"] == 2]

    # Compute unconditional moments of the wage distribution
    info = (
        data_by_educ_level[data_by_educ_level["Choice"] != 0]
        .groupby(["Experience_Full_Time"])["Wage_Observed"]
        .describe()
        .to_dict()
    )

    # Save mean and standard deviation of wages for each period
    # to Wage Distribution section of the moments dictionary
    for exp in range(29):  # TODO: Remove hard coded number
        moments["Wage_Distribution"][exp] = []
        try:
            for label in ["mean"]:
                moments["Wage_Distribution"][exp].append(info[label][exp])
        except KeyError:
            moments["Wage_Distribution"][exp].append(0.0)

    return moments
