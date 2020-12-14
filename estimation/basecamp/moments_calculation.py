"""This module contains functions to calculate the moments based on the simulated
data."""


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
