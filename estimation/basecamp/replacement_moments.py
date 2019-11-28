"""This module contains functions to calculate the moments based on the observed and simulated
datasets."""
import pickle as pkl
from collections import OrderedDict
from collections import defaultdict

import pandas as pd

EDU_RANGE = range(5, 30)
MISSING_INT = -99


def get_moments(df, optim_paras, is_store=False):
    """This function computes the moments based on a dataframe."""

    groups = [
        "Wage Distribution",
        "Wage Correlation",
        "Choice Probability",
        "Final Schooling",
    ]
    moments = OrderedDict()
    for group in groups:
        moments[group] = OrderedDict()

    # Setup of auxiliary objects
    ability_levels = sorted(list(range(len(optim_paras["observables"]["ability"]))))
    choices = sorted(list(optim_paras["choices"].keys()))

    # This needs to be constructed from the dataset and not the options as otherwise the function
    # does not work for the observed data that might have much less periods than the simulated
    # model.
    periods = sorted(df["Period"].unique())

    # We also include moments over all ability levels and these are indicated by the MISSING_INT
    # value for ability.
    ability_levels_ext = ability_levels + [MISSING_INT]

    df_indexed = df.set_index(["Identifier", "Period"], drop=True)
    df_grouped_period_ability = df_indexed.groupby(["Period", "Ability"])
    df_grouped_period = df_indexed.groupby(["Period"])

    # We now add descriptive statistics of the wage distribution and correlations. Note that there
    # might be periods where there is no information available. In this case, it is simply not
    # added to the dictionary.
    info_period_ability = df_grouped_period_ability["Wage"].describe().to_dict()
    info_period = df_grouped_period["Wage"].describe().to_dict()

    for period in periods:
        for stat in ["mean", "std"]:
            label = (period, MISSING_INT)
            try:
                info_period_ability[stat][label] = info_period[stat][period]
            except KeyError:
                pass

    for period in periods:
        for ability in ability_levels_ext:
            label = (period, ability)
            if pd.isnull(info_period_ability["std"][label]):
                continue
            moments["Wage Distribution"][label] = []
            for stat in ["mean", "std"]:
                moments["Wage Distribution"][label].append(
                    info_period_ability[stat][label]
                )

    for period in periods[:-1]:
        for ability in ability_levels_ext:
            if ability == MISSING_INT:
                subset = df_indexed
            else:
                subset = df_indexed[df_indexed["Ability"] == ability]

            current = subset["Wage"].loc[:, period]
            future = subset["Wage"].loc[:, period + 1]
            stat = current.corr(future)
            if pd.isnull(stat):
                continue
            label = (period, ability)
            moments["Wage Correlation"][label] = [stat]

    # We first compute the information about choice probabilities. We need to address the case
    # that a particular choice is not taken at all in a period and then these are not included in
    # the dictionary. This cannot be addressed by using categorical variables as the categories
    # without a value are not included after the groupby operation.
    info_period_ability = (
        df_grouped_period_ability["Choice"].value_counts(normalize=True).to_dict()
    )
    info_period = df_grouped_period["Choice"].value_counts(normalize=True).to_dict()

    # We need to fill in missing values. i.e. where for a particular subgroup no individual is
    # observed in a given choice.
    info_period_ability = defaultdict(lambda: 0.00, info_period_ability)
    info_period = defaultdict(lambda: 0.00, info_period)

    for period in periods:
        for choice in choices:
            name = (period, MISSING_INT, choice)
            info_period_ability[name] = info_period[(period, choice)]

    for period in periods:
        for ability in ability_levels_ext:
            label = (period, ability)
            moments["Choice Probability"][label] = []
            for choice in choices:
                name = (period, ability, choice)
                stat = info_period_ability[name]
                moments["Choice Probability"][label].append(stat)

    # We add the relative share of the final levels of schooling. Note that we simply loop over a
    # large range of maximal schooling levels to avoid the need to pass any further details from
    # the package to the function such as the initial and maximal level of education at this point.
    for ability in ability_levels_ext:
        if ability == MISSING_INT:
            subset = df
        else:
            subset = df[df["Ability"] == ability]

        info = subset.groupby(["Identifier"])["Experience_Edu"]
        info = info.max().value_counts(normalize=True).to_dict()

        # We need to fill in missing values as not all education levels might be observed in the
        # data. These are then not included in the dictionary.
        info = defaultdict(lambda: 0.00, info)

        for edu_max in EDU_RANGE:
            moments["Final Schooling"][(edu_max, ability)] = [info[edu_max]]

    # We might want to store the data from the moments calculation for transfer to a different
    # estimation machine.
    if is_store:
        pkl.dump(moments, open("moments.respy.pkl", "wb"))

    return moments
