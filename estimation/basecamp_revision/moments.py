import pandas as pd
import numpy as np

LABELS_EDUCATION = ["High", "Medium", "Low"]
LABELS_CHOICE = ["Home", "Part", "Full"]
LABELS_AGE = ['0-2', '3-5', '6-10']
LABELS_WORK = ["Part", "Full"]


def get_moments(df):

    df_int = df.copy()

    # For the observed dataset, we have many missing values in our dataset and so we must
    # restrict attention to those that work and make sure we have a numeric type.
    df_sim_working = df_int[df_int["Choice"].isin(LABELS_WORK)]
    df_sim_working = df_sim_working.astype({"Wage_Observed": np.float})

    # We need to add information on the age range of the youngest child.
    bins = pd.IntervalIndex.from_tuples([(-0.1, 2.1), (2.9, 5.1), (5.9, 10.1)])
    df_int["Age_Range"] = pd.cut(df_int["Age_Youngest_Child"], bins, labels=LABELS_AGE)

    num_periods = df_int.index.get_level_values("Period").max()

    # Choice probabilities, differentiating by education, default entry is zero
    entries = [list(range(num_periods)), LABELS_EDUCATION, LABELS_CHOICE]
    conditioning = ["Period", "Education_Level", "Choice"]
    default_entry = 0

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    df_probs_grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    df_probs = df_int.groupby(conditioning[:2]).Choice.value_counts(normalize=True).rename("Value")
    df_probs_grid.update(df_probs)

    # We drop all information on early decisions among the high educated due to data issues.
    index = pd.MultiIndex.from_product([range(5), ["High"], LABELS_CHOICE])
    df_probs_grid = df_probs_grid.drop(index)

    moments = list(df_probs_grid.sort_index().values.flatten())

    # Choice probabilities, differentiating by age range of youngest child, default entry is zero
    # We restrict attention to the first 20 periods as afterwards the cells get rather thin
    max_period = 20
    entries = [list(range(max_period)), bins.get_level_values(0), LABELS_CHOICE]
    conditioning = ["Period", "Age_Range", "Choice"]
    default_entry = 0

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    df_probs_grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    df_probs = df_int.groupby(conditioning[:2]).Choice.value_counts(normalize=True).rename("Value")
    df_probs_grid.update(df_probs)

    moments += list(df_probs_grid.sort_index().values.flatten())

    # Average wages, differentiating by education, default entry is average wage in sample
    entries = [list(range(num_periods)), LABELS_EDUCATION, LABELS_WORK]
    conditioning = ["Period", "Education_Level", "Choice"]
    default_entry = df_int["Wage_Observed"].mean()

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    df_wages_mean_grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    df_wages_mean = df_sim_working.groupby(conditioning)["Wage_Observed"].mean().rename("Value")
    df_wages_mean_grid.update(df_wages_mean)

    moments += list(df_wages_mean_grid.sort_index().values.flatten())

    # Average wages, differentiating by education and experience, default entry is average wage
    # in sample.
    default_entry = df_sim_working["Wage_Observed"].mean()

    for choice in LABELS_WORK:
        exp_label = f"Experience_{choice}_Time"

        conditioning = ["Choice", "Education_Level", exp_label]
        entries = [[choice], LABELS_EDUCATION, range(20)]
        index = pd.MultiIndex.from_product(entries, names=conditioning)

        grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)
        rslt = df_sim_working.groupby(conditioning)["Wage_Observed"].mean().rename("Value")

        grid.update(rslt)

        moments += list(grid.sort_index().values.flatten())

    # Distribution of wages, default entry is average wage in sample.
    default_entry = df_sim_working["Wage_Observed"].mean()

    quantiles = [0.1, 0.25, 0.50, 0.75, 0.9]
    conditioning = ["Choice", "Quantile"]
    entries = [LABELS_WORK, quantiles]

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)
    rslt = df_sim_working.groupby(["Choice"])["Wage_Observed"].quantile(quantiles).rename("Value")
    grid.update(rslt)

    moments += list(grid.sort_index().values.flatten())

    # Variance of wages by work status, overall, default entry is variance of wage in sample
    default_entry = df_int["Wage_Observed"].var()

    index = ["Full", "Part"]
    df_wages_var_grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    df_wages_var = df_sim_working.groupby(["Choice"])["Wage_Observed"].var().rename("Value")
    df_wages_var_grid.update(df_wages_var)

    moments += list(df_wages_var_grid.sort_index().values.flatten())

    # Persistence in choices
    df_int.loc[:, "Choice_Lagged"] = df_int.groupby("Identifier").shift(1)[["Choice"]]
    rslt = pd.crosstab(df_int["Choice"], df_int["Choice_Lagged"], normalize="index")

    moments += list(rslt.sort_index().values.flatten())

    return moments
