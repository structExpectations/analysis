import pandas as pd

LABELS_EDUCATION = ["High", "Medium", "Low"]
LABELS_CHOICE = ["Home", "Part", "Full"]
LABELS_WORK = ["Part", "Full"]


def get_moments(df):

    num_periods = df.index.get_level_values("Period").max()

    # Choice probabilities, differentiating by education, default entry is zero
    entries = [list(range(num_periods)), LABELS_EDUCATION, LABELS_CHOICE]
    conditioning = ["Period", "Education_Level", "Choice"]
    default_entry = 0

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    df_probs_grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    df_probs = (
        df.groupby(conditioning[:2]).Choice.value_counts(normalize=True).rename("Value")
    )
    df_probs_grid.update(df_probs)

    # We drop all information on early decisions among the high educated due to data issues.
    index = pd.MultiIndex.from_product([range(5), ["High"], LABELS_CHOICE])
    df_probs_grid = df_probs_grid.drop(index)

    moments = list(df_probs_grid.sort_index().values[:, 0])

    # Average wages, differentiating by education, default entry is average wage in sample
    entries = [list(range(num_periods)), LABELS_EDUCATION, LABELS_WORK]
    conditioning = ["Period", "Education_Level", "Choice"]
    default_entry = df["Wage_Observed"].mean()

    index = pd.MultiIndex.from_product(entries, names=conditioning)
    df_wages_grid = pd.DataFrame(data=default_entry, columns=["Value"], index=index)

    df_sim_working = df[df["Choice"].isin(LABELS_WORK)]
    df_wage = (
        df_sim_working.groupby(conditioning)["Wage_Observed"].mean().rename("Value")
    )
    df_wages_grid.update(df_wage)

    moments += list(df_wages_grid.sort_index().values[:, 0])

    return moments
