import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import soepy
from configurations.analysis_soepy_config import (
    RESOURCES_DIR,
    FIGURES_DIR,
)
from python.auxiliary import get_moments


def create_fig_model_fit():
    """Plots simulated against observed moments"""

    # Get simulated moments
    model_params = pd.read_pickle(str(RESOURCES_DIR) + "/model_params.pkl")
    model_params = model_params.drop(model_params.columns[1:], axis=1)
    data_sim = soepy.simulate(
        model_params, str(RESOURCES_DIR) + "/model_spec_init.yml", is_expected=False
    )
    moments_sim = get_moments(data_sim)

    # Get observed moments
    with open(str(RESOURCES_DIR) + "/moments_obs.pkl", "rb") as f:
        moments_obs = pickle.load(f)

    # Plot choice probabilities
    lables = [
        "Non-employment choice rates",
        "Part-time choice rates",
        "Full-time choice rates",
    ]
    for choice in range(3):

        obs_choice_prob = []
        for _, value in moments_obs["Choice_Probability"].items():
            temp = [_, value]
            obs_choice_prob.append(temp[1][choice])

        sim_choice_prob = []
        for _, value in moments_sim["Choice_Probability"].items():
            temp = [_, value]
            sim_choice_prob.append(temp[1][choice])

        x = np.arange(5, 30)

        # Start plot
        ax = plt.figure(figsize=[16, 9]).add_subplot(111)
        ax.set_ylabel(lables[choice])
        ax.set_xlabel("Period")

        plt.plot(x, sim_choice_prob, color="mediumturquoise")
        plt.plot(x, obs_choice_prob)

        ax.legend(["Simulated", "Observed"])

        plt.savefig(
            str(FIGURES_DIR) + "/choice_prob_" + str(choice) + ".pdf",
            bbox_inches="tight",
        )
        plt.close()

    # Plot wages
    obs_wages = []
    for _, value in moments_obs["Wage_Distribution"].items():
        temp = [_, value]
        obs_wages.append(temp[1][0])

    sim_wages = []
    for _, value in moments_sim["Wage_Distribution"].items():
        temp = [_, value]
        sim_wages.append(temp[1][0])

    x = np.arange(5, 30)

    # Start plot
    ax = plt.figure(figsize=[16, 9]).add_subplot(111)
    ax.set_ylabel(lables[choice])
    ax.set_xlabel("Period")

    plt.plot(x, sim_wages, color="mediumturquoise")
    plt.plot(x, obs_wages)

    ax.legend(["Simulated", "Observed"])

    plt.savefig(str(FIGURES_DIR) + "/wages.pdf", bbox_inches="tight")
    plt.close()
