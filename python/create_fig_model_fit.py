import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import soepy
from smm_soepy_config import LOG_FILES_DIR, INIT_FILES_DIR
from smm_estimagic.auxiliary import get_moments

def create_fig_model_fit():
    """Plots simulated against observed moments"""

    # Get simulated moments
    model_params = pd.read_pickle(LOG_FILES_DIR / "step.soepy.pkl")
    model_params = model_params.drop(model_params.columns[1:], axis=1)
    data_sim = soepy.simulate(model_params, str(INIT_FILES_DIR) + "/model_spec_init.yml", is_expected=False)
    moments_sim = get_moments(data_sim)

    # Get observed moments
    with open( str(INIT_FILES_DIR) + "/moments_obs.pkl", "rb") as f:
        moments_obs = pickle.load(f)

    # Plot choice probabilities
    lables = ["Non-employment choice rates", "Part-time choice rates", "Full-time choice rates"]
    for choice in range(3):

        obs_choice_prob = []
        for _, value in moments_obs["Choice_Probability"].items():
            temp = [_, value]
            obs_choice_prob.append(temp[1][choice])

        sim_choice_prob = []
        for _, value in moments_sim["Choice_Probability"].items():
            temp = [_, value]
            sim_choice_prob.append(temp[1][choice])

        x = np.arange(30)

        # Start plot
        ax = plt.figure(figsize=[16, 9]).add_subplot(111)
        ax.set_ylabel(lables[choice], fontsize=16)
        ax.set_xlabel("Period", fontsize=16)

        plt.plot(x, sim_choice_prob, color='mediumturquoise')
        plt.plot(x, obs_choice_prob)

        ax.legend(["Simulated", "Observed"])

        plt.savefig("figures/choice_prob_" + str(choice) + ".pdf", ax = ax, bbox_inches='tight')
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

    x = np.arange(30)

    # Start plot
    ax = plt.figure(figsize=[16,9]).add_subplot(111)
    ax.set_ylabel(lables[choice], fontsize=16)
    ax.set_xlabel("Period", fontsize=16)

    plt.plot(x, sim_wages, color = 'mediumturquoise')
    plt.plot(x, obs_wages)

    ax.legend(["Simulated", "Observed"])

    plt.savefig('figures/wages.pdf', ax = ax, bbox_inches='tight')
    plt.close()