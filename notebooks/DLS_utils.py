# Basic data manipulation
import copy
import numpy as np
import pandas as pd
import json

# Embeddings and basic machine learning
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

# Plotting gear
import matplotlib.pyplot as plt
import seaborn as sns

# DeepOF
import deepof.data


def signed_z_score(data, z_bases, z_dls):
    """Given a domain data frame, compute the Z-scores per variable and return a single value per animal. If the mean in DLS is lower than in control, multiply by -1"""

    # Compute the Z-score per variable
    z_scores = (data - z_bases.loc["mean"]) / z_bases.loc["std"]
    # Check whether DLS animals have a higher or lower score than the control animals. If lower, multiply by -1
    z_scores = z_scores * np.sign(z_dls.loc["mean"] - z_bases.loc["mean"])

    return z_scores


def anhedonia_domain(data, columns_per_domain):
    """Process the required variables to compute the anhedonia domain Z score. Returns a single value per animal"""
    domain_data = copy.deepcopy(data)

    # Compute aggregation stats per variable on the control animals only
    z_bases = domain_data.loc[
        domain_data.Condition == "Control", columns_per_domain["Anhedonia"]
    ].agg({"mean", "std"})
    # Compute aggregation stats per variable on the DLS animals only
    z_dls = domain_data.loc[
        domain_data.Condition == "DLS", columns_per_domain["Anhedonia"]
    ].agg({"mean"})
    # Compute the Z-score per variable
    z_scores = signed_z_score(
        domain_data[columns_per_domain["Anhedonia"]], z_bases, z_dls
    )
    # Aggregate and return
    domain_data["Anhedonia"] = z_scores.mean(axis=1)

    return domain_data[["Sample name", "Condition", "Anhedonia"]].drop_duplicates()


def psychomotor_domain(data, columns_per_domain):
    """Process the required variables to compute the anhedonia domain Z score. Returns a single value per animal"""
    domain_data = copy.deepcopy(data)

    # Compute aggregation stats per variable on the control animals only
    z_bases = domain_data.loc[
        domain_data.Condition == "Control", columns_per_domain["Psychomotor changes"]
    ].agg({"mean", "std"})
    # Compute aggregation stats per variable on the DLS animals only
    z_dls = domain_data.loc[
        domain_data.Condition == "DLS", columns_per_domain["Psychomotor changes"]
    ].agg({"mean"})
    # Compute the Z-score per variable
    z_scores = signed_z_score(
        domain_data[columns_per_domain["Psychomotor changes"]], z_bases, z_dls
    )
    # Aggregate and return
    domain_data["Psychomotor changes"] = z_scores.mean(axis=1)

    return domain_data[
        ["Sample name", "Condition", "Psychomotor changes"]
    ].drop_duplicates()


def weight_domain(data, columns_per_domain):
    """Process the required variables to compute the anhedonia domain Z score. Returns a single value per animal"""
    domain_data = copy.deepcopy(data)
    # Compute the area under the curve as the only relevant variable to follow
    domain_data["Weight AUC"] = np.trapz(
        domain_data[columns_per_domain["Apetite / weight"]].iloc[:, 2:9], axis=1
    )
    # Compute Z-scores per variable on the control animals only
    z_bases = domain_data.loc[domain_data.Condition == "Control", "Weight AUC"].agg(
        {"mean", "std"}
    )
    # Check whether DLS animals have a higher or lower score than the control animals. If lower, multiply by -1
    z_dls = domain_data.loc[domain_data.Condition == "DLS", "Weight AUC"].agg(
        {"mean"}
    )
    # Compute the Z-score per variable
    z_scores = signed_z_score(domain_data["Weight AUC"], z_bases, z_dls)
    # Aggregate and return
    domain_data["Apetite / weight"] = pd.DataFrame(z_scores).mean(axis=1)

    return domain_data[
        ["Sample name", "Condition", "Apetite / weight"]
    ].drop_duplicates()


def lack_of_concentration_domain(data, columns_per_domain):
    """Process the required variables to compute the anhedonia domain Z score. Returns a single value per animal"""
    domain_data = copy.deepcopy(data)

    # Compute aggregation stats per variable on the control animals only
    z_bases = domain_data.loc[
        domain_data.Condition == "Control", columns_per_domain["Lack of concentration"]
    ].agg({"mean", "std"})
    # Compute aggregation stats per variable on the DLS animals only
    z_dls = domain_data.loc[
        domain_data.Condition == "DLS", columns_per_domain["Lack of concentration"]
    ].agg({"mean"})
    # Compute the Z-score per variable
    z_scores = signed_z_score(
        domain_data[columns_per_domain["Lack of concentration"]], z_bases, z_dls
    )
    # Aggregate and return
    domain_data["Lack of concentration"] = z_scores.mean(axis=1)

    return domain_data[
        ["Sample name", "Condition", "Lack of concentration"]
    ].drop_duplicates()


def fatigue_domain(data, columns_per_domain):
    """Process the required variables to compute the anhedonia domain Z score. Returns a single value per animal"""
    domain_data = copy.deepcopy(data)

    # Compute aggregation stats per variable on the control animals only
    z_bases = domain_data.loc[
        domain_data.Condition == "Control", columns_per_domain["Fatigue"]
    ].agg({"mean", "std"})
    # Compute aggregation stats per variable on the DLS animals only
    z_dls = domain_data.loc[
        domain_data.Condition == "DLS", columns_per_domain["Fatigue"]
    ].agg({"mean"})
    # Compute the Z-score per variable
    z_scores = signed_z_score(
        domain_data[columns_per_domain["Fatigue"]], z_bases, z_dls
    )
    # Aggregate and return
    domain_data["Fatigue"] = z_scores.mean(axis=1)

    return domain_data[["Sample name", "Condition", "Fatigue"]].drop_duplicates()


def anxiety_domain(data, columns_per_domain):
    """Process the required variables to compute the anhedonia domain Z score. Returns a single value per animal"""
    domain_data = copy.deepcopy(data)

    # Compute aggregation stats per variable on the control animals only
    z_bases = domain_data.loc[
        domain_data.Condition == "Control", columns_per_domain["Anxiety"]
    ].agg({"mean", "std"})
    # Compute aggregation stats per variable on the DLS animals only
    z_dls = domain_data.loc[
        domain_data.Condition == "DLS", columns_per_domain["Anxiety"]
    ].agg({"mean"})
    # Compute the Z-score per variable
    z_scores = signed_z_score(
        domain_data[columns_per_domain["Anxiety"]], z_bases, z_dls
    )
    # Aggregate and return
    domain_data["Anxiety"] = z_scores.mean(axis=1)

    return domain_data[["Sample name", "Condition", "Anxiety"]].drop_duplicates()


def biological_domain(data, columns_per_domain):
    """Process the required variables to compute the anhedonia domain Z score. Returns a single value per animal"""
    domain_data = copy.deepcopy(data)

    # Compute aggregation stats per variable on the control animals only
    z_bases = domain_data.loc[
        domain_data.Condition == "Control", columns_per_domain["Biological markers"]
    ].agg({"mean", "std"})
    # Compute aggregation stats per variable on the DLS animals only
    z_dls = domain_data.loc[
        domain_data.Condition == "DLS", columns_per_domain["Biological markers"]
    ].agg({"mean"})
    # Compute the Z-score per variable
    z_scores = signed_z_score(
        domain_data[columns_per_domain["Biological markers"]], z_bases, z_dls
    )
    # Aggregate and return
    domain_data["Biological markers"] = z_scores.mean(axis=1)

    return domain_data[["Sample name", "Condition", "Biological markers"]].drop_duplicates()


def social_domain(data, columns_per_domain):
    """Process the required variables to compute the anhedonia domain Z score. Returns a single value per animal"""
    domain_data = copy.deepcopy(data)

    # Compute aggregation stats per variable on the control animals only
    z_bases = domain_data.loc[
        domain_data.Condition == "Control",
        columns_per_domain["Socio-temporal functions"],
    ].agg({"mean", "std"})
    # Compute aggregation stats per variable on the DLS animals only
    z_dls = domain_data.loc[
        domain_data.Condition == "DLS",
        columns_per_domain["Socio-temporal functions"],
    ].agg({"mean"})
    # Compute the Z-score per variable
    z_scores = signed_z_score(
        domain_data[columns_per_domain["Socio-temporal functions"]], z_bases, z_dls
    )
    # Aggregate and return
    domain_data["Socio-temporal functions"] = z_scores.mean(axis=1)

    return domain_data[
        ["Sample name", "Condition", "Socio-temporal functions"]
    ].drop_duplicates()


def compute_domain_scores(data, columns_per_domain):
    """Compute the domain scores for all animals"""

    # Compute the domain scores
    domain_score_dict = {
        "Anhedonia": anhedonia_domain(data, columns_per_domain),
        "Psychomotor changes": psychomotor_domain(data, columns_per_domain),
        "Apetite / weight": weight_domain(data, columns_per_domain),
        "Lack of concentration": lack_of_concentration_domain(data, columns_per_domain),
        "Fatigue": fatigue_domain(data, columns_per_domain),
        "Anxiety": anxiety_domain(data, columns_per_domain),
        "Biological markers": biological_domain(data, columns_per_domain),
        "Socio-temporal functions": social_domain(data, columns_per_domain),
    }

    # Merge the domain scores into a single data frame
    domain_scores = domain_score_dict["Anhedonia"]
    for domain in domain_score_dict.keys():
        if domain != "Anhedonia":
            domain_scores = domain_scores.merge(
                domain_score_dict[domain], on=["Sample name", "Condition"]
            )

    # Compute the total score
    domain_scores["DLS score"] = domain_scores.iloc[:, 2:].sum(axis=1)

    return domain_scores.drop_duplicates(["Sample name", "Condition"])
