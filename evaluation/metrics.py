from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import numpy as np
import random
import scipy as sp
from scipy.stats import pearsonr
from result_analysis.analyze_RSA import plot
from . import evaluation_util
import math
import logging


# This is a collection of evaluation metrics that can be used for evaluating encoding and decoding experiments.
# We re-use implementations from sklearn and scipy.


# ----- Evaluation procedures ------ #
# This method is used to do the pairwise evaluation as described by Mitchell et al. (2008).
# It is used in many encoding and decoding papers.
def pairwise_matches(prediction1, target1, prediction2, target2):
    matches = {}
    for similarity_metric_fn in [cosine_similarity, euclidean_similarity, pearson_correlation]:
        correct1 = similarity_metric_fn(prediction1, target1)
        false1 = similarity_metric_fn(prediction1, target2)
        correct2 = similarity_metric_fn(prediction2, target2)
        false2 = similarity_metric_fn(prediction2, target1)

        metric_name = similarity_metric_fn.__name__

        # In the paper, we use slightly different names:
        # Mitchell = sum, Wehbe1 = single, _Strict = strict
        matches[metric_name + "__Mitchell"] = int((correct1 + correct2) > (false1 + false2))
        matches[metric_name + "_Wehbe1"] = int(correct1 > false1)
        matches[metric_name + "_Wehbe2"] = int(correct2 > false2)
        matches[metric_name + "_Strict"] = int((correct1 > false1) & (correct2 > false2))

    return matches


# Choose a random correct prediction/target pair and select a random
# incorrect prediction for comparison. Repeat number_of_trials times.
# This method can be used, if you want to do normal cross-validation and not the weird leave-two out procedure.
def pairwise_accuracy_randomized(predictions, targets, number_of_trials):
    # We do not want to compare directly neighbouring stimuli because of the hemodynamic response pattern
    # The random sample should thus be at least 20 steps ahead (20 is a pretty long distance, we just want to be sure).
    constraint = 20
    collected_results = {}
    for trial in range(0, number_of_trials):
        for i in range(0, len(predictions)):
            prediction1 = predictions[i]
            target1 = targets[i]
            index_for_pair = random.randint(0, len(predictions) - 1)
            # Get a random value that does not fall within the constrained region
            while abs(i - index_for_pair) < constraint:
                index_for_pair = random.randint(0, len(predictions) - 1)

            prediction2 = predictions[index_for_pair]
            target2 = targets[index_for_pair]
            matches = pairwise_matches(prediction1, target1, prediction2, target2)
            collected_results = evaluation_util.add_to_collected_results(matches, collected_results)

    averaged_results = {}
    for key, matches in collected_results.items():
        avg_trial_matches = matches / float(number_of_trials)
        averaged_results[key] = avg_trial_matches

    return averaged_results


# ----- Evaluation metrics ------ #

def cosine_similarity(vector1, vector2):
    return 1 - sp.spatial.distance.cosine(vector1, vector2)


def euclidean_similarity(vector1, vector2):
    return 1 - sp.spatial.distance.euclidean(vector1, vector2)


def pearson_correlation(vector1, vector2):
    return pearsonr(vector1, vector2)[0]


# Complex means that we return the raw values, the average and the mean.
# We repeat this and report the results for only the top n voxels.

def r2_score_complex(predictions, targets, n=500):
    r2values = r2_score(targets, predictions, multioutput="raw_values")
    nan = 0
    adjusted_r2 = []

    # Constant voxels should have been removed in voxel selection.
    # It might occur though, that we find a voxel that is constantly 0 in the test data,
    # but has not been constant in the training data.
    # We need to check for these.
    for score in r2values:
        if math.isnan(score):
            print("Found nan")
            nan += 1
            adjusted_r2.append(0.0)
        else:
            adjusted_r2.append(score)

    top_r2 = sorted(adjusted_r2)[-n:]

    return adjusted_r2, np.mean(np.asarray(adjusted_r2)), np.sum(np.asarray(adjusted_r2)), top_r2, np.mean(
        np.asarray(top_r2)), np.sum(np.asarray(top_r2))


def explained_variance_complex(predictions, targets, n=500):
    ev_scores = explained_variance_score(targets, predictions, multioutput="raw_values")
    nan = 0
    adjusted_ev = []
    for score in ev_scores:
        # Constant voxels should have been removed in voxel selection.
        # It might occur though, that we find a voxel that is constantly 0 in the test data,
        # but has not been constant in the training data.
        # We need to check for these.
        if math.isnan(score):
            print("Found nan")
            nan += 1
            adjusted_ev.append(0.0)
        else:
            adjusted_ev.append(score)

    top_ev = sorted(adjusted_ev)[-n:]

    return adjusted_ev, np.mean(np.asarray(adjusted_ev)), np.sum(np.asarray(adjusted_ev)), top_ev, np.mean(
        np.asarray(top_ev)), np.sum(np.asarray(top_ev))


def explained_variance(predictions, targets):
    return explained_variance_score(targets, predictions, multioutput="raw_values")


# Jain & Huth (2018) calculate R2 as abs(correlation) * correlation
# With this metric, results are more likely to be positive than with R2 or EV.
def pearson_jain_complex(predictions, targets):
    corr_squared_per_voxel = []
    nan = 0
    for voxel_id in range(0, len(targets[0])):
        correlation = pearsonr(predictions[:, voxel_id], targets[:, voxel_id])[0]

        # Constant voxels should have been removed in voxel selection.
        # It might occur though, that we find a voxel that is constantly 0 in the test data,
        # but has not been constant in the training data.
        # We need to check for these.
        if (math.isnan(correlation)):
            print("\n\n!!!!")
            print("Encountered NaN value")
            print("Voxel: " + str(voxel_id))
            print("Predictions")
            print(predictions[:, voxel_id])
            print("Targets")
            print(targets[:, voxel_id])
            corr_squared = 0.0
            nan += 1
        else:
            corr_squared = abs(correlation) * correlation
        corr_squared_per_voxel.append(corr_squared)
    print("Number of NaN: " + str(nan))
    top_corr = sorted(corr_squared_per_voxel)[-500:]

    return np.asarray(corr_squared_per_voxel), np.mean(np.asarray(corr_squared_per_voxel)), np.sum(
        np.asarray(corr_squared_per_voxel)), np.asarray(top_corr), np.mean(np.asarray(top_corr)), np.sum(
        np.asarray(top_corr))


def pearson_complex(predictions, targets, n=500):
    correlations_per_voxel = []
    nan = 0
    for voxel_id in range(0, len(targets[0])):

        # Constant voxels should have been removed in voxel selection.
        # It might occur though, that we find a voxel that is constantly 0 in the test data,
        # but has not been constant in the training data.
        # We need to check for these.
        if math.isnan(correlation):
            print("\n\n!!!!")
            print("Encountered NaN value")
            print("Voxel: " + str(voxel_id))
            print("Predictions")
            print(predictions[:, voxel_id])
            print("Targets")
            print(targets[:, voxel_id])
        correlation = 0.0
        nan += 1
    else:
        corr_squared = abs(correlation) * correlation
    correlations_per_voxel.append(corr_squared)
    print("Number of NaN: " + str(nan))
    top_corr = sorted(correlations_per_voxel)[-500:]
    correlations_per_voxel.append(correlation)
    return np.asarray(correlations_per_voxel), np.mean(np.asarray(correlations_per_voxel)), np.sum(
        np.asarray(correlations_per_voxel)), np.asarray(top_corr), np.mean(np.asarray(top_corr)), np.sum(
        np.asarray(top_corr))


# This is used for evaluating the mapping model.
def mse(predictions, targets):
    """Mean Squared Error.
    :param predictions: (n_samples, n_outputs)
    :param targets: (n_samples, n_outputs)
    :return:
      a scalar which is mean squared error
    """
    return mean_squared_error(predictions, targets)


# ----- Representational similarity analysis ------ #

# Methods for calculating dissimilarity matrices
# In the original papers, they are called RDMs
# Data is a list of n vector lists.
#
def get_dists(data, labels=[]):
    logging.info("Calculating dissimilarity matrix")
    x = {}
    C = {}

    # For each list of vectors
    for i in np.arange(len(data)):
        x[i] = data[i]

        # Calculate distances between vectors
        print("Calculating cosine for: " + labels[i])
        C[i] = sp.spatial.distance.cdist(x[i], x[i], 'cosine') + 0.00000000001
        print("Normalizing")
        # Normalize
        C[i] /= C[i].max()

    for i in C:
        print(C[i].shape)
        print("Start plotting")
        plot(C[i], [x for x in range(1, len(C[i]+1))], labels[i],cbarlabel="Cosine Distance")
    return x, C


# Compare two or more RDMs
def compute_distance_over_dists(x, C, labels):
    logging.info("Calculate correlation over RDMs")
    keys = np.asarray(list(x.keys()))

    # We calculate three different measures.
    kullback = np.zeros((len(keys), len(keys)))
    spearman = np.zeros((len(keys), len(keys)))
    pearson = np.zeros((len(keys), len(keys)))
    for i in np.arange(len(keys)):
        for j in np.arange(len(keys)):
            corr_s = []
            corr_p = []
            kullback[i][j] = np.sum(sp.stats.entropy(C[keys[i]], C[keys[j]], base=None))
            for a, b in zip(C[keys[i]], C[keys[j]]):
                s, _ = sp.stats.spearmanr(a, b)
                p, _ = sp.stats.pearsonr(a, b)
                corr_s.append(s)
                corr_p.append(p)
            spearman[i][j] = np.mean(corr_s)
            pearson[i][j] = np.mean(corr_p)
        # # get indexes of the 6 highest values
        # max_indexes = np.argpartition(pearson[i], -6)[-6:]
        # print("Representations which correlate most with: " + labels[i])
        # # Ignore self (correlation =1)
        # print([labels[x] for x in max_indexes if not x == i])

    print(spearman, pearson, kullback)
    plot(kullback, labels, title="RDM Comparison Kullback", cbarlabel="KL Divergence")
    plot(pearson, labels, title="RDM Comparison Pearson", cbarlabel="Pearson Correlation")
    plot(spearman, labels, title="RDM Comparison Spearman", cbarlabel="Spearman Correlation")
    return spearman, pearson, kullback
