from sklearn.metrics import mean_squared_error
from utils import find_k_most_similar
from math import sqrt
import numpy as np
import itertools


def compute_ndcg(uim_predicted, uim_full, test, k):
    mat = np.copy(uim_predicted)
    gains = []
    test = test.astype(int)
    testUsers = np.unique(test[:, 0])
    for i in np.arange(len(testUsers)):
        user = testUsers[i]
        userSamples = test[np.where(test[:, 0] == user)]
        validItems = userSamples[:, 1]
        if len(validItems) < k:
            continue
        predictedRatings = setToZero(mat[user], validItems)
        sortedPreferences = np.flip(np.argsort(predictedRatings))
        relevances = uim_full[user, sortedPreferences]
        gains.append(ndcg_at_k(relevances, k))
    return np.mean(np.array(gains))


def compute_aggregated_diversity(uim_predicted, k):
    recommendations = []
    for user in range(uim_predicted.shape[0]):
        predicted_ratings = uim_predicted[user, :]
        top_k_recommendations = find_k_most_similar(predicted_ratings, k)
        for item in top_k_recommendations:
            recommendations.append(item)
    unique = list(dict.fromkeys(recommendations))
    AD = len(unique)/uim_predicted.shape[1]
    return AD


def compute_individual_diversity(uim_predicted, k, LF_model):
    users_diversity = []
    for user in range(uim_predicted.shape[0]):
        predicted_ratings = uim_predicted[user, :]
        top_k_recommendations = find_k_most_similar(predicted_ratings, k)
        user_div = []
        for i1, i2 in itertools.combinations(top_k_recommendations, 2):
            dissimilarity = LF_model[i1, i2]
            user_div.append(dissimilarity)
        users_diversity.append(np.mean(np.array(user_div)))
    ID = np.mean(np.array(users_diversity))
    return ID


def RMSE(uim_predicted, test_pairs):
    y_pred = []
    y_true = []
    for (user, item, rating) in test_pairs:
        y_true.append(rating)
        y_pred.append(uim_predicted[user, item])
    y_pred = np.nan_to_num(np.array(np.nan_to_num(y_pred)))
    y_true = np.nan_to_num(np.array(np.nan_to_num(y_true)))
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def LF_model(dataset, num_factors=7):
    path = f'./{dataset}/matrices/'
    uim = np.load(path + 'UIM.npy')
    uim = np.nan_to_num(uim)
    u, s, vh = np.linalg.svd(uim)
    k = num_factors
    vh = vh[0:k, :]
    item_sim = np.zeros(shape=(uim.shape[1], uim.shape[1]))
    for i in range(0, uim.shape[1]):
        for j in range(i + 1, uim.shape[1]):
            dist = np.linalg.norm(vh[:, i] - vh[:, j])
            item_sim[i, j] = dist
            item_sim[j, i] = dist
    return np.nan_to_num(item_sim)


def setToZero(row, indices):
    for element in np.arange(len(row)):
        if element not in indices:
            row[element] = 0
    return row


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
