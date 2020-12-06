import os
import sys

import numpy as np
import torch


def check_base_files(dataset):
    return os.path.isfile(f'./{dataset}/matrices/test_split.npy') and \
           os.path.isfile((f'./{dataset}/matrices/uim_full.npy')) and \
           os.path.isfile(f'./{dataset}/matrices/{dataset}_item_LF.npy')


def create_dicts(pairs):
    item_dict = {}
    user_dict = {}
    for (user, item, rating) in pairs:
        if item not in item_dict:
            item_dict[item] = [user]
        else:
            item_dict[item].append(user)

        if user not in user_dict:
            user_dict[user] = [item]
        else:
            user_dict[user].append(item)
    return user_dict, item_dict


def get_available_pairs(filepath, dataset):
    available_pairs = []
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            if dataset == 'ml100k':
                fields = line.split("\t")
            elif dataset == 'ml1m':
                fields = line.split('::')
            elif dataset == 'Yahoo':
                fields = line.split(' ')
            elif dataset == 'netflix':
                fields = line.split(',')
            user = int(float(fields[0]) - 1)
            movie = int(float(fields[1]) - 1)
            rating = int(float(fields[2]))
            line = fp.readline()
            available_pairs.append((user, movie, rating))
    return available_pairs


def get_items_in_split(pairs):
    items = []
    for (user, item, rating) in pairs:
        if item not in items:
            items.append(item)
    return items


def get_users_in_split(pairs):
    users = []
    for (user, item, rating) in pairs:
        if user not in users:
            users.append(user)
    return users


def create_uim(num_users, num_items, pairs):
    uim = np.zeros(shape=(num_users, num_items))
    for (user, item, rating) in pairs:
        uim[user, item] = rating
    return uim


def create_uim_full(num_users, num_items, train, test):
    uim = np.zeros(shape=(num_users, num_items))
    for (user, item, rating) in train:
        uim[user, item] = rating
    for (user, item, rating) in test:
        uim[user, item] = rating
    return uim


def find_k_most_similar(vec, k):
    return vec.argsort()[-k:][::-1]


def find_k_most_dissimilar(vec, k):
    return vec.argsort()[0:k]


def create_users_adjacency_matrix(B, item_dict, item_i, k, mode):
    all_users = list(range(0, B.shape[0]))

    if item_i in item_dict:
        valid_users = item_dict[item_i]
    else:
        valid_users = []

    B_i = np.copy(B)
    not_valid_users = [user for user in all_users if user not in valid_users]
    B_i[:, not_valid_users] = 0
    A = np.zeros(shape=B.shape)

    for user in range(0, B.shape[0]):
        if mode == 'similarity':
            KNN = find_k_most_similar(B_i[user, :], k)
        elif mode == 'dissimilarity':
            KNN = find_k_most_dissimilar(B_i[user, :], k)

        KNN = KNN[0:np.count_nonzero(B_i[user, KNN])]

        A[user, KNN] = B_i[user, KNN]
        norm = np.sum(A[user, KNN])

        if norm == 0:
            continue
        else:
            A[user, KNN] = A[user, KNN]/norm
    return A


def create_items_adjacency_matrix(C, user_dict, user_u, k, mode):
    all_items = list(range(0, C.shape[0]))
    if user_u in user_dict:
        valid_items = user_dict[user_u]
    else:
        valid_items = []

    C_u = np.copy(C)
    not_valid_items = [item for item in all_items if item not in valid_items]
    C_u[:, not_valid_items] = 0
    A = np.zeros(shape=C.shape)

    for item in range(0, C.shape[0]):
        if mode == 'similarity':
            KNN = find_k_most_similar(C_u[item, :], k)
        elif mode == 'dissimilarity':
            KNN = find_k_most_dissimilar(C_u[item, :], k)

        KNN = KNN[0:np.count_nonzero(C_u[item, KNN])]

        A[item, KNN] = C_u[item, KNN]
        norm = np.sum(A[item, KNN])

        if norm == 0:
            continue
        else:
            A[item, KNN] = A[item, KNN]/norm
    return A


def get_user_means(uim, user_dict):
    user_means = {}
    for user in range(0, uim.shape[0]):
        if user in user_dict:
            consumed_items = user_dict[user]
            tot = uim[user, consumed_items]
        else:
            tot = []
        user_means[user] = np.mean(np.array(tot))
    return user_means


def remove_user_means(user_means, uim):
    for user in range(0, uim.shape[0]):
        if user in user_means:
            if np.isnan(user_means[user]):
                continue
            else:
                uim[user, :] = uim[user, :] - user_means[user]
    return uim


def add_user_means(user_means, mat):
    for user in range(0, mat.shape[0]):
        if np.isnan(user_means[user]):
            continue
        else:
            mat[user, :] = mat[user, :] + user_means[user]
    return mat


def constraint_in_range(mat):
    if mat.shape == (3000, 3000):
        mat[mat > 100] = 100
        mat[mat < 0] = 0
    else:
        mat[mat > 5] = 5
        mat[mat < 1] = 1
    return mat


def get_graph_types(graphType):
    if graphType == 'UU':
        return 'user', 'user'
    elif graphType == 'II':
        return 'item', 'item'
    elif graphType == 'UI':
        return 'user', 'item'
    elif graphType == 'IU':
        return 'item', 'user'
    else:
        sys.exit(-1)


def str2bool(v):
    if type(v) is bool:
        return v
    return v.lower() in ("yes", "true", "t", "1")


def float2tensor(num):
    tmp = torch.zeros(1)
    tmp[0] = num
    return tmp


def sigmoid(x):
    """
    Calculates Sigmoid of x
    """
    return 1/(1+np.exp(-x))
