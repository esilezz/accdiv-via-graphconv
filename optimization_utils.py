import sys
import numpy as np
import os
import pickle as pkl
from scipy import sparse
from utils import create_users_adjacency_matrix, create_items_adjacency_matrix, add_user_means, \
    constraint_in_range, create_dicts, get_user_means, remove_user_means, get_graph_types


def open_graphs(dataset, uim, k, corr_mat, dict_to_use, method, mode):
    if mode == 'similarity':
        if method == 'USERS':
            path = f'./{dataset}/matrices/item_specific_graphs_k{k}'
        else:
            path = f'./{dataset}/matrices/user_specific_graphs_k{k}'
    else:
        if method == 'USERS':
            path = f'./{dataset}/matrices/DIS_item_specific_graphs_k{k}'
        else:
            path = f'./{dataset}/matrices/DIS_user_specific_graphs_k{k}'
    if os.path.isfile(path):
        with open(path, 'rb') as handle:
            graphs = pkl.load(handle)
    else:
        graphs = adj_matrices(uim, corr_mat, dict_to_use, k, path, method, mode)
    return graphs


def adj_matrices(uim, corr_mat, dict, k, path, method, mode):
    if method == 'USERS':
        print("CREATING USER ADJACENCY MATRICES")
        size = uim.shape[1]
    else:
        print("CREATING ITEM ADJACENCY MATRICES")
        size = uim.shape[0]
    graphs = {}

    for cnt, element in enumerate(range(0, size)):
        if cnt % 100 == 0:
            sys.stdout.write('\rProcessing %d' % cnt)
            sys.stdout.flush()

        if method == 'USERS':
            GSO = create_users_adjacency_matrix(corr_mat, dict, element, k, mode)
        else:
            GSO = create_items_adjacency_matrix(corr_mat, dict, element, k, mode)
        GSO = sparse.csr_matrix(GSO)
        graphs[element] = GSO
    with open(path, 'wb') as handle:
        pkl.dump(graphs, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print()
    return graphs


def training_matrices(alpha, uim, train, graphs_sim, graphs_dis, sim_ord, dis_ord, method1, method2, mean):
    A = np.zeros(shape=(len(train), sim_ord+dis_ord))
    y = np.zeros(shape=len(train))
    print('-------- TRAINING --------')
    for cnt_process, (user, item, rating) in enumerate(train):
        if cnt_process % 10000 == 0:
            sys.stdout.write('\rProcessing %d out of %d' % (cnt_process, len(train)))
            sys.stdout.flush()

        if method1 == 'USERS':
            x = uim[:, item]

            GSO = graphs_sim[item]
            tmp = GSO[user, :]
            for l in range(0, sim_ord):
                A[cnt_process, l] = tmp.dot(x)
                tmp = tmp.dot(GSO)
        else:
            x = uim[user, :]

            GSO = graphs_sim[user]
            tmp = GSO[item, :]
            for l in range(0, sim_ord):
                A[cnt_process, l] = tmp.dot(x)

                tmp = tmp.dot(GSO)

        if method2 == 'USERS':
            x = uim[:, item]

            GSO = graphs_dis[item]
            tmp = GSO[user, :]
            for q in range(0, dis_ord):
                A[cnt_process, l + q + 1] = tmp.dot(x)
                tmp = tmp.dot(GSO)
        else:
            x = uim[user, :]

            GSO = graphs_dis[user]
            tmp = GSO[item, :]
            for q in range(0, dis_ord):
                A[cnt_process, l + q + 1] = tmp.dot(x)
                tmp = tmp.dot(GSO)

        y[cnt_process] = rating - mean[user]

    print()
    print('--------------------------')
    return np.nan_to_num(A), y


def prediction_with_parameter(uim, graphs, h, filter_order, method):
    uim_graph_predicted = np.zeros(shape=uim.shape)
    if method == 'USER':
        tot_elements = uim.shape[1]
    else:
        tot_elements = uim.shape[0]

    for cnt, element in enumerate(range(0, tot_elements)):
        if method == 'USERS':
            x = uim[:, element]
        else:
            x = uim[element, :]

        if element in graphs:
            GSO = graphs[element]
            x_hat = np.zeros(shape=x.shape)
            tmp = x
            for l in range(filter_order):
                tmp = GSO.dot(tmp)
                x_hat = x_hat + h[l] * tmp
            if method == 'USERS':
                uim_graph_predicted[:, element] = np.nan_to_num(x_hat)
            else:
                uim_graph_predicted[element, :] = np.nan_to_num(x_hat)
        else:
            if method == 'USERS':
                uim_graph_predicted[:, element] = x
            else:
                uim_graph_predicted[element, :] = x
    uim_graph_predicted = np.nan_to_num(uim_graph_predicted)
    return uim_graph_predicted


def LLS(mat, y, alpha, h_sim_norm, h_dis_norm, mu):
    mat_T = mat.transpose()
    ident_sim = np.identity(mat.shape[1])*h_sim_norm
    ident_dis = np.identity(mat.shape[1])*h_dis_norm
    inverse = mat_T.dot(mat)+mu*((1/(1-alpha))*ident_sim + (1/alpha)*ident_dis)
    tmp = (np.linalg.inv(inverse)).dot(mat_T)
    return tmp.dot(y)


def general_double(dataset, k_most_similar, k_most_dissimilar, sim_ord, dis_ord, alpha, mu, method):
    path = f'./{dataset}/matrices/'

    train = np.load(f'{path}train_split.npy')
    B = np.load(f'{path}B.npy')
    C = np.load(f'{path}C.npy')
    uim = np.load(f'{path}uim.npy')

    # user_dict is a dict which associates to each user_id the list of the consumed items (written as item_id)
    # item_dict is a dict which associates to each item_id the list of the users who consumed the item (written as
    # user_id)
    user_dict, item_dict = create_dicts(train)

    # user mean normalization to compensate the bias
    means = get_user_means(uim, user_dict)
    uim = remove_user_means(means, uim)

    # generate the similarity and dissimilarity graphs
    if method == 'UU':
        method1 = 'USERS'
        method2 = 'USERS'

        graphs1 = open_graphs(dataset, uim, k_most_similar, B, item_dict, method1, 'similarity')
        graphs2 = open_graphs(dataset, uim, k_most_dissimilar, B, item_dict, method2, 'dissimilarity')
    elif method == 'II':
        method1 = 'ITEMS'
        method2 = 'ITEMS'

        graphs1 = open_graphs(dataset, uim, k_most_similar, C, user_dict, method1, 'similarity')
        graphs2 = open_graphs(dataset, uim, k_most_dissimilar, C, user_dict, method2, 'dissimilarity')
    elif method == 'UI':
        method1 = 'USERS'
        method2 = 'ITEMS'

        graphs1 = open_graphs(dataset, uim, k_most_similar, B, item_dict, method1, 'similarity')
        graphs2 = open_graphs(dataset, uim, k_most_dissimilar, C, user_dict, method2, 'dissimilarity')
    else:
        method1 = 'ITEMS'
        method2 = 'USERS'

        graphs1 = open_graphs(dataset, uim, k_most_similar, C, user_dict, method1, 'similarity')
        graphs2 = open_graphs(dataset, uim, k_most_dissimilar, B, item_dict, method2, 'dissimilarity')

    if sim_ord > 0 and dis_ord > 0:
        h_sim = np.random.randn(sim_ord)
        h_dis = np.random.randn(dis_ord)

        # generate the M_s and M_d matrices of equation (15). Here they are merged into a single matrix A for coding
        # purposes.
        print(f'Optimizing with alpha = {alpha}')
        A, y = training_matrices(alpha, uim, train, graphs1, graphs2, sim_ord, dis_ord, method1, method2,
                                 means)

        # apply linear least squares to find the optimal solution of the system (17)
        h_star = LLS(A, y, alpha, np.linalg.norm(h_sim), np.linalg.norm(h_dis), mu)
        h_sim = h_star[0:sim_ord]
        h_dis = h_star[-dis_ord:]

        # populate the matrix X_sim and X_dis of equation (11), here written as uim_sim and uim_dis, respectively
        uim_sim = prediction_with_parameter(uim, graphs1, h_sim, sim_ord, method1)
        uim_dis = prediction_with_parameter(uim, graphs2, h_dis, dis_ord, method2)
        uim_graph_predicted = (1 - alpha) * uim_sim + alpha * uim_dis

        uim_graph_predicted = add_user_means(means, uim_graph_predicted)
        uim_graph_predicted = constraint_in_range(uim_graph_predicted)
        return uim_graph_predicted
    else:
        print(f'ERROR - both sim_ord and dis_ord must be greater (>) than zero! (You entered sim:{sim_ord} and dis:'
              f'{dis_ord}')
        exit(-1)


def BPRprediction(uimTrain, user, item_i, item_j, GSOSimDict, GSODisDict, GSOSimOrd, GSODisOrd, h_sim, h_dis, graphType):
    graph1, graph2 = get_graph_types(graphType)
    x_ui_sim = 0
    x_uj_sim = 0
    x_ui_dis = 0
    x_uj_dis = 0

    if graph1 == 'user':
        tmp1 = uimTrain[:, item_i]
        tmp2 = uimTrain[:, item_j]
        for l in range(GSOSimOrd):
            tmp1 = h_sim[l]*GSOSimDict[item_i].dot(tmp1)
            x_ui_sim += tmp1[user]
            tmp2 = h_sim[l]*GSOSimDict[item_j].dot(tmp2)
            x_uj_sim += tmp2[user]
    else:
        tmp = uimTrain[user, :]
        for l in range(GSOSimOrd):
            tmp = h_sim[l]*GSOSimDict[user].dot(tmp)
            x_ui_sim += tmp[item_i]
            x_uj_sim += tmp[item_j]

    if graph2 == 'user':
        tmp1 = uimTrain[:, item_i]
        tmp2 = uimTrain[:, item_j]
        for l in range(GSODisOrd):
            tmp1 = h_dis[l]*GSODisDict[item_i].dot(tmp1)
            x_ui_dis += tmp1[user]
            tmp2 = h_dis[l]*GSODisDict[item_j].dot(tmp2)
            x_uj_dis += tmp2[user]
    else:
        tmp = uimTrain[user, :]
        for l in range(GSODisOrd):
            tmp = h_dis[l]*GSODisDict[user].dot(tmp)
            x_ui_dis += tmp[item_i]
            x_uj_dis += tmp[item_j]

    x_ui = x_ui_sim + x_ui_dis
    x_uj = x_uj_sim + x_uj_dis
    return x_ui, x_uj
