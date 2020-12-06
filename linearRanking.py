import argparse
import ast
import time
import datetime
import os
import numpy as np
from data_utils import BPRSampling, Movielens100kBPR

from torch.utils.data import DataLoader

from metrics import compute_ndcg
from optimization_utils import open_graphs, prediction_with_parameter, BPRprediction
from test_train_split import train_test_split
from utils import get_graph_types, create_dicts, check_base_files, str2bool, sigmoid


def runAll(args, alpha):
    ##### PARAMETERS DECLARATION #####
    graphType = args.graphType
    dataset = args.dataset
    if not check_base_files(dataset):
        train_test_split(dataset)

    path = f'./{dataset}/matrices/'
    B = np.load(path + 'B.npy')
    C = np.load(path + 'C.npy')

    if args.dataset == 'ml100k' or args.dataset == 'ml1M':
        ndcg_items = 20
    else:
        ndcg_items = 5

    #### 1 LAYER ARCHITECTURE ####
    GSOSimOrder = args.simOrder  # filter order for the similarity network
    GSODisOrder = args.disOrder  # filter order for the dissimilarity network

    nEpochs = args.nEpochs  # how many times we cross the entire dataset
    lr = args.lr
    mu = args.mu
    neg = args.numNeg

    print(f'{graphType.upper()} -> running with {GSOSimOrder} sim, {GSODisOrder} dis, '
          f'{lr} learning rate, mu={mu}, neg.samples={neg}, alpha={alpha}, '
          f'epochs={nEpochs}')

    if args.dataset == 'ml100k':
        n_users = 943
        n_items = 1682
    elif args.dataset == 'ml1M':
        n_users = 6040
        n_items = 3952
    else:
        n_users = 3000
        n_items = 3000

    ##### DATASET INITIALIZATION #####
    uimFull, uimTrain, train, testSignals = BPRSampling(args.dataset, n_users, n_items)
    means = np.mean(uimTrain, axis=0)
    uimTrain = uimTrain - means
    h_sim = np.random.randn(GSOSimOrder)
    h_dis = np.random.randn(GSODisOrder)

    data = Movielens100kBPR(uimTrain, uimTrain, train, args.numNeg, args.dataset)
    loader = DataLoader(data, batch_size=1, shuffle=True)

    ##### MODEL CREATION #####
    print('Model initialization...')
    graph1, graph2 = get_graph_types(graphType)
    user_dict, item_dict = create_dicts(train)

    if graph1 == 'user':
        corr_mat = B
        dict_to_use = item_dict
        method1 = 'USERS'
        k = 30
    else:
        corr_mat = C
        dict_to_use = user_dict
        method1 = 'ITEMS'
        k = 35
    GSOSimDict = open_graphs(dataset, uimTrain, k, corr_mat, dict_to_use, method1, 'similarity')

    if graph2 == 'user':
        corr_mat = B
        dict_to_use = item_dict
        method2 = 'USERS'
    else:
        corr_mat = C
        dict_to_use = user_dict
        method2 = 'ITEMS'
    k = 40
    GSODisDict = open_graphs(dataset, uimTrain, k, corr_mat, dict_to_use, method2, 'dissimilarity')

    ##### TRAINING #####
    if not os.path.isdir(f'./{dataset}/experiments'):
        os.mkdir(f'./{dataset}/experiments')
    today = datetime.datetime.now().strftime("%m_%d_%H_%M")
    savePath = f'./{dataset}/experiments/{dataset}_BPR{today}_{graphType}{GSOSimOrder}{GSODisOrder}{str(alpha)[-1]}'
    os.mkdir(savePath)

    best_ndcg = 0
    limit = 20
    print('--------training processing-------')
    for epoch in np.arange(nEpochs):
        cnt = 1
        start_time = time.time()
        loader.dataset.ng_sample()
        cnt_test = 0
        earlyStop = 0
        for user, item_i, item_j in loader:
            if (cnt-1) % (np.round(len(loader)/1000)) == 0:
                print(f'cnt: {cnt}')
                uim_sim = prediction_with_parameter(uimTrain, GSOSimDict, h_sim, GSOSimOrder, method1)
                uim_dis = prediction_with_parameter(uimTrain, GSODisDict, h_dis, GSODisOrder, method2)

                uim_graph_predicted = (1-alpha) * uim_sim + alpha *uim_dis
                uim_graph_predicted = uim_graph_predicted + means

                ndcg = compute_ndcg(uim_graph_predicted, uimFull, testSignals, k=ndcg_items)
                if np.isnan(h_sim).any():
                    print('h_sim has NaNs! Re-initialize!')
                    h_sim = np.random.randn(GSOSimOrder)
                if np.isnan(h_dis).any():
                    print('h_dis has NaNs! Re-initialize!')
                    h_dis = np.random.randn(GSODisOrder)
                if ndcg > best_ndcg:
                    h_sim_best = np.copy(h_sim)
                    h_dis_best = np.copy(h_dis)
                    best_ndcg = ndcg
                    print('New best!')
                    print(f'NDCG -> {ndcg:.4f}')
                    print(f'h_sim_best = {h_sim_best}, h_dis_best = {h_dis_best}')
                    earlyStop = 0
                else:
                    earlyStop += 1
                cnt_test += 1
            user = int(user[0])
            item_i = int(item_i[0])
            item_j = int(item_j[0])

            x_ui, x_uj = BPRprediction(uimTrain, user, item_i, item_j, GSOSimDict, GSODisDict, GSOSimOrder, GSODisOrder, h_sim, h_dis, graphType)

            x_uij = x_ui - x_uj
            constant_term = np.exp(-x_uij) * sigmoid(x_uij)

            derivatives = []
            if graph1 == 'user':
                tmp1 = uimTrain[:, item_i]
                tmp2 = uimTrain[:, item_j]
            else:
                tmp = uimTrain[user, :]
            for l in range(GSOSimOrder):
                if graph1 == 'user':
                    tmp1 = GSOSimDict[item_i].dot(tmp1)
                    tmp2 = GSOSimDict[item_j].dot(tmp2)
                    derivative = tmp1[user] - tmp2[user]
                    derivatives.append(derivative)
                else:
                    tmp = GSOSimDict[user].dot(tmp)
                    derivative = tmp[item_i] - tmp[item_j]
                    derivatives.append(derivative)

            if graph2 == 'user':
                tmp1 = uimTrain[:, item_i]
                tmp2 = uimTrain[:, item_j]
            else:
                tmp = uimTrain[user, :]
            for l in range(GSODisOrder):
                if graph2 == 'user':
                    tmp1 = GSODisDict[item_i].dot(tmp1)
                    tmp2 = GSODisDict[item_j].dot(tmp2)
                    derivative = tmp1[user] - tmp2[user]
                    derivatives.append(derivative)
                else:
                    tmp = GSODisDict[user].dot(tmp)
                    derivative = tmp[item_i] - tmp[item_j]
                    derivatives.append(derivative)

            cnt_der = 0
            for l in range(GSOSimOrder):
                h_sim[l] = h_sim[l] + lr*(constant_term * derivatives[cnt_der] - 2*mu*alpha*(h_sim[l]))
                cnt_der += 1

            for l in range(GSODisOrder):
                h_dis[l] = h_dis[l] + lr*(constant_term * derivatives[cnt_der] - 2*mu*(1-alpha)*(h_dis[l]))
                cnt_der += 1

            cnt += 1

            if earlyStop == limit:
                print('Early stop!')
                break
        elapsed_time = time.time() - start_time
        str_print_train = f"epoch: {epoch + 1}\ttime: {datetime.timedelta(seconds=elapsed_time)}"
        print(str_print_train)

    uim_sim = prediction_with_parameter(uimTrain, GSOSimDict, h_sim_best, GSOSimOrder, method1)
    uim_dis = prediction_with_parameter(uimTrain, GSODisDict, h_dis_best, GSODisOrder, method2)

    uim_graph_predicted = (1 - alpha) * uim_sim + alpha * uim_dis
    uim_graph_predicted = uim_graph_predicted + means

    np.save(f'{savePath}/uim_graph_predicted.npy', uim_graph_predicted)


def main():
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('-method', action='store', dest='graphType', default='UI')
    parser.add_argument('-simOrd', action='store', dest='simOrder', default='1', type=int)
    parser.add_argument('-disOrd', action='store', dest='disOrder', default='1', type=int)
    parser.add_argument('-epochs', action='store', dest='nEpochs', default=1, type=int)
    parser.add_argument('-neg', action='store', dest='numNeg', default=4, type=int)
    parser.add_argument('-lr', action='store', dest='lr', default=0.001, type=float)
    parser.add_argument('-mu', action='store', dest='mu', default=0.01, type=float)
    parser.add_argument('-alpha', action='store', dest='alpha', default=0.1, type=float)
    parser.add_argument('-dataset', action='store', dest='dataset', default='ml100k')

    args = parser.parse_args()

    alpha = args.alpha

    runAll(args, alpha)


if __name__ == '__main__':
    main()

