import argparse
import ast
import time
import datetime
import torch
import torch.sparse
import os
import numpy as np

from architectures import GCN
from data_utils import BPRSampling, Movielens100kBPR

from torch.utils.data import DataLoader
import torch.nn.functional as F

from metrics import RMSE, compute_ndcg
from optimization_utils import open_graphs
from test_train_split import train_test_split
from utils import str2bool, check_base_files, get_graph_types, create_dicts


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
    GSOSimOrder = args.simOrd  # filter order for the similarity network
    GSODisOrder = args.disOrd  # filter order for the dissimilarity network
    features = ast.literal_eval(args.features)

    nEpochs = args.nEpochs  # how many times we cross the entire dataset
    lr = args.lr
    batchSize = args.batchSize
    mu = args.mu
    neg = args.numNeg

    print(f'BPR4GCN {graphType.upper()} -> running with {GSOSimOrder} sim, {GSODisOrder} dis, '
          f'feat={features}, {lr} learning rate, mu={mu}, batch={batchSize}, neg.samples={neg}, alpha={alpha}, '
          f'epochs={nEpochs}')

    if args.dataset == 'ml100k':
        n_users = 943
        n_items = 1682
    elif args.dataset == 'yahoo' or args.dataset == 'douban' or args.dataset == 'flixster':
        n_users = 3000
        n_items = 3000
    elif args.dataset == 'ml1M':
        n_users = 6040
        n_items = 3952

    ##### DATASET INITIALIZATION #####
    uimFull, uimTrain, train, testSignals = BPRSampling(dataset, n_users, n_items)
    means = np.mean(uimTrain, axis=0)
    uimTrainOriginal = np.copy(uimTrain)
    data = Movielens100kBPR(uimTrain, uimTrainOriginal, train, numNgTrain=neg, dataset=args.dataset)
    loader = DataLoader(data, batch_size=1, shuffle=True)

    ##### PRINTING OPTIONS #####
    nBatches = np.ceil(len(data) / batchSize).astype(np.int64)
    print(f'There are {len(data)} samples to examine -> {nBatches} batches')
    printBatch = np.ceil(nBatches/1000)  # after how many batches we want to print information on the screen

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

    model = GCN(graphType,
                uimTrain,
                GSOSimDict,
                GSODisDict,
                GSOSimOrder,
                GSODisOrder,
                features,
                alpha,
                application='BPR')

    model.getNumberParameters()

    ##### OPTIMIZER DEFINITION #####
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ##### TRAINING #####
    if not os.path.isdir(f'./{dataset}/experiments'):
        os.mkdir(f'./{dataset}/experiments')
    today = datetime.datetime.now().strftime("%m_%d_%H_%M")
    savePath = f'./{dataset}/experiments/{dataset}_BPR4GCN{today}_{graphType}{GSOSimOrder}{GSODisOrder}{batchSize}{str(alpha)[-1]}'
    os.mkdir(savePath)
    print(f'The predicted matrix, parameters and best model will be saved in {savePath}')

    toll = 1.e-4
    before = 0
    smooth_loss = 0
    best_ndcg = 0
    early_stop = 0
    limit = 15
    print('--------training processing-------')
    for epoch in np.arange(nEpochs):
        cnt = 1
        model.train()

        start_time = time.time()

        cnt_batch = 0

        loader.dataset.ng_sample()

        optimizer.zero_grad()
        batchLoss = 0
        cnt_limit = 0
        cumLoss = 0
        cntCumLoss = 0
        for user, item_i, item_j in loader:
            if early_stop == limit:
                print("Early stop!")
                break
            if (cnt-1) % np.round((len(loader)/1000)) == 0:
                model.eval()
                uim_graph_predicted = model.prediction(alpha=alpha)

                rmse = RMSE(uim_graph_predicted, testSignals)
                ndcg = compute_ndcg(uim_graph_predicted, uimFull, testSignals, k=ndcg_items)
                print(f'VAL: RMSE -> {rmse:.4f}, NDCG -> {ndcg:.4f}')

                if ndcg > best_ndcg:
                    torch.save(model.state_dict(), savePath+'/best_model.pt')
                    np.save(savePath+'/uim_graph_predicted.npy', uim_graph_predicted)
                    best_ndcg = ndcg
                    print('New best!')
                else:
                    early_stop += 1
                model.train()

            user = int(user[0])
            item_i = int(item_i[0])
            item_j = int(item_j[0])

            y_ui = model(user, item_i, BPR=True)
            y_uj = model(user, item_j, BPR=True)

            cnt += 1
            cnt_batch += 1
            logProb = -F.logsigmoid(y_ui-y_uj)
            loss = logProb + mu * model.l2regularizer(application='BPR')
            cumLoss += loss
            cntCumLoss += 1
            batchLoss += loss
            loss.backward()
            smooth_loss = smooth_loss * 0.99 + batchLoss * 0.01
            if cnt_batch == batchSize:
                optimizer.step()
                optimizer.zero_grad()
                if np.abs(batchLoss.item() - before) < toll:
                    cnt_limit += 1
                else:
                    cnt_limit = 0
                before = batchLoss.item()

                if cnt_limit == limit:
                    print('Early stop!')
                    break

                cnt_batch = 0
                batchLoss = 0

            if cnt/batchSize % printBatch == 0 and cnt != 1:
                print(f'TRAIN: batch #{np.round(cnt/batchSize)} --> loss = {cumLoss.item()/cntCumLoss:.4f}')
                cumLoss = 0
                cntCumLoss = 0


        elapsed_time = time.time() - start_time
        str_print_train = f"epoch: {epoch + 1}\ttime: {datetime.timedelta(seconds=elapsed_time)}"
        print(str_print_train)

    best_model = GCN(graphType,
                    uimTrain,
                    GSOSimDict,
                    GSODisDict,
                    GSOSimOrder,
                    GSODisOrder,
                    features,
                    alpha,
                    application='BPR')
    best_model.load_state_dict(torch.load(savePath+'/best_model.pt'))
    uim_graph_predicted = best_model.prediction(alpha=alpha)
    np.save(f'{savePath}/best_val.npy', uim_graph_predicted)
    print('Complete!')

    file = open(f'{savePath}/parameters.txt', 'w')

    file.write(f'graphType: {graphType}\n')
    file.write(f'simOrder: {GSOSimOrder}\n')
    file.write(f'disOrder: {GSODisOrder}\n')
    file.write(f'nEpochs: {nEpochs}\n')
    file.write(f'batchSize: {batchSize}\n')
    file.write(f'Num negative samples per user: {neg}\n')
    file.write(f'lr: {lr}\n')
    file.write(f'mu: {mu}\n')
    file.write(f'alpha: {alpha}\n')

    file.write(f'alpha: {alpha} -> RMSE -> {rmse:.4f}, NDCG - {ndcg:.4f}\n')

    file.close()


def main():
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('-method', action='store', dest='graphType', default='UI')
    parser.add_argument('-simOrd', action='store', dest='simOrd', default='1', type=int)
    parser.add_argument('-disOrd', action='store', dest='disOrd', default='1', type=int)
    parser.add_argument('-epochs', action='store', dest='nEpochs', default=1, type=int)
    parser.add_argument('-batch', action='store', dest='batchSize', default=1, type=int)
    parser.add_argument('-neg', action='store', dest='numNeg', default=4, type=int)
    parser.add_argument('-lr', action='store', dest='lr', default=0.001, type=float)
    parser.add_argument('-mu', action='store', dest='mu', default=0.01, type=float)
    parser.add_argument('-alpha', action='store', dest='alpha', default=0.1, type=float)
    parser.add_argument('-feat', action='store', dest='features', default='[1,2]')
    parser.add_argument('-dataset', action='store', dest='dataset', default='ml100k')

    args = parser.parse_args()

    alpha = args.alpha

    runAll(args, alpha)


if __name__ == '__main__':
    main()