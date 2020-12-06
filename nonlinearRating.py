import argparse
import time
import datetime
import torch
import torch.sparse
import torch.nn as nn

import os
import numpy as np

from architectures import GCN
from data_utils import Movielens100kGCN, trainValTestSplit

from torch.utils.data import DataLoader
import ast

from metrics import RMSE, compute_ndcg
from optimization_utils import open_graphs
from test_train_split import train_test_split
from utils import get_graph_types, str2bool, create_dicts, check_base_files, float2tensor, constraint_in_range


def runAll(args, alpha, dataset):
    if not check_base_files(dataset):
        train_test_split(dataset)

    path = f'./{dataset}/matrices/'
    B = np.load(path+'B.npy')
    C = np.load(path+'C.npy')
    train = np.load(path+'train_split.npy')

    ##### PARAMETERS DECLARATION #####
    graphType = args.graphType

    ### 2 LAYERS ARCHITECTURE ###
    GSOSimOrder = args.simOrder  # filter order for the similarity network per layer
    GSODisOrder = args.disOrder  # filter order for the dissimilarity network

    simFeatures = ast.literal_eval(args.features)  # number of features per layer
    if len(simFeatures) > 2:
        print(f'ERROR - it is possible to define only architectures with two layers (you chose {simFeatures}, but you'
              f' can only define [{simFeatures[0]},{simFeatures[1]}])')
        exit(-1)
    nEpochs = args.nEpochs  # how many times we cross the entire dataset
    lr = args.lr
    batchSize = args.batchSize
    mu = args.mu

    print(f'{graphType.upper()} -> running with {simFeatures}, {GSOSimOrder} sim. degree, {GSODisOrder} dis. degree,'
          f' mu={mu}, {lr} learning rate, batch={batchSize}, alpha={alpha}')

    if dataset == 'ml100k':
        NUM_USERS = 943
        NUM_ITEMS = 1682
    elif dataset == 'yahoo' or dataset == 'douban' or dataset == 'flixster':
        NUM_ITEMS = 3000
        NUM_USERS = 3000
    elif args.dataset == 'ml1M':
        NUM_USERS = 6040
        NUM_ITEMS = 3952

    ##### DATASET INITIALIZATION #####
    uimFull, uimTrain, trainSignals, valSignals, testSignals = trainValTestSplit(dataset,
                                                                                 NUM_USERS,
                                                                                 NUM_ITEMS,
                                                                                 graphType)
    trainData = Movielens100kGCN(trainSignals, graphType)
    trainLoader = DataLoader(trainData, batch_size=1, shuffle=True)

    valData = Movielens100kGCN(valSignals, graphType)
    valLoader = DataLoader(valData, batch_size=1, shuffle=True)

    nBatches = np.ceil(len(trainData) / batchSize).astype(np.int64)
    print(f'There are {nEpochs} epochs and {nBatches} batches per epoch')

    validationInterval = np.floor(nBatches/100)  # after how many batches we want to plot the validation curve
    printBatch = np.floor(nBatches/100)  # after how many batches we want to print information on the screen

    ##### MODEL CREATION #####
    print('Model initialization...')
    graph1, graph2 = get_graph_types(graphType)
    user_dict, item_dict = create_dicts(train)

    if graph1 == 'user':
        corr_mat = B
        dict_to_use = item_dict
        method = 'USERS'
        k = 30
    else:
        corr_mat = C
        dict_to_use = user_dict
        method = 'ITEMS'
        k = 35
    GSOSimDict = open_graphs(dataset, uimTrain, k, corr_mat, dict_to_use, method, 'similarity')

    if graph2 == 'user':
        corr_mat = B
        dict_to_use = item_dict
        method = 'USERS'
    else:
        corr_mat = C
        dict_to_use = user_dict
        method = 'ITEMS'
    k = 40
    GSODisDict = open_graphs(dataset, uimTrain, k, corr_mat, dict_to_use, method, 'dissimilarity')

    model = GCN(graphType,
                uimTrain,
                GSOSimDict,
                GSODisDict,
                GSOSimOrder,
                GSODisOrder,
                simFeatures,
                alpha)
    model.getNumberParameters()

    ##### OPTIMIZER AND LOSS FUNCTION DEFINITION #####
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ##### VARIABLES DECLARATION #####
    means = np.mean(uimTrain, axis=0)
    if not os.path.isdir(f'./{dataset}/experiments'):
        os.mkdir(f'./{dataset}/experiments')

    today = datetime.datetime.now().strftime("%m_%d_%H_%M")
    savePath = f'./{dataset}/experiments/{dataset}_GCN{today}_{graphType}{GSOSimOrder}{GSODisOrder}{batchSize}{str(alpha)[-1]}'
    os.mkdir(savePath)

    toll = 1.e-4
    before = 0
    limit = 50

    cnt_batch = 0
    best_rmse = 100
    best = 100000

    ##### TRAINING #####
    print('--------training processing-------')
    for epoch in np.arange(nEpochs):
        cnt = 1
        model.train()

        start_time = time.time()
        optimizer.zero_grad()
        batchLoss = 0
        for user, item, rating in trainLoader:
            if (cnt-1) % (np.round(len(trainLoader)/40)) == 0:
                model.eval()
                uim_graph_predicted = model.prediction(alpha=alpha, verbose=True)
                rmse = RMSE(uim_graph_predicted, testSignals)
                print(f'RMSE -> {rmse:.4f}')
                if rmse < best_rmse:
                    print('New best model!')
                    torch.save(model.state_dict(), savePath + '/best_model.pt')
                    np.save(f"{savePath}/uim_graph_predicted.npy", uim_graph_predicted)
                    best_rmse = rmse
                model.train()
            user = int(user[0])
            item = int(item[0])
            rating = int(rating[0])
            yHat = model(user, item)

            cnt_batch += 1

            rating = rating - means[item]
            reg = mu * model.l2regularizer(application='GCN')
            loss = criterion(yHat[0], float2tensor(rating)) + reg
            loss.backward()
            batchLoss += loss

            if cnt_batch == batchSize:
                optimizer.step()
                optimizer.zero_grad()
                cnt_batch = 0
                lastBatchLoss = batchLoss.clone()
                batchLoss = 0
                if np.abs(lastBatchLoss.item() - before) < toll:
                    cnt_limit += 1
                else:
                    cnt_limit = 0
                before = lastBatchLoss.item()

                if cnt_limit == limit:
                    break

            if cnt/batchSize % printBatch == 0:
                print('epoch %d batch #%d --> loss = %.4f' % (epoch + 1,
                                                              np.round(cnt/batchSize),
                                                              lastBatchLoss.item()/batchSize))
            if cnt/batchSize % validationInterval == 0:
                model.eval()
                print("Validation...", end=' ', flush=True)
                cnt_val = 0
                lossVal = 0
                reg = mu*model.l2regularizer()
                for user, item, rating in valLoader:
                    with torch.no_grad():
                        user = int(user[0])
                        item = int(item[0])
                        rating = int(rating[0])

                        yHatVal = model(user, item)
                        rating = rating - means[item]
                        lossVal += criterion(yHatVal[0], float2tensor(rating)) + reg
                        cnt_val += 1
                normValLoss = lossVal.item() / len(valLoader)
                if normValLoss < best:
                    torch.save(model.state_dict(), savePath + '/best_VAL_model.pt')
                    best = normValLoss
                print(f'VAL -> LOSS {normValLoss:.4f}')
                model.train()
            cnt = cnt + 1

        elapsed_time = time.time() - start_time
        str_print_train = f"epoch: {epoch+1}\ttime: {datetime.timedelta(seconds=elapsed_time)}"
        print(str_print_train)

        best_VAL_model = GCN(graphType,
                uimTrain,
                GSOSimDict,
                GSODisDict,
                GSOSimOrder,
                GSODisOrder,
                simFeatures,
                alpha)
        best_VAL_model.load_state_dict(torch.load(savePath + '/best_VAL_model.pt'))
        uim_graph_predicted = best_VAL_model.prediction(alpha=alpha)
        uim_graph_predicted = constraint_in_range(uim_graph_predicted)
        np.save(f"{savePath}/uim_graph_predicted_bestVAL.npy", uim_graph_predicted)

        best_model = GCN(graphType,
                uimTrain,
                GSOSimDict,
                GSODisDict,
                GSOSimOrder,
                GSODisOrder,
                simFeatures,
                alpha)
        best_model.load_state_dict(torch.load(savePath + '/best_model.pt'))
        uim_graph_predicted = best_model.prediction(alpha=alpha)
        uim_graph_predicted = constraint_in_range(uim_graph_predicted)
        np.save(f"{savePath}/uim_graph_predicted_best_iteration.npy", uim_graph_predicted)

    file = open(f'{savePath}/parameters{today}.txt', 'w')

    file.write(f'graphType: {args.graphType}\n')
    file.write(f'simOrder: {args.simOrder}\n')
    file.write(f'disOrder: {args.disOrder}\n')
    file.write(f'features: {args.features}\n')
    file.write(f'nEpochs: {args.nEpochs}\n')
    file.write(f'lr: {args.lr}\n')
    file.write(f'mu: {args.mu}\n')
    file.write(f'loss: {args.lossFunc}\n')
    file.write('\n')
    file.close()


def main():
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('-method', action='store', dest='graphType', default='UI')
    parser.add_argument('-simOrd', action='store', dest='simOrder', default='1', type=int)
    parser.add_argument('-disOrd', action='store', dest='disOrder', default='1', type=int)
    parser.add_argument('-feat', action='store', dest='features', default='[1,2]')
    parser.add_argument('-epochs', action='store', dest='nEpochs', default=8, type=int)
    parser.add_argument('-batch', action='store', dest='batchSize', default=256, type=int)
    parser.add_argument('-lr', action='store', dest='lr', default=0.001, type=float)
    parser.add_argument('-mu', action='store', dest='mu', default=0.1, type=float)
    parser.add_argument('-alpha', action='store', dest='alpha', default=0.1, type=float)
    parser.add_argument('-dataset', action='store', dest='dataset', default='ml100k')

    args = parser.parse_args()

    alpha = args.alpha

    runAll(args, alpha, args.dataset)


if __name__ == '__main__':
    main()
