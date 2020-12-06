import sys
import torch
from torch import nn
from layers import GraphFilteringLayer
import numpy as np

from utils import get_graph_types


class GCN(nn.Module):
    def __init__(self,
                 graphType,
                 uimTrain,
                 GSOSimDict,
                 GSODisDict,
                 simOrder,
                 disOrder,
                 features,
                 alpha,
                 application='GCN'):
        super(GCN, self).__init__()
        numNodesSim = GSOSimDict[0].shape[0]
        numNodesDis = GSODisDict[0].shape[0]

        self.application = application
        self.graphType = graphType
        self.originalUIM = uimTrain
        if application == 'GCN':
            uimMeanCentered = uimTrain - np.mean(uimTrain, axis=0)
        else:
            uimMeanCentered = uimTrain

        self.uimTrain = torch.Tensor(uimMeanCentered)
        self.convSim1 = GraphFilteringLayer(GSOSimDict, simOrder, features[1])
        self.nonLinearity = nn.LeakyReLU()
        self.finalSimFC = nn.Parameter(torch.randn(features[1], numNodesSim), requires_grad=True)
        self.sharedSimFC = nn.Parameter(torch.randn(numNodesSim, 1), requires_grad=True)

        self.alpha = alpha
        self.convDis1 = GraphFilteringLayer(GSODisDict, disOrder, features[1])
        self.finalDisFC = nn.Parameter(torch.randn(features[1], numNodesDis), requires_grad=True)
        self.sharedDisFC = nn.Parameter(torch.randn(numNodesDis, 1), requires_grad=True)

    def toSparse(self, mat):
        sparse_mat = mat.tocoo()

        values = sparse_mat.data
        indices = np.vstack((sparse_mat.row, sparse_mat.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = sparse_mat.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def forward(self, user, item, BPR=False):
        if self.graphType == 'UU':
            xSim = self.uimTrain[:, item]
            indSim = item
            toSelectSim = user
            xDis = self.uimTrain[:, item]
            indDis = item
            toSelectDis = user
        elif self.graphType == 'II':
            xSim = self.uimTrain[user, :]
            indSim = user
            toSelectSim = item
            xDis = self.uimTrain[user, :]
            indDis = user
            toSelectDis = item
        elif self.graphType == 'UI':
            xSim = self.uimTrain[:, item]
            indSim = item
            toSelectSim = user
            xDis = self.uimTrain[user, :]
            indDis = user
            toSelectDis = item
        elif self.graphType == 'IU':
            xSim = self.uimTrain[user, :]
            indSim = user
            toSelectSim = item
            xDis = self.uimTrain[:, item]
            indDis = item
            toSelectDis = user
        xSim[toSelectSim] = 0
        xDis[toSelectDis] = 0
        xSim = xSim.reshape(-1, 1)
        xDis = xDis.reshape(-1, 1)
        x = self.completeForward(xSim, indSim, toSelectSim, xDis, indDis, toSelectDis, BPR)
        return x

    def completeForward(self, xSim, indSim, toSelectSim, xDis, indDis, toSelectDis, BPR=False):
        xSim = self.convSim1(xSim, indSim)
        xSim = self.nonLinearity(xSim)
        xDis = self.convDis1(xDis, indDis)
        xDis = self.nonLinearity(xDis)
        xSim = torch.mm(xSim, self.finalSimFC).mm(self.sharedSimFC)
        xDis = torch.mm(xDis, self.finalDisFC).mm(self.sharedDisFC)

        if BPR:
            x = xSim[toSelectSim] + xDis[toSelectDis]
        else:
            x = (1-self.alpha) * xSim[toSelectSim] + self.alpha * xDis[toSelectDis]
        return x

    def prediction(self, alpha=0.1, verbose=False):
        if verbose:
            print('Full prediction')
        with torch.no_grad():
            uimSim = torch.zeros(self.uimTrain.shape)
            uimDis = torch.zeros(self.uimTrain.shape)
            numUsers = self.uimTrain.shape[0]
            numItems = self.uimTrain.shape[1]
            graph1, graphs2 = get_graph_types(self.graphType)

            if graph1 == 'user':
                for item in np.arange(numItems):
                    if verbose:
                        sys.stdout.write("\rPredicting %d out of %d" % (item, numItems))
                        sys.stdout.flush()
                    x = self.uimTrain[:, item].reshape(-1, 1)
                    x = self.convSim1(x, item)
                    x = self.nonLinearity(x)
                    x = torch.mm(x, self.finalSimFC).mm(self.sharedSimFC)
                    uimSim[:, item] = x.reshape(-1)
            else:
                for user in np.arange(numUsers):
                    if verbose:
                        sys.stdout.write("\rPredicting %d out of %d" % (user, numUsers))
                        sys.stdout.flush()
                    x = self.uimTrain[user, :].reshape(-1, 1)
                    x = self.convSim1(x, user)
                    x = self.nonLinearity(x)
                    x = torch.mm(x, self.finalSimFC).mm(self.sharedSimFC)
                    uimSim[user, :] = x.reshape(-1)
            if verbose:
                print()
            if graphs2 == 'user':
                for item in np.arange(numItems):
                    if verbose:
                        sys.stdout.write("\rPredicting %d out of %d" % (item, numItems))
                        sys.stdout.flush()
                    x = self.uimTrain[:, item].reshape(-1, 1)
                    x = self.convDis1(x, item)
                    x = self.nonLinearity(x)
                    x = torch.mm(x, self.finalDisFC).mm(self.sharedDisFC)
                    uimDis[:, item] = x.reshape(-1)
            else:
                for user in np.arange(numUsers):
                    if verbose:
                        sys.stdout.write("\rPredicting %d out of %d" % (user, numUsers))
                        sys.stdout.flush()
                    x = self.uimTrain[user, :].reshape(-1, 1)
                    x = self.convDis1(x, user)
                    x = self.nonLinearity(x)
                    x = torch.mm(x, self.finalDisFC).mm(self.sharedDisFC)
                    uimDis[user, :] = x.reshape(-1)
            if verbose:
                print()
            uimPredicted = (1-alpha)*uimSim + alpha*uimDis

            uimPredicted = uimPredicted.numpy()

            if self.application == 'GCN':
                uimPredicted = uimPredicted + np.mean(self.originalUIM, axis=0)

        return uimPredicted

    def getNumberParameters(self, printNames=False):
        tot = 0
        for name, par in self.named_parameters():
            if par.requires_grad:
                if printNames:
                    print(f'{name}: {par.shape}')
                if len(par.shape) > 1:
                    tot += par.shape[0]*par.shape[1]
                else:
                    tot += par.shape[0]
        print(f'The model created has {tot} parameters to learn')

    def l2regularizer(self, application='GCN'):
        if application == 'GCN':
            simNorm = self.getParameters('similar')
            simNorm = simNorm / (1 - self.alpha)
            disNorm = self.getParameters('dissimilar')
            disNorm = disNorm / self.alpha

            reg = 0.5 * (simNorm + disNorm)
        else:
            simNorm = self.getParameters('similar')
            simNorm = simNorm * self.alpha
            disNorm = self.getParameters('dissimilar')
            disNorm = disNorm * (1-self.alpha)

            reg = simNorm + disNorm
        return reg

    def getParameters(self, mode):
        normPar = 0
        if mode == 'similar':
            normPar += torch.norm(self.convSim1.filterCoeff, 2)
            normPar += torch.norm(self.sharedSimFC, 2)
            normPar += torch.norm(self.finalSimFC, 2)
        else:
            normPar += torch.norm(self.convDis1.filterCoeff, 2)
            normPar += torch.norm(self.sharedDisFC, 2)
            normPar += torch.norm(self.finalDisFC, 2)
        return normPar
