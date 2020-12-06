import torch
from torch import nn
import numpy as np


class GraphFilteringLayer(nn.Module):
    def __init__(self,
                 GSODict,
                 order,
                 outFeat):
        super(GraphFilteringLayer, self).__init__()
        self.GSODict = GSODict
        self.filterCoeff = nn.Parameter(torch.randn(outFeat, order)*0.1, requires_grad=True)

    def forward(self, x, ind):
        GSOSpec = self.toSparse(self.GSODict[ind])
        numOutFeatures = self.filterCoeff.shape[0]
        numInFeatures = x.shape[1]
        nNodes = x.shape[0]
        numFilterTaps = self.filterCoeff.shape[1]
        y = torch.zeros(nNodes, numOutFeatures, numInFeatures)
        for inFeat in np.arange(numInFeatures):
            GSOTimesVectors = torch.sparse.mm(GSOSpec, x[:, inFeat].view(-1, 1))
            for outFeat in np.arange(numOutFeatures):
                tmp = torch.zeros(nNodes, 1)
                for i in np.arange(numFilterTaps):
                    tmp = tmp + self.filterCoeff[outFeat][i] * GSOTimesVectors
                    if i == numFilterTaps - 1:
                        continue
                    GSOTimesVectors = torch.sparse.mm(GSOSpec, GSOTimesVectors)
                y[:, outFeat, inFeat] = tmp.reshape(-1)
        y = torch.sum(y, dim=2)
        return y

    def toSparse(self, mat):
        sparse_mat = mat.tocoo()

        values = sparse_mat.data
        indices = np.vstack((sparse_mat.row, sparse_mat.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = sparse_mat.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))