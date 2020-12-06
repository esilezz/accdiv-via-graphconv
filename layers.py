import torch
from torch import nn
import numpy as np


def toSparse(mat):
    sparse_mat = mat.tocoo()

    values = sparse_mat.data
    indices = np.vstack((sparse_mat.row, sparse_mat.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sparse_mat.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


class GFL(nn.Module):
    def __init__(self, GSO, filter_order, outFeat):
        super(GFL, self).__init__()
        self.GSO = GSO
        self.filterCoeff = nn.Parameter(torch.randn(outFeat, filter_order) * 0.1, requires_grad=True)

    def forward(self, x):
        numOutFeatures = self.filterCoeff.shape[0]
        numInFeatures = x.shape[1]
        nNodes = x.shape[0]
        numFilterTaps = self.filterCoeff.shape[1]
        y = torch.zeros(nNodes, numOutFeatures, numInFeatures)
        for inFeat in np.arange(numInFeatures):
            GSOTimesVectors = torch.sparse.mm(self.GSO, x[:, inFeat].view(-1, 1))
            for outFeat in np.arange(numOutFeatures):
                tmp = torch.zeros(nNodes, 1)
                for i in np.arange(numFilterTaps):
                    tmp = tmp + self.filterCoeff[outFeat][i] * GSOTimesVectors
                    if i == numFilterTaps - 1:
                        continue
                    GSOTimesVectors = torch.sparse.mm(self.GSO, GSOTimesVectors)
                y[:, outFeat, inFeat] = tmp.reshape(-1)
        y = torch.sum(y, dim=2)
        return y


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


class LocalLinear(nn.Module):
    def __init__(self, numNodes, outFeatures, inFeatures):
        super(LocalLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(numNodes, inFeatures, outFeatures)*0.1, requires_grad=True)
        self.bias = nn.Parameter(torch.randn(numNodes, outFeatures)*0.1, requires_grad=True)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.matmul(x, self.weight).squeeze(2)+self.bias
        return x


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        k = self.kernel_size
        d = self.stride
        x = x.unfold(2, k, d)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


def LSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f.

    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert S.shape[0] == E
    N = S.shape[1]
    assert S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in F x E x K x G
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    x = x.reshape([B, 1, G, N])
    S = S.reshape([1, E, N, N])
    z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1) # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1,K):
        x = torch.matmul(x, S) # B x E x G x N
        xS = x.reshape([B, E, 1, G, N]) # B x E x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E*K*G]),
                     h.reshape([F, E*K*G]).permute(1, 0)).permute(0, 2, 1)
    # And permute againt to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y


class GraphFilter(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter

    Initialization:

        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, GSOdict, G, F, K, E = 1, bias = True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.GSODict = GSOdict
        self.N = GSOdict[0].shape[0]
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.Parameter(torch.randn(F, E, K, G))
        if bias:
            self.bias = nn.Parameter(torch.randn(F, 1))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, ind):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        x = x.reshape(1, x.shape[1], x.shape[0])
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # Compute the filter output
        S = toSparse(self.GSODict[ind]).to_dense()
        u = LSIGF(self.weight, S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString




class BayesianGFL(nn.Module):
    def __init__(self,
                 useCuda,
                 numUsers,
                 GSODict,
                 order,
                 inFeat,
                 outFeat):
        super(BayesianGFL, self).__init__()

        self.GSODict = GSODict
        self.numUsers = numUsers
        self.filterCoeff = nn.Parameter(torch.randn(outFeat, order))

    def forward(self, x, ind):
        GSOSpec = self.toSparse(self.GSODict[ind])
        numOutFeatures = self.filterCoeff.shape[0]
        numNodes = x.shape[0]
        numInFeatures = x.shape[1]
        numFilterTaps = self.filterCoeff.shape[1]
        y = torch.zeros(numNodes, numOutFeatures, numInFeatures)
        for inFeat in np.arange(numInFeatures):
            for outFeat in np.arange(numOutFeatures):
                tmp = torch.zeros(numNodes, 1)
                GSOPow = GSOSpec
                for i in np.arange(numFilterTaps):
                    GSOTimesVectors = torch.sparse.mm(GSOPow, x[:, inFeat].view(-1, 1))
                    tmp = tmp + self.filterCoeff[outFeat][i] * GSOTimesVectors
                    if i == numFilterTaps - 1:
                        continue
                    GSOPow = GSOPow ** (i + 1)
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


class MaxPoolingLayer(nn.Module):
    def __init__(self):
        super(MaxPoolingLayer, self).__init__()
        pass

    def forward(self, x):
        tmp = torch.zeros(x.shape)
        for i in np.arange(x.shape[0]):
            maxInd = torch.argmax(x[i])
            tmp[i, maxInd] = x[i, maxInd]
        return tmp