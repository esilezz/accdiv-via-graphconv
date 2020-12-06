import numpy as np
import torch.utils.data as data
from scipy.io import loadmat


from utils import get_available_pairs

GLOBAL_SEED = 1


def trainValTestSplit(dataset,
                      nUsers,
                      nItems,
                      graphType,
                      trainSize=80,
                      valSize=0.001):
    np.random.seed(GLOBAL_SEED)
    if dataset == 'ml100k':
        wrap = loadmat(f'./{dataset}/ml100k_matrices.mat')
        train_set = wrap['train']
        train_set[:, 0] = train_set[:, 0] - 1
        train_set[:, 1] = train_set[:, 1] - 1
        test_set = wrap['test']
        test_set[:, 0] = test_set[:, 0] - 1
        test_set[:, 1] = test_set[:, 1] - 1
        pairs = np.concatenate((train_set, test_set))
    elif dataset == 'douban':
        wrap = loadmat('./data/douban_matrices.mat')
        train_set = wrap['train']
        train_set[:, 0] = train_set[:, 0] - 1
        train_set[:, 1] = train_set[:, 1] - 1
        train_set = train_set.astype(int)
        test_set = wrap['test']
        test_set[:, 0] = test_set[:, 0] - 1
        test_set[:, 1] = test_set[:, 1] - 1
        test_set = test_set.astype(int)
        pairs = np.concatenate((train_set, test_set))
    elif dataset == 'flixster':
        wrap = loadmat('./data/flixster_matrices.mat')
        train_set = wrap['train']
        train_set[:, 0] = train_set[:, 0] - 1
        train_set[:, 1] = train_set[:, 1] - 1
        train_set = train_set.astype(int)
        test_set = wrap['test']
        test_set[:, 0] = test_set[:, 0] - 1
        test_set[:, 1] = test_set[:, 1] - 1
        test_set = test_set.astype(int)
        pairs = np.concatenate((train_set, test_set))
    elif dataset == 'ml1M':
        train_set = np.load('./data/ml1M_train_split9.npy')
        test_set = np.load('./data/ml1M_test_split9.npy')
    pairs = np.concatenate((train_set, test_set))

    uimFull = np.zeros(shape=(nUsers, nItems))
    uimFull[pairs[:, 0], pairs[:, 1]] = pairs[:, 2]

    totalSamples = len(pairs)
    nTrain = int(np.round((totalSamples * trainSize) / 100))
    nVal = int(np.round(nTrain * valSize))

    trainUsers = train_set[:, 0]
    trainItems = train_set[:, 1]
    trainRatings = train_set[:, 2]

    uimTrain = np.zeros(shape=uimFull.shape)
    uimTrain[trainUsers, trainItems] = trainRatings

    train = train_set
    np.random.shuffle(train)

    val = train[:nVal]
    train = train[nVal:]

    train = train.astype(int)
    val = val.astype(int)

    test = test_set
    test = test.astype(int)
    return uimFull, uimTrain, train, val, test


def BPRSampling(dataset,
                nUsers,
                nItems):
    np.random.seed(GLOBAL_SEED)

    if dataset == 'ml100k':
        wrap = loadmat(f'./{dataset}/ml100k_matrices.mat')
        train_set = wrap['train']
        train_set[:, 0] = train_set[:, 0] - 1
        train_set[:, 1] = train_set[:, 1] - 1
        test_set = wrap['test']
        test_set[:, 0] = test_set[:, 0] - 1
        test_set[:, 1] = test_set[:, 1] - 1
    elif dataset == 'douban':
        wrap = loadmat(f'./{dataset}/data/douban_matrices.mat')
        train_set = wrap['train']
        train_set[:, 0] = train_set[:, 0] - 1
        train_set[:, 1] = train_set[:, 1] - 1
        train_set = train_set.astype(int)
        test_set = wrap['test']
        test_set[:, 0] = test_set[:, 0] - 1
        test_set[:, 1] = test_set[:, 1] - 1
        test_set = test_set.astype(int)
    elif dataset == 'flixster':
        wrap = loadmat(f'./{dataset}/data/flixster_matrices.mat')
        train_set = wrap['train']
        train_set[:, 0] = train_set[:, 0] - 1
        train_set[:, 1] = train_set[:, 1] - 1
        train_set = train_set.astype(int)
        test_set = wrap['test']
        test_set[:, 0] = test_set[:, 0] - 1
        test_set[:, 1] = test_set[:, 1] - 1
        test_set = test_set.astype(int)
    elif dataset == 'ml1M':
        train_set = np.load(f'./{dataset}/data/ml1M_train_split9.npy')
        test_set = np.load(f'./{dataset}/data/ml1M_test_split9.npy')

    pairs = np.concatenate((train_set, test_set))

    uimFull = np.zeros(shape=(nUsers, nItems))
    uimFull[pairs[:, 0], pairs[:, 1]] = pairs[:, 2]

    trainUsers = train_set[:, 0]
    trainItems = train_set[:, 1]
    trainRatings = train_set[:, 2]
    uimTrain = np.zeros(shape=uimFull.shape)
    uimTrain[trainUsers, trainItems] = trainRatings

    trainData = train_set
    trainData = trainData.astype(int)

    test = test_set
    test = test.astype(int)
    return uimFull, uimTrain, trainData, test


class Movielens100kGCN(data.Dataset):
    def __init__(self, samples, graphType):
        self.graphType = graphType
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        user = self.samples[i, 0]
        item = self.samples[i, 1]
        rating = self.samples[i, 2]
        return user, item, rating


class Movielens100kBPR(data.Dataset):
    def __init__(self, uimTrain, uimTrainOriginal, trainData, numNgTrain, dataset):
        np.random.seed(GLOBAL_SEED)
        self.uimTrain = uimTrain
        self.uimTrainOriginal = uimTrainOriginal
        self.numUser = uimTrain.shape[0]
        self.numItem = uimTrain.shape[1]
        self.trainData = trainData
        self.numNgTrain = numNgTrain
        self.dataset = dataset

    def testNeg(self, testSamples, numNgTest):
        user = []
        item = []
        for cnt in np.arange(len(testSamples)):
            s = testSamples[cnt]
            tmp_item = []
            u = s[0]
            i = s[1]
            tmp_item.append(i)
            neglist = set()
            neglist.add(i)
            for t in range(numNgTest):
                j = np.random.randint(self.numItem)
                while self.uimTrain[u, j] != 0 or j in neglist:
                    j = np.random.randint(self.numItem)
                neglist.add(j)
                tmp_item.append(j)
            user.append(u)
            item.append(tmp_item)
        self.testUsers = np.array(user)
        self.testItems = np.array(item)

    def ng_sample(self):
        self.triplets = []
        for x in self.trainData:
            u, i, r = x[0], x[1], x[2]
            for t in range(self.numNgTrain):
                j = np.random.randint(self.numItem)
                while self.uimTrainOriginal[u, j] > self.uimTrainOriginal[u, i]:
                        j = np.random.randint(self.numItem)
                self.triplets.append([u, i, j])

    def __len__(self):
        return self.numNgTrain * len(self.trainData)

    def __getitem__(self, idx):
        user = self.triplets[idx][0]
        item_i = self.triplets[idx][1]
        item_j = self.triplets[idx][2]
        return user, item_i, item_j

