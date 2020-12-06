import warnings
import pandas as pd
from scipy import sparse
from scipy.io import loadmat
from metrics import LF_model
from utils import *
import numpy as np
np.set_printoptions(suppress=True)

warnings.filterwarnings('ignore')
NUM_FACTORS = 7


def train_test_split(dataset):
    if not os.path.isdir(f'./{dataset}'):
        os.mkdir(f'./{dataset}')

    if not os.path.isdir(f'./{dataset}/experiments'):
        os.mkdir(f'./{dataset}/experiments')

    path = f'./{dataset}/matrices'
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        return

    if dataset == 'ml100k':
        TOT_USERS = 943
        TOT_ITEMS = 1682
        COMMON_MIN = 2
    elif dataset == 'ml1m':
        TOT_USERS = 6040
        TOT_ITEMS = 3952
        COMMON_MIN = 2
    elif dataset == 'flixster' or dataset == 'douban':
        TOT_USERS = 3000
        TOT_ITEMS = 3000
        COMMON_MIN = 1

    if dataset == 'flixster':
        train = np.load('./flixster/data/train_pairs.npy')
        test = np.load('./flixster/data/test_pairs.npy')
    elif dataset == 'ml100k':
        wrap = loadmat('./ml100k/ml100k_matrices.mat')
        train = wrap['train']
        train[:, 0] = train[:, 0] - 1
        train[:, 1] = train[:, 1] - 1
        test = wrap['test']
        test[:, 0] = test[:, 0] - 1
        test[:, 1] = test[:, 1] - 1
    elif dataset == 'douban':
        wrap = loadmat('./douban/data/douban_matrices.mat')
        train = wrap['train']
        train[:, 0] = train[:, 0] - 1
        train[:, 1] = train[:, 1] - 1
        test = wrap['test']
        test[:, 0] = test[:, 0] - 1
        test[:, 1] = test[:, 1] - 1
    elif dataset == 'ml1m':
        PATH_DATA = './ml1m/data/ratings.dat'
        pairs = get_available_pairs(PATH_DATA, dataset)
        np.random.shuffle(pairs)
        len_train = int(np.ceil(len(pairs)*90/100))
        train = np.array(pairs[:len_train])
        test = np.array(pairs[len_train+1:-1])

    train = train.astype(int)
    test = test.astype(int)
    uim = create_uim(TOT_USERS, TOT_ITEMS, train)
    uim_full = create_uim_full(TOT_USERS, TOT_ITEMS, train, test)

    is_rated = uim > 0
    is_rated = is_rated * 1
    is_rated = sparse.csr_matrix(is_rated)

    print('Computing user similarities')
    df = pd.DataFrame(uim.transpose())
    sims = df.corr()
    sims = sims.to_numpy()
    sims = np.nan_to_num(sims)
    tmp = is_rated.dot(is_rated.transpose())
    min_check = tmp < COMMON_MIN
    min_check = min_check * 1
    min_check = min_check.toarray()
    min_check = min_check - np.diag(np.diag(min_check))
    sims[min_check > 0] = 0
    B = sims - np.diag(np.diag(sims))

    print('Computing item similarities')
    df = pd.DataFrame(uim)
    sims = df.corr()
    sims = sims.to_numpy()
    sims = np.nan_to_num(sims)
    tmp = is_rated.transpose().dot(is_rated)
    min_check = tmp < COMMON_MIN
    min_check = min_check * 1
    min_check = min_check.toarray()
    min_check = min_check - np.diag(np.diag(min_check))
    sims[min_check > 0] = 0
    C = sims - np.diag(np.diag(sims))

    np.save(path + '/B', B)
    np.save(path + '/C', C)
    np.save(path + '/UIM', uim)
    np.save(path + '/test_split', test)
    np.save(path + '/train_split', train)
    np.save(path + '/uim_full', uim_full)

    print('Computing LF')
    item_LF = LF_model(dataset)
    np.save(f'{path}/{dataset}_item_LF', item_LF)
