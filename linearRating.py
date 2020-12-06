import argparse
import os

from optimization_utils import general_double
from test_train_split import train_test_split
from metrics import *
from utils import check_base_files


def complete(dataset, sim_ord, dis_ord, sim_neigh, dis_neigh, alpha, mu, graphType):
    save_path = f'./{dataset}/experiments/'
    mat_name = f'linear_rating_{graphType}_{sim_neigh}NN_{dis_neigh}FN_{sim_ord}simord_{dis_ord}disord_{str(alpha)[-1]}.npy'

    if not check_base_files(dataset):
        train_test_split(dataset)

    if not os.path.isfile(save_path+mat_name):
        train_test_split(dataset)

        uim_predicted = general_double(dataset, sim_neigh, dis_neigh, sim_ord, dis_ord, alpha, mu, graphType)
        np.save(save_path + mat_name, uim_predicted)
        print(f'Completed! The generated matrix is in {save_path} and it is saved as {mat_name}')
    else:
        print('The matrix is already present! Loading...')
        uim_predicted = np.load(save_path + mat_name)

    #### RESULTS ####
    test = np.load(f'{dataset}/matrices/test_split.npy')
    item_LF = np.load(f'{dataset}/matrices/{dataset}_item_LF.npy')
    rmse = RMSE(uim_predicted, test)
    ad = compute_aggregated_diversity(uim_predicted, 20)
    id = compute_individual_diversity(uim_predicted, 20, item_LF)
    print(f'RMSE: {rmse:.2f}, AD: {ad:.2f}, ID: {id:.2f}')
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('-method', action='store', dest='graphType', default='UU')
    parser.add_argument('-simOrd', action='store', dest='simOrder', default=1, type=int)
    parser.add_argument('-disOrd', action='store', dest='disOrder', default=1, type=int)
    parser.add_argument('-simNeigh', action='store', dest='simNeigh', default=30, type=int)
    parser.add_argument('-disNeigh', action='store', dest='disNeigh', default=40, type=int)
    parser.add_argument('-mu', action='store', dest='mu', default=1.0, type=float)
    parser.add_argument('-alpha', action='store', dest='alpha', default=0.1, type=float)
    parser.add_argument('-dataset', action='store', dest='dataset', default='ml100k')

    args = parser.parse_args()

    graphType = args.graphType
    sim_ord = args.simOrder
    dis_ord = args.disOrder
    sim_neigh = args.simNeigh
    dis_neigh = args.disNeigh
    mu = args.mu
    alpha = args.alpha
    dataset = args.dataset
    print(f'Dataset: {dataset}\n'
          f'Method: {graphType}\n'
          f'Neighbors in similarity graph: {sim_neigh}\n'
          f'Order of similarity filter: {sim_ord}\n'
          f'Neighbors in dissimilarity graph: {dis_neigh}\n'
          f'Order of dissimilarity filter: {dis_ord}\n'
          f'Alpha: {alpha}\n'
          f'mu: {mu}\n'
          )
    complete(dataset, sim_ord, dis_ord, sim_neigh, dis_neigh, alpha, mu, graphType)
