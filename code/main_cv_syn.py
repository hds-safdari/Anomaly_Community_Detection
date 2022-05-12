"""
    Main function to implement cross-validation given a number of communities.

    - Hold-out part of the dataset (pairs of edges labeled by unordered pairs (i,j));
    - Infer parameters on the training set;
    - Calculate performance measures in the test set (AUC).
"""

import csv
import os
import pickle
from argparse import ArgumentParser
import cv_functions as cvfun
import numpy as np
import tools as tl
import yaml
import sktensor as skt
import time


def main():
    p = ArgumentParser()
    p.add_argument('-K', '--K', type=int, default=2)  # number of communities
    p.add_argument('-A', '--adj', type=str, default='dutch2.dat')  # name of the network
    p.add_argument('-l', '--label', type=str, default='500_3_20.0_1.0_False_1')  # name of the network
    p.add_argument('-f', '--in_folder', type=str, default='../data/input/synthetic/edge_anomalies/')  # path of the input network
    p.add_argument('-o', '--out_folder', type=str, default='../data/output/5-fold_cv/synthetic/edge_anomalies/')  # path to store outputs
    p.add_argument('-E', '--flag_anomaly', type=int, default=1)  # if 1: then use model with anomaly
    p.add_argument('-e', '--ego', type=str, default='source')  # name of the source of the edge
    p.add_argument('-t', '--alter', type=str, default='target')  # name of the target of the edge 
    p.add_argument('-F', '--NFold', type=int, default=5)  # number of fold to perform cross-validation
    p.add_argument('-m', '--out_mask', type=bool, default=False)  # flag to output the masks
    p.add_argument('-r', '--out_results', type=bool, default=True)  # flag to output the results in a csv file
    p.add_argument('-i', '--out_inference', type=bool, default=False)  # flag to output the inferred parameters
    args = p.parse_args()
 

    '''
    Cross validation parameters and set up output directory
    '''
    NFold = args.NFold
    out_mask = args.out_mask
    out_results = args.out_results

    out_folder = args.out_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    '''
    Model parameters
    '''
    K = args.K 
    with open('setting_inference.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf['out_folder'] = out_folder
    conf['out_inference'] = args.out_inference
    conf['flag_anomaly'] = bool(args.flag_anomaly)

    if conf['flag_anomaly'] == True:
        conf['N_real'] = 5
    else:
        conf['N_real'] = 10
        
    '''
    Import data
    '''

    label = args.label
    network = args.in_folder+'syn_'+label+'.dat' 
    A, B, B_T, data_T_vals = tl.import_data(network, ego=args.ego, alter=args.alter, force_dense=True, header=0)
    nodes = A[0].nodes()
    valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
    assert any(isinstance(B, vt) for vt in valid_types)

    print('\n### CV procedure ###')

    # save the results
    cols = ['K', 'fold', 'rseed','flag_anomaly','mu','pi']
    cols.extend(['aucA_train', 'aucA_test','aucZ_train', 'aucZ_test', 'ELBO','CS_U','CS_V','F1Q_train','F1Q_test'])
    comparison = [0 for _ in range(len(cols))]
    if out_results:
        out_file = out_folder + label + '_cv.csv'
        if not os.path.isfile(out_file):  # write header
            with open(out_file, 'w') as outfile:
                wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
                wrtr.writerow(cols)
        outfile = open(out_file, 'a')
        wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
        print(f'Results will be saved in: {out_file}')

    time_start = time.time()
    L = B.shape[0]
    N = B.shape[-1] 

    prng = np.random.RandomState(seed=10)  # set seed random number generator
    rseed = prng.randint(1000)
    indices = cvfun.shuffle_indices_all_matrix(N, L, rseed=rseed)

    theta = np.load(args.in_folder + 'theta_'+label+'.npz',allow_pickle=True)
    K = theta['u'].shape[1]
    comparison[0] = K

    conf['end_file'] = str(label)  # needed to plot inferred theta
    init_end_file = conf['end_file']

    for fold in range(NFold):
        print('\nFOLD ', fold)
        comparison[1], comparison[2],comparison[3] = fold, rseed, conf['flag_anomaly']

        mask = cvfun.extract_mask_kfold(indices, N, fold=fold, NFold=NFold)
        if out_mask:
            outmask = out_folder + 'mask_f' + str(fold) + '_' + label + '.pkl'
            print(f'Mask saved in: {outmask}')
            with open(outmask, 'wb') as f:
                pickle.dump(np.where(mask > 0), f)

        '''
        Set up training dataset    
        '''
        B_train = B.copy()
        B_train[mask > 0] = 0

        '''
        Run CRep on the training 
        '''
        tic = time.time()
        conf['end_file'] = init_end_file + '_' + str(fold) + 'K' + str(K)
        u, v, w, pi, mu, maxELBO, algo_obj = cvfun.fit_model(B_train, nodes=nodes,mask=np.logical_not(mask), N=N, L=L, K=K,
                                                         **conf)

        '''
        Output performance results
        '''
        comparison[4], comparison[5] = mu,pi

        M0 = cvfun._lambda0_full(u, v, w)
        if conf['flag_anomaly'] == True:
            # Q = algo_obj.Q_ij_dense_f.copy()
            Q = cvfun.calculate_Q_dense(B,M0,pi,mu)
        else:
            Q = np.zeros_like(M0)
        M = cvfun.calculate_expectation(u,v,w,Q,pi)
        Q = cvfun.calculate_Q_dense(B,M0,pi,mu)
        
        comparison[6] = cvfun.calculate_AUC(M[0], B[0], mask = np.logical_not(mask[0]))
        comparison[7] = cvfun.calculate_AUC(M[0], B[0], mask = mask[0])

        Z = theta['z']
        
        maskQ_train = np.logical_and(np.logical_not(mask[0]),B[0]>0)
        maskQ_test = np.logical_and(mask[0]>0,B[0]>0) # count for existing edges-only 
        comparison[8] = cvfun.calculate_AUC(Q[0], Z, mask = maskQ_train)
        comparison[9] = cvfun.calculate_AUC(Q[0], Z, mask = maskQ_test) 

        comparison[10] = maxELBO

        u, cs_u = cvfun.cosine_similarity(u,theta['u'])
        v, cs_v = cvfun.cosine_similarity(v,theta['v'])
        comparison[11], comparison[12] = cs_u,cs_v

        f1_train = cvfun.calculate_f1_score(Q,Z, mask = maskQ_train)
        f1_test = cvfun.calculate_f1_score(Q,Z, mask = maskQ_test)
        comparison[13], comparison[14] = f1_train,f1_test

        print(f'Time elapsed: {np.round(time.time() - tic, 2)} seconds.')

        print(f'AUC A test: {comparison[7]} - AUC Z test: {comparison[9]}')
        print(f'CS U: {comparison[11]} - CS V: {comparison[12]}')
        print(f'F1-Z: {comparison[14]}')

        if out_results:
            wrtr.writerow(comparison)
            outfile.flush()

    if out_results:
        outfile.close()

    print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')
    print(f'Results saved in: {out_file}')

if __name__ == '__main__':
    main()








