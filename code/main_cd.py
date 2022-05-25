"""
    Main function to implement community-detection given a number of communities.

    - Infer parameters on the whole dataset;
    - Calculate performance measures in the test set (CS).
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
    p.add_argument('-o', '--out_folder', type=str, default='../data/output/community_detection/synthetic/edge_anomalies/')  # path to store outputs
    p.add_argument('-E', '--flag_anomaly', type=int, default=1)  # if 1: then use model with anomaly
    p.add_argument('-e', '--ego', type=str, default='source')  # name of the source of the edge
    p.add_argument('-t', '--alter', type=str, default='target')  # name of the target of the edge
    # p.add_argument('-d', '--force_dense', type=bool, default=True)  # flag to force a dense transformation in input
    p.add_argument('-F', '--NFold', type=int, default=5)  # number of fold to perform cross-validation
    p.add_argument('-m', '--out_mask', type=bool, default=False)  # flag to output the masks
    p.add_argument('-r', '--out_results', type=bool, default=True)  # flag to output the results in a csv file
    p.add_argument('-i', '--out_inference', type=bool, default=False)  # flag to output the inferred parameters
    args = p.parse_args()

    # TODO: optimize when force_dense=False

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
    # network = args.in_folder + args.adj  # network complete path
    # algorithm = args.algorithm  # algorithm to use to generate the samples
    # adjacency = args.adj.split('.dat')[0]  # name of the network without extension
    with open('setting_cd.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf['out_folder'] = out_folder
    conf['out_inference'] = args.out_inference
    conf['flag_anomaly'] = bool(args.flag_anomaly)
    '''
    Import data
    '''

    label = args.label
    network = args.in_folder+'syn_'+label+'.dat'
    # A, B = tl.import_data(network, ego=args.ego, alter=args.alter, force_dense=args.force_dense, header=0)
    A, B, B_T, data_T_vals = tl.import_data(network, ego=args.ego, alter=args.alter, force_dense=True, header=0)
    nodes = A[0].nodes()
    valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
    assert any(isinstance(B, vt) for vt in valid_types)

    print('\n### Community detection procedure ###')

    # save the results
    cols = ['K','flag_anomaly','muGT','piGT','mu','pi']
    cols.extend(['aucA', 'aucZ', 'ELBO','CS_U','CS_V','F1_U','F1_V','F1Q'])
    comparison = [0 for _ in range(len(cols))]
    if out_results:
        out_file = out_folder + label + '_cd.csv'
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

    theta = np.load(args.in_folder + 'theta_'+label+'.npz',allow_pickle=True)
    K = theta['u'].shape[1]
    comparison[0] = K
    comparison[1] = conf['flag_anomaly']
    comparison[2], comparison[3] = theta['mu'], theta['pi']

    conf['end_file'] = str(label)  # needed to plot inferred theta
    init_end_file = conf['end_file']

    '''
    Run anomaly detection
    '''
    tic = time.time()
    conf['end_file'] = init_end_file + '_' +'K' + str(K)
    u, v, w, pi, mu, maxELBO, algo_obj = cvfun.fit_model(B, nodes=nodes, N=N, L=L, K=K,
                                                     **conf)

    '''
    Output performance results
    '''
    comparison[4], comparison[5] = mu,pi

    M0 = cvfun._lambda0_full(u, v, w)
    if conf['flag_anomaly'] == True:
        Q = cvfun.calculate_Q_dense(B,M0,pi,mu)
    else:
        Q = np.zeros_like(M0)
    M = cvfun.calculate_expectation(u,v,w,Q,pi)
    Q = cvfun.calculate_Q_dense(B,M0,pi,mu)
    
    comparison[6] = cvfun.calculate_AUC(M[0], B[0])

    Z = theta['z']
    
    maskQ = B[0]>0
    comparison[7] = cvfun.calculate_AUC(Q[0], Z, mask = maskQ)

    comparison[8] = maxELBO


    u, cs_u = cvfun.cosine_similarity(u,theta['u'])
    v, cs_v = cvfun.cosine_similarity(v,theta['v'])
    comparison[9], comparison[10] = cs_u,cs_v

    u = cvfun.normalize_nonzero_membership(u)
    v = cvfun.normalize_nonzero_membership(v)

    f1_u = cvfun.evalu(u,theta['u'], 'f1')
    f1_v = cvfun.evalu(v,theta['v'], 'f1')
    comparison[11], comparison[12] = f1_u,f1_v

    f1 = cvfun.calculate_f1_score(Q,Z, mask = maskQ)
    comparison[13] = f1

    print(f'Time elapsed: {np.round(time.time() - tic, 2)} seconds.')

    print(f'AUC A: {comparison[6]} - AUC Z: {comparison[7]}')
    print(f'F1-Z: {comparison[13]}')
    print(f'CS U: {comparison[9]} - CS V: {comparison[10]}')
    print(f'F1 U: {comparison[11]} - F1 V: {comparison[12]}')

    if out_results:
        wrtr.writerow(comparison)
        outfile.flush()

    if out_results:
        outfile.close()

    print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')
    print(f'Results saved in: {out_file}')

if __name__ == '__main__':
    main()








