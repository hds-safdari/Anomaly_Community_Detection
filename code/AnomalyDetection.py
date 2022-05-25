"""
    Class definition of ACD, the algorithm to perform inference in networks with anomaly.
    The latent variables are related to community memberships and anomaly parameters.
"""

from __future__ import print_function
import time
import sys
import sktensor as skt
import numpy as np
import pandas as pd
from termcolor import colored 
import numpy.random as rn
import scipy.special
from scipy.stats import poisson

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import time_glob as gl


EPS = 1e-12

class AnomalyDetection:
    def __init__(self, N=100, L=1, K=5, undirected=False, initialization=1, ag=1.,bg=5.,  rseed=10, inf=1e10, err_max=1e-8, err=0.01,
                 N_real=1, tolerance=0.1, decision=2, max_iter=500, out_inference=False,
                 out_folder='../data/output/', end_file='.dat', assortative=False, pibr0 = None, mupr0= None,
                 in_parameters = '../data/input/theta_111',
                 fix_communities=False,fix_w=False,fix_pibr=False, fix_mupr=False,plot_loglik=False,
                 constrained=False, verbose=False, flag_anomaly = True):
        self.N = N  # number of nodes
        self.L = L  # number of layers
        self.K = K  # number of communities
        self.undirected = undirected  # flag to call the undirected network
        self.rseed = rseed  # random seed for the initialization
        self.inf = inf  # initial value of the log-likelihood
        self.err_max = err_max  # minimum value for the parameters
        self.err = err  # noise for the initialization
        self.N_real = N_real  # number of iterations with different random initialization
        self.tolerance = tolerance  # tolerance parameter for convergence
        self.decision = decision  # convergence parameter
        self.max_iter = max_iter  # maximum number of EM steps before aborting
        self.out_inference = out_inference  # flag for storing the inferred parameters
        self.out_folder = out_folder  # path for storing the output
        self.end_file = end_file  # output file suffix
        self.assortative = assortative  # if True, the network is assortative
        self.fix_pibr = fix_pibr  # if True, the pibr parameter is fixed
        self.fix_mupr = fix_mupr  # if True, the mupr parameter is fixed
        self.fix_communities = fix_communities  # if True, the community membership parametera are fixed
        self.fix_w = fix_w  # if True, the affinity matrix is fixed
        self.constrained = constrained  # if True, use the configuration with constraints on the updates
        self.verbose = verbose  # flag to print details 
        self.ag = ag # shape of gamma prior
        self.bg = bg # rate of gamma prior
        self.pibr = pibr0  # initial value for the mu 
        self.mupr = mupr0  # initial value for the pi  
        self.plot_loglik = plot_loglik
        self.flag_anomaly = flag_anomaly
        self.in_parameters = in_parameters  # path of the input community membership (when initialization=1)

        if initialization not in {0, 1}:  # indicator for choosing how to initialize u, v and w
            raise ValueError('The initialization parameter can be either 0 or 1. It is used as an indicator to '
                             'initialize the membership matrices u and v and the affinity matrix w. If it is 0, they '
                             'will be generated randomly, otherwise they will upload from file.')
        self.initialization = initialization  

        if self.pibr is not None:
            if (self.pibr < 0) or (self.pibr > 1):
                raise ValueError('The anomaly parameter pibr0 has to be in [0, 1]!')

        if self.mupr is not None:
            if (self.mupr < 0) or (self.mupr > 1):
                raise ValueError('The prior mupr0 has to be in [0, 1]!')

        if self.initialization == 1:
            theta = np.load(self.in_parameters + '.npz',allow_pickle=True) 
            self.N, self.K = theta['u'].shape
            print('self.N,self.K:', self.N,self.K)

        # values of the parameters used during the update
        self.u = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v = np.zeros((self.N, self.K), dtype=float)  # in-going membership 

        # values of the parameters in the previous iteration
        self.u_old = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v_old = np.zeros((self.N, self.K), dtype=float)  # in-going membership 
        self.pibr_old = self.pibr  #
        self.mupr_old = self.mupr #

        # final values after convergence --> the ones that maximize the log-likelihood
        self.u_f = np.zeros((self.N, self.K), dtype=float)  # Out-going membership
        self.v_f = np.zeros((self.N, self.K), dtype=float)  # In-going membership 
        self.pibr_f = self.pibr  # pi: anomaly parameter
        self.mupr_f = self.mupr  # mu: prior

        # values of the affinity tensor: in this case w is always ASSORTATIVE 
        if self.assortative:  # purely diagonal matrix
            self.w = np.zeros((self.L, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K), dtype=float)
        else:
            self.w = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K, self.K), dtype=float)

        if self.fix_pibr == True:
            self.pibr = self.pibr_old = self.pibr_f = pibr0
        if self.fix_mupr == True:
            self.mupr = self.mupr_old = self.mupr_f = mupr0

        if self.flag_anomaly == False:
            self.pibr = self.pibr_old = self.pibr_f = 1.
            self.mupr = self.mupr_old = self.mupr_f = 0.
            self.fix_pibr = self.fix_mupr = True

    @gl.timeit('fit')
    def fit(self, data, nodes, mask=None):
        """
            Model  networks by using a probabilistic generative model that assume community parameters and
            anomaly parameters. The inference is performed via EM algorithm.

            Parameters
            ----------
            data : ndarray/sptensor
                   Graph adjacency tensor.
            data_T: None/sptensor
                    Graph adjacency tensor (transpose). 
            nodes : list
                    List of nodes IDs.
            mask : ndarray
                   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

            Returns
            -------
            u_f : ndarray
                  Out-going membership matrix.
            v_f : ndarray
                  In-coming membership matrix.
            w_f : ndarray
                  Affinity tensor.
            pibr_f : float
                    Bernolie parameter.
            mupr_f : float
                    prior .
            maxL : float
                   Maximum  log-likelihood.
            final_it : int
                       Total number of iterations.
        """

        maxL = -self.inf  # initialization of the maximum  log-likelihood 

        # if data_T is None: 
        data_T = np.einsum('aij->aji', data)
        data_T_vals = get_item_array_from_subs(data_T, data.nonzero())
        # pre-processing of the data to handle the sparsity
        data = preprocess(data)
        data_T = preprocess(data_T) 

        # save the indexes of the nonzero entries
        if isinstance(data, skt.dtensor):
            subs_nz = data.nonzero()
        elif isinstance(data, skt.sptensor):
            subs_nz = data.subs 

        if mask is not None:
            subs_nz_mask = mask.nonzero() 
        else:
            subs_nz_mask = None


        rng = np.random.RandomState(self.rseed)

        for r in range(self.N_real):

            self._initialize(rng=np.random.RandomState(self.rseed))

            self._update_old_variables()
            self._update_cache(data, data_T_vals, subs_nz)

            # convergence local variables
            coincide, it = 0, 0
            convergence = False
            loglik = self.inf
            loglik_values = []

            if self.verbose  == 2:
                print(f'Updating realization {r} ...', end=' ')
            time_start = time.time()
            # --- single step iteration update ---
            while not convergence and it < self.max_iter:
                # main EM update: updates memberships and calculates max difference new vs old

                delta_u, delta_v, delta_w, delta_pibr, delta_mupr = self._update_em(data, data_T_vals, subs_nz,mask=mask,subs_nz_mask=subs_nz_mask)
                it, loglik, coincide, convergence = self._check_for_convergence(data, it, loglik, coincide, convergence,
                                                                                data_T=data_T, mask=mask,subs_nz_mask=subs_nz_mask)
                loglik_values.append(loglik)

            if maxL < loglik:
                self._update_optimal_parameters()
                maxL = loglik
                self.final_it = it
                conv = convergence
                best_loglik_values = list(loglik_values)
            self.rseed += rng.randint(100000000)

            if self.verbose > 0:
                print(f'Nreal = {r} - ELBO = {loglik} - ELBOmax = {maxL} - it = {it}  '

                      f'time = {np.round(time.time() - time_start, 2)} seconds')
        # end cycle over realizations

        self.maxL = maxL
        if self.final_it == self.max_iter and not conv:
            # convergence not reaches
            print(colored('Solution failed to converge in {0} EM steps!'.format(self.max_iter), 'blue'))

        if self.plot_loglik:
            plot_L(best_loglik_values, int_ticks=True)

        if self.out_inference:
            self.output_results(nodes)

        return self.u_f, self.v_f, self.w_f, self.pibr_f, self.mupr_f, maxL

    def _initialize(self, rng=None):
        """
            Random initialization of the parameters u, v, w, beta.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)

        if self.fix_pibr == False: 
            self._randomize_pibr(rng) 
        
        if self.fix_mupr == False:   
            self._randomize_mupr(rng)

        if self.initialization == 0:
            if self.verbose == 2:
                print('u, v and w are initialized randomly.')
            self._randomize_w(rng=rng)
            self._randomize_u_v(rng=rng)

        if self.initialization == 0:
            if self.verbose:
                print('u, v and w are initialized randomly.')
            self._randomize_w(rng=rng)
            self._randomize_u_v(rng=rng)

        elif self.initialization == 1:
            if self.verbose:
                print('u, v and w are initialized using the input files:')
                print(self.in_parameters + '.npz')
            theta = np.load(self.in_parameters + '.npz',allow_pickle=True)
            self._initialize_u(theta['u'])
            self._initialize_v(theta['v'])
            self.N = self.u.shape[0]

            self._randomize_w(rng=rng)

    def _initialize_u(self, u0):
        if u0.shape[0] != self.N:
            raise ValueError('u.shape is different that the initialized one.',self.N,u0.shape[0])
        self.u = u0.copy()
        max_entry = np.max(u0)
        self.u += max_entry * self.err * np.random.random_sample(self.u.shape)

    def _initialize_v(self, v0):
        if v0.shape[0] != self.N:
            raise ValueError('v.shape is different that the initialized one.',self.N,v0.shape[0])
        self.v = v0.copy()
        max_entry = np.max(v0)
        self.v += max_entry * self.err * np.random.random_sample(self.v.shape)

    
    def _randomize_pibr(self, rng=None):
        """
            Generate a random number in (0, 1.).

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        self.pibr = rng.random_sample(1)[0]
    
    def _randomize_mupr(self, rng=None):
        """
            Generate a random number in (0, 1.).

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        self.mupr = rng.random_sample(1)[0]

    def _randomize_w(self, rng):
        """
            Assign a random number in (0, 1.) to each entry of the affinity tensor w.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        for i in range(self.L):
            for k in range(self.K):
                if self.assortative:
                    self.w[i, k] =  rng.random_sample(1)
                else:
                    for q in range(k, self.K):
                        if q == k:
                            self.w[i, k, q] =  rng.random_sample(1)
                        else:
                            self.w[i, k, q] = self.w[i, q, k] = self.err * rng.random_sample(1)

    def _randomize_u_v(self, rng=None):
        """
            Assign a random number in (0, 1.) to each entry of the membership matrices u and v, and normalize each row.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        self.u = rng.random_sample(self.u.shape)
        row_sums = self.u.sum(axis=1)
        self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
 
        if not self.undirected:
            self.v = rng.random_sample(self.v.shape)
            row_sums = self.v.sum(axis=1)
            self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        else:
            self.v = self.u


    def _initialize_w(self, infile_name):
        """
            Initialize affinity tensor w from file.

            Parameters
            ----------
            infile_name : str
                          Path of the input file.
        """

        with open(infile_name, 'rb') as f:
            dfW = pd.read_csv(f, sep='\s+',header=None)
            if self.assortative:
                self.w = np.diag(dfW)[np.newaxis, :].copy()
            else:
                self.w = dfW.values[np.newaxis, :, :]
        if self.fix_w == False:
            max_entry = np.max(self.w)
            self.w += max_entry * self.err * np.random.random_sample(self.w.shape)

    def _update_old_variables(self):
        """
            Update values of the parameters in the previous iteration.
        """

        self.u_old[self.u > 0] = np.copy(self.u[self.u > 0])
        self.v_old[self.v > 0] = np.copy(self.v[self.v > 0])
        self.w_old[self.w > 0] = np.copy(self.w[self.w > 0]) 
        self.pibr_old = np.copy(self.pibr)
        self.mupr_old = np.copy(self.mupr)

    @gl.timeit('cache')
    def _update_cache(self, data, data_T_vals, subs_nz):
        """
            Update the cache used in the em_update.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            data_T_vals : ndarray
                          Array with values of entries A[j, i] given non-zero entry (i, j).
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
        """

        self.lambda0_nz = self._lambda0_nz(subs_nz, self.u, self.v, self.w) 
        if self.assortative == False:
            self.lambda0_nzT = self._lambda0_nz(subs_nz, self.v, self.u, np.einsum('akq->aqk',self.w))
        else:
            self.lambda0_nzT = self._lambda0_nz(subs_nz, self.v, self.u,self.w)
        if self.flag_anomaly == True:
            self.Qij_dense,self.Qij_nz = self._QIJ(data, data_T_vals, subs_nz) 
        self.M_nz = self.lambda0_nz 
        self.M_nz[self.M_nz == 0] = 1 

        if isinstance(data, skt.dtensor): 
            self.data_M_nz = data[subs_nz] / self.M_nz
        elif isinstance(data, skt.sptensor):
            self.data_M_nz   = data.vals / self.M_nz
            if self.flag_anomaly == True:
                self.data_M_nz_Q = data.vals * (1-self.Qij_nz) / self.M_nz 
            else:
                self.data_M_nz_Q = data.vals / self.M_nz 

    @gl.timeit('QIJ')
    def _QIJ(self, data, data_T_vals, subs_nz):
        """
            Compute the mean lambda0_ij for only non-zero entries.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.
            data_T_vals : ndarray
                Array with values of entries A[j, i] given non-zero entry (i, j).

            Returns
            -------
            nz_recon_I : ndarray
                         Mean lambda0_ij for only non-zero entries.
        """  

        if isinstance(data, skt.dtensor): 
            nz_recon_I =  np.power(self.pibr,data[subs_nz])
        elif isinstance(data, skt.sptensor):
            A_tot = data.vals + data_T_vals
            nz_recon_I =  self.mupr * np.power(self.pibr,A_tot) * np.power(1-self.pibr,2-A_tot)
            nz_recon_Id = nz_recon_I + (1-self.mupr) * poisson.pmf(data.vals, self.lambda0_nz) * poisson.pmf(data_T_vals, self.lambda0_nzT)
            non_zeros = nz_recon_Id > 0 
            nz_recon_I[non_zeros] /=  nz_recon_Id[non_zeros]   

        lambda0_ija = self._lambda0_full(self.u, self.v, self.w) 
        Q_ij_dense = np.ones(lambda0_ija.shape)
        Q_ij_dense *=  self.mupr * np.power(1-self.pibr,2)
        Q_ij_dense_d = Q_ij_dense + (1-self.mupr) * np.exp(-(lambda0_ija+transpose_tensor(lambda0_ija)))
        non_zeros = Q_ij_dense_d > 0
        Q_ij_dense[non_zeros] /= Q_ij_dense_d[non_zeros]
        assert np.allclose(Q_ij_dense[0], Q_ij_dense[0].T, rtol=1e-05, atol=1e-08)

        Q_ij_dense[subs_nz] = nz_recon_I 

        Q_ij_dense = np.maximum( Q_ij_dense, transpose_tensor(Q_ij_dense))  
        np.fill_diagonal(Q_ij_dense[0], 0.)

        assert (Q_ij_dense > 1).sum() == 0
        return Q_ij_dense, Q_ij_dense[subs_nz]

    def Q_func_dense(self,data):
        A = data.toarray()
        A = A[0]
        lambdaij = self._lambda0_full(self.u, self.v, self.w)[0]
        numerator = self.mupr * np.power(self.pibr,A) * np.power(1.-self.pibr,1.-A) * np.power(self.pibr,A.T) * np.power(1.-self.pibr,1.-A.T)
        denominator = numerator + poisson.pmf(A,lambdaij) * poisson.pmf(A.T,lambdaij.T) * (1-self.mupr)
        idx = (denominator == 0)
        numerator[idx] = 0.
        denominator[idx] = 1.
        Q = numerator/denominator
        np.fill_diagonal(Q, 0.)
        return Q[np.newaxis,:,:]

    def _lambda0_nz(self, subs_nz, u, v, w):
        """
            Compute the mean lambda0_ij for only non-zero entries.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.

            Returns
            -------
            nz_recon_I : ndarray
                         Mean lambda0_ij for only non-zero entries.
        """

        if not self.assortative:
            nz_recon_IQ = np.einsum('Ik,Ikq->Iq', u[subs_nz[1], :], w[subs_nz[0], :, :])
        else:
            nz_recon_IQ = np.einsum('Ik,Ik->Ik', u[subs_nz[1], :], w[subs_nz[0], :])
        nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, v[subs_nz[2], :])

        return nz_recon_I
    
    @gl.timeit('em')
    def _update_em(self, data, data_T_vals, subs_nz,mask=None,subs_nz_mask=None):
        """
            Update parameters via EM procedure.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            data_T_vals : ndarray
                          Array with values of entries A[j, i] given non-zero entry (i, j).
            subs_nz : tuple
                      Indices of elements of data that are non-zero. 

            Returns
            -------
            d_u : float
                  Maximum distance between the old and the new membership matrix u.
            d_v : float
                  Maximum distance between the old and the new membership matrix v.
            d_w : float
                  Maximum distance between the old and the new affinity tensor w.
            d_pibr : float
                    Maximum distance between the old and the new anoamly parameter pi.
            d_mupr : float
                    Maximum distance between the old and the new prior mu.
        """
        
        if self.fix_communities == False: 
            d_u = self._update_U(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask) 
            self._update_cache(data, data_T_vals, subs_nz) 
            if self.undirected:
                self.v = self.u
                self.v_old = self.v
                d_v = d_u
            else: 
                d_v = self._update_V(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask) 
            self._update_cache(data, data_T_vals, subs_nz)

        else:
            d_u = d_v = 0.

        if self.fix_w == False:
            if not self.assortative:
                d_w = self._update_W(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask)
            else: 
                d_w = self._update_W_assortative(subs_nz,mask=mask,subs_nz_mask=subs_nz_mask) 
            self._update_cache(data, data_T_vals, subs_nz)
        else:
            d_w = 0

        if self.fix_pibr == False: 
            d_pibr = self._update_pibr(data, data_T_vals, subs_nz,mask=mask,subs_nz_mask=subs_nz_mask) 
            self._update_cache(data, data_T_vals, subs_nz)

        else:
            d_pibr = 0.
        
        if self.fix_mupr == False: 
            d_mupr = self._update_mupr(data, data_T_vals, subs_nz,mask=mask,subs_nz_mask=subs_nz_mask) 
            self._update_cache(data, data_T_vals, subs_nz)
        else:
            d_mupr = 0.

        return d_u, d_v, d_w, d_pibr, d_mupr

    @gl.timeit('pibr')
    def _update_pibr(self, data, data_T_vals, subs_nz,mask=None,subs_nz_mask=None):
        """
            Update  anomaly parameter pi.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.  

            Returns
            -------
            dist_pibr : float
                       Maximum distance between the old and the new anomaly parameter pi.
        """  
        if isinstance(data, skt.dtensor):
            Adata = (data[subs_nz] * self.Qij_nz).sum()
        elif isinstance(data, skt.sptensor):
            Adata   = (data.vals * self.Qij_nz).sum() 
        if mask is None:    
            self.pibr = Adata / self.Qij_dense.sum() 
        else:
            self.pibr = Adata / self.Qij_dense[subs_nz_mask].sum()  
 
        dist_pibr = abs(self.pibr - self.pibr_old) 
        self.pibr_old = np.copy(self.pibr)

        return dist_pibr

    @gl.timeit('mupr')
    def _update_mupr(self, data, data_T_vals, subs_nz,mask=None,subs_nz_mask=None):
        """
            Update prior eta.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.  

            Returns
            -------
            dist_mupr : float
                       Maximum distance between the old and the new rprior mu.
        """
        if mask is None:
            self.mupr = self.Qij_dense.sum() / ( self.N * (self.N-1) )
        else:
            self.mupr = self.Qij_dense[subs_nz_mask].sum() / ( self.N * (self.N-1) )

        dist_mupr = abs(self.pibr - self.mupr_old)
        self.mupr_old = np.copy(self.mupr) 

        return dist_mupr 

    @gl.timeit('U')
    def _update_U(self, subs_nz,mask=None,subs_nz_mask=None):
        """
            Update out-going membership matrix.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.

            Returns
            -------
            dist_u : float
                     Maximum distance between the old and the new membership matrix u.
        """    

        self.u = self.ag - 1 + self.u_old * self._update_membership_Q(subs_nz, self.u, self.v, self.w, 1)  

        if not self.constrained:  
            if self.flag_anomaly == True:
                if mask is None:
                    Du = np.einsum('aij,jq->iq', 1-self.Qij_dense,self.v)
                else:
                    Du = np.einsum('aij,jq->iq', mask * (1-self.Qij_dense),self.v)
                if not self.assortative:
                    w_k  = np.einsum('akq->kq', self.w)
                    Z_uk = np.einsum('iq,kq->ik', Du, w_k)
                else:
                    w_k = np.einsum('ak->k', self.w)
                    Z_uk = np.einsum('ik,k->ik', Du, w_k) 
                
            else: # flag_anomaly == False
                Du = np.einsum('jq->q',self.v)
                if not self.assortative:
                    w_k  = np.einsum('akq->kq', self.w)
                    Z_uk = np.einsum('q,kq->k', Du, w_k)
                else:
                    w_k = np.einsum('ak->k', self.w)
                    Z_uk = np.einsum('k,k->k', Du, w_k)
            Z_uk += self.bg 
            non_zeros = Z_uk > 0. 

            if self.flag_anomaly == True:
                self.u[Z_uk == 0] = 0.
                self.u[non_zeros] /=  Z_uk[non_zeros]
            else:
                self.u[:,Z_uk == 0] = 0.
                self.u[:,non_zeros] /=  Z_uk[non_zeros]
 
        else: 
            row_sums = self.u.sum(axis=1)
            self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    @gl.timeit('V')
    def _update_V(self, subs_nz,mask=None,subs_nz_mask=None):
        """
            Update in-coming membership matrix.
            Same as _update_U but with:
            data <-> data_T
            w <-> w_T
            u <-> v

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.

            Returns
            -------
            dist_v : float
                     Maximum distance between the old and the new membership matrix v.
        """

        self.v = self.ag - 1 + self._update_membership_Q(subs_nz, self.u, self.v, self.w, 2)

        if not self.constrained:
            if self.flag_anomaly == True:
                if mask is None:
                    Dv = np.einsum('aij,ik->jk', 1-self.Qij_dense, self.u)
                else:
                    Dv = np.einsum('aij,ik->jk', mask * (1-self.Qij_dense), self.u)
                if not self.assortative:
                    w_k = np.einsum('aqk->qk', self.w)
                    Z_vk = np.einsum('iq,qk->ik', Dv, w_k)
                else:
                    w_k = np.einsum('ak->k', self.w)
                    Z_vk = np.einsum('ik,k->ik', Dv, w_k) 

            else: # flag_anomaly == False
                Dv = np.einsum('ik->k', self.u)
                if not self.assortative:
                    w_k = np.einsum('aqk->qk', self.w)
                    Z_vk = np.einsum('q,qk->k', Dv, w_k)
                else:
                    w_k = np.einsum('ak->k', self.w)
                    Z_vk = np.einsum('k,k->k', Dv, w_k)

            Z_vk += self.bg
            non_zeros = Z_vk > 0

            if self.flag_anomaly == True:
                self.v[Z_vk == 0] = 0.
                self.v[non_zeros] /=  Z_vk[non_zeros]
            else:
                self.v[:,Z_vk == 0] = 0.
                self.v[:,non_zeros] /=  Z_vk[non_zeros] 
        else:
            row_sums = self.v.sum(axis=1)
            self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    @gl.timeit('W')
    def _update_W(self, subs_nz,mask=None,subs_nz_mask=None):
        """
            Update affinity tensor.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.

            Returns
            -------
            dist_w : float
                     Maximum distance between the old and the new affinity tensor w.
        """
        sub_w_nz = self.w.nonzero()
        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz_Q[:, np.newaxis, np.newaxis] * UV
        for a, k, q in zip(*sub_w_nz):
            uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k, q], minlength=self.L)

        self.w = self.ag - 1 + self.w * uttkrp_DKQ

        if self.flag_anomaly == True:
            if mask is None:
                UQk = np.einsum('aij,ik->ajk', (1-self.Qij_dense), self.u)
            else:
                UQk = np.einsum('aij,ik->ajk', mask * (1-self.Qij_dense), self.u)
            Z = np.einsum('ajk,jq->akq', UQk, self.v) 
        else: # flag_anomaly == False
            # Z = np.einsum('ik,jq->kq',self.u,self.v)
            Z = np.einsum('k,q->kq', self.u.sum(axis=0), self.v.sum(axis=0))[np.newaxis, :, :]
        Z += self.bg

        non_zeros = Z > 0
        self.w[Z == 0] = 0.
        self.w[non_zeros] /= Z[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    @gl.timeit('W_assortative')
    def _update_W_assortative(self, subs_nz,mask=None,subs_nz_mask=None):
        """
            Update affinity tensor (assuming assortativity).

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.

            Returns
            -------
            dist_w : float
                     Maximum distance between the old and the new affinity tensor w.
        """
        # Let's make some changes!

        uttkrp_DKQ = np.zeros_like(self.w)  

        UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz_Q[:, np.newaxis] * UV
        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L)

        self.w = self.ag - 1 + self.w * uttkrp_DKQ
         
        if self.flag_anomaly == True:
            if mask is None:
                UQk = np.einsum('aij,ik->jk', (1-self.Qij_dense), self.u)
                Zk = np.einsum('jk,jk->k', UQk, self.v)
                Zk = Zk[np.newaxis,:]
            else:
                Zk = np.einsum('aij,ijk->ak',mask * (1-self.Qij_dense),np.einsum('ik,jk->ijk',self.u,self.v)) 
        else: # flag_anomaly == False
            Zk = np.einsum('ik,jk->k', self.u, self.v)
            Zk = Zk[np.newaxis,:]
        Zk += self.bg

        non_zeros = Zk > 0
        self.w[Zk == 0] = 0
        self.w[non_zeros] /= Zk[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)    

        return dist_w 

    def _update_membership_Qd(self, subs_nz, u, v, w, m):
        """
            Return the Khatri-Rao product (sparse version) used in the update of the membership matrices.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.
            m : int
                Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
                works with the matrix u; if 2 it works with v.

            Returns
            -------
            uttkrp_DK : ndarray
                        Matrix which is the result of the matrix product of the unfolding of the tensor and the
                        Khatri-Rao product of the membership matrix.
        """

        if not self.assortative:
            uttkrp_DK = sp_uttkrp((1-self.Qij), subs_nz, m, u, v, w)
        else:
            uttkrp_DK = sp_uttkrp_assortative((1-self.Qij), subs_nz, m, u, v, w)

        return uttkrp_DK

    def _update_membership_Q(self, subs_nz, u, v, w, m):
        """
            Return the Khatri-Rao product (sparse version) used in the update of the membership matrices.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.
            m : int
                Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
                works with the matrix u; if 2 it works with v.

            Returns
            -------
            uttkrp_DK : ndarray
                        Matrix which is the result of the matrix product of the unfolding of the tensor and the
                        Khatri-Rao product of the membership matrix.
        """

        if not self.assortative:
            uttkrp_DK = sp_uttkrp(self.data_M_nz_Q, subs_nz, m, u, v, w)
        else:
            uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz_Q, subs_nz, m, u, v, w)

        return uttkrp_DK
 
    @gl.timeit('convergence')
    def _check_for_convergence(self, data, it, loglik, coincide, convergence, data_T=None, mask=None,subs_nz_mask=None):
        """
            Check for convergence by using the  log-likelihood values.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            it : int
                 Number of iteration.
            loglik : float
                      log-likelihood value.
            coincide : int
                       Number of time the update of the  log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
            data_T : sptensor/dtensor
                     Graph adjacency tensor (transpose).
            mask : ndarray
                   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

            Returns
            -------
            it : int
                 Number of iteration.
            loglik : float
                     Log-likelihood value.
            coincide : int
                       Number of time the update of the  log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
        """

        if it % 10 == 0:
            old_L = loglik
            loglik = self._ELBO(data, data_T=data_T, mask=mask,subs_nz_mask=subs_nz_mask)
            if abs(loglik - old_L) < self.tolerance:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    @gl.timeit('ELBO')
    def _ELBO(self, data, data_T, mask=None,subs_nz_mask=None):
        """
            Compute the  ELBO of the data.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            data_T : sptensor/dtensor
                     Graph adjacency tensor (transpose).
            mask : ndarray
                   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

            Returns
            -------
            l : float
                ELBO value.
        """


        self.lambda0_ija = self._lambda0_full(self.u, self.v, self.w)  

        if mask is not None:
            Adense = data.toarray()

        if self.flag_anomaly == False:
            l = (data.vals * np.log(self.lambda0_ija[data.subs]+EPS)).sum() 
            if mask is not None:
                l -= self.lambda0_ija.sum()
            else:
                l -= self.lambda0_ija[subs_nz_mask].sum()
            return l
        else:
            l = 0.
            if 1-self.pibr > 0:
                if mask is None:
                    logpi = self.Qij_dense * np.log(1-self.pibr)
                    logpi[data.subs] = 0.
                    l += logpi.sum()
                else:
                    subs_nz = np.logical_and(mask > 0,Adense == 0)
                    l += (self.Qij_dense[subs_nz] * (1-Adense[subs_nz])).sum() * np.log(1-self.pibr) 
                
            if self.pibr > 0:
                l += np.log(self.pibr) * (self.Qij_dense[data.subs] * data.vals).sum()
            
            if mask is None:
                non_zeros = self.Qij_dense > 0
                non_zeros1 = (1-self.Qij_dense) > 0
            else:
                non_zeros = np.logical_and( mask > 0,self.Qij_dense > 0 )
                non_zeros1 = np.logical_and( mask > 0, (1-self.Qij_dense ) > 0 )

            l -= (self.Qij_dense[non_zeros] * np.log(self.Qij_dense[non_zeros]+EPS)).sum()
            l -= ((1-self.Qij_dense)[non_zeros1]*np.log((1-self.Qij_dense)[non_zeros1]+EPS)).sum()

            if mask is None:
                l -= ((1-self.Qij_dense) * self.lambda0_ija).sum()
                l += ( ((1-self.Qij_dense)[data.subs]) * data.vals * np.log(self.lambda0_ija[data.subs]+EPS) ).sum()

                if 1 - self.mupr > 0 :
                    l += np.log(1-self.mupr+EPS) * (1-self.Qij_dense).sum()
                if self.mupr > 0 :
                    l += np.log(self.mupr+EPS) * (self.Qij_dense).sum()
            else:
                l -= ((1-self.Qij_dense[subs_nz_mask]) * self.lambda0_ija[subs_nz_mask]).sum()
                subs_nz = np.logical_and(mask > 0,Adense > 0)
                l += ( ((1-self.Qij_dense)[subs_nz]) * data.vals * np.log(self.lambda0_ija[subs_nz]+EPS) ).sum()

                if 1 - self.mupr > 0 :
                    l += np.log(1-self.mupr+EPS) * (1-self.Qij_dense)[subs_nz_mask].sum()
                if self.mupr > 0 :
                    l += np.log(self.mupr+EPS) * (self.Qij_dense[subs_nz_mask]).sum()

            if self.ag >= 1.:
                l += (self.ag -1) * np.log(self.u+EPS).sum()
                l += (self.ag -1) * np.log(self.v+EPS).sum()
            if self.bg >= 0. :
                l -= self.bg * self.u.sum()
                l -= self.bg * self.v.sum()

            if np.isnan(l):
                print("ELBO is NaN!!!!")
                sys.exit(1)
            else:
                return l


    def _lambda0_full(self, u, v, w):
        """
            Compute the mean lambda0 for all entries.

            Parameters
            ----------
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.

            Returns
            -------
            M : ndarray
                Mean lambda0 for all entries.
        """

        if w.ndim == 2:
            M = np.einsum('ik,jk->ijk', u, v)
            M = np.einsum('ijk,ak->aij', M, w)
        else:
            M = np.einsum('ik,jq->ijkq', u, v)
            M = np.einsum('ijkq,akq->aij', M, w)
        return M

    def _update_optimal_parameters(self):
        """
            Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.w_f = np.copy(self.w)
        self.pibr_f = np.copy(self.pibr)
        self.mupr_f = np.copy(self.mupr)
        if self.flag_anomaly == True:
            self.Q_ij_dense_f = np.copy(self.Qij_dense)
        else:
            self.Q_ij_dense_f = np.zeros((1,self.N,self.N))

    def output_results(self, nodes):
        """
            Output results.

            Parameters
            ----------
            nodes : list
                    List of nodes IDs.
        """

        outfile = self.out_folder + 'theta_inf_' + str(self.flag_anomaly) +'_'+ self.end_file
        np.savez_compressed(outfile + '.npz', u=self.u_f, v=self.v_f, w=self.w_f, pibr=self.pibr_f, mupr=self.mupr_f, max_it=self.final_it,
                Q = self.Q_ij_dense_f, maxL=self.maxL, nodes=nodes)
        print(f'\nInferred parameters saved in: {outfile + ".npz"}')
        print('To load: theta=np.load(filename), then e.g. theta["u"]')


def sp_uttkrp(vals, subs, m, u, v, w):
    """
        Compute the Khatri-Rao product (sparse version).

        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
               equal to the dimension of tensor.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix u; if 2 it works with v.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.

        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
              of the membership matrix.
    """

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating u
            tmp *= (w[subs[0], k, :].astype(tmp.dtype) * v[subs[2], :].astype(tmp.dtype)).sum(axis=1)
        elif m == 2:  # we are updating v
            tmp *= (w[subs[0], :, k].astype(tmp.dtype) * u[subs[1], :].astype(tmp.dtype)).sum(axis=1)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out


def sp_uttkrp_assortative(vals, subs, m, u, v, w):
    """
        Compute the Khatri-Rao product (sparse version) with the assumption of assortativity.

        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
               equal to the dimension of tensor.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix u; if 2 it works with v.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.

        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
              of the membership matrix.
    """

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u) 
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating u
            tmp *= w[subs[0], k].astype(tmp.dtype) * v[subs[2], k].astype(tmp.dtype) 
        elif m == 2:  # we are updating v
            tmp *= w[subs[0], k].astype(tmp.dtype) * u[subs[1], k].astype(tmp.dtype)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)
        

    return out


def get_item_array_from_subs(A, ref_subs):
    """
        Get values of ref_subs entries of a dense tensor.
        Output is a 1-d array with dimension = number of non zero entries.
    """

    return np.array([A[a, i, j] for a, i, j in zip(*ref_subs)])


def preprocess(X):
    """
        Pre-process input data tensor.
        If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.

        Parameters
        ----------
        X : ndarray
            Input data (tensor).

        Returns
        -------
        X : sptensor/dtensor
            Pre-processed data. If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.
    """

    if not X.dtype == np.dtype(int).type:
        X = X.astype(int)
    if isinstance(X, np.ndarray) and is_sparse(X):
        X = sptensor_from_dense_array(X)
    else:
        X = skt.dtensor(X)

    return X


def is_sparse(X):
    """
        Check whether the input tensor is sparse.
        It implements a heuristic definition of sparsity. A tensor is considered sparse if:
        given
        M = number of modes
        S = number of entries
        I = number of non-zero entries
        then
        N > M(I + 1)

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        Boolean flag: true if the input tensor is sparse, false otherwise.
    """

    M = X.ndim
    S = X.size
    I = X.nonzero()[0].size

    return S > (I + 1) * M


def sptensor_from_dense_array(X):
    """
        Create an sptensor from a ndarray or dtensor.
        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        sptensor from a ndarray or dtensor.
    """

    subs = X.nonzero()
    vals = X[subs] 

    return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)

def transpose_tensor(A):
    '''
    Assuming the first index is for the layer, it transposes the second and third
    '''
    return np.einsum('aij->aji',A)

def plot_L(values, indices = None, k_i = 5, figsize=(7, 7), int_ticks=False, xlab='Iterations'):

    fig, ax = plt.subplots(1,1, figsize=figsize)
    #print('\n\nL: \n\n',values[k_i:])

    if indices is None:
        ax.plot(values[k_i:])
    else:
        ax.plot(indices[k_i:], values[k_i:])
    ax.set_xlabel(xlab)
    ax.set_ylabel('ELBO')
    if int_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()

    plt.tight_layout()
    plt.show()

