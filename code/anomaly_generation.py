"""
    Class for generation and management of synthetic networks with anomalies
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from scipy.stats import poisson

from scipy.optimize import brentq, root

EPS = 1e-12

class SyntNetAnomaly(object):

    def __init__(self, m = 1, N = 100, K = 2, prng = 10, avg_degree = 4., rho_anomaly = 0.1,
                structure = 'assortative', label = None, mu = None, pi = 0.8,
                flag_node_anomalies = False, rho_node=0.9,
                gamma = 0.5, eta = 0.5, L1=False,ag = 0.6, bg = 1., corr = 0., over = 0.,
                verbose = 0, folder = '../../data/input', output_parameters = False,
                output_adj = False, outfile_adj = None):

        # Set network size (node number)
        self.N = N
        # Set number of communities
        self.K = K
        # Set number of networks to be generated
        self.m = m
        # Set seed random number generator
        self.prng = prng
        # Set label (associated uniquely with the set of inputs)
        if label is not None:
            self.label = label
        else:
            self.label = ('_').join([str(N),str(K),str(avg_degree),str(flt(rho_anomaly,d=2)),str(flag_node_anomalies)])
        # Initialize data folder path
        self.folder = folder
        # Set flag for storing the parameters
        self.output_parameters = output_parameters
        # Set flag for storing the generated adjacency matrix
        self.output_adj = output_adj
        # Set name for saving the adjacency matrix
        self.outfile_adj = outfile_adj
        # Set required average degree
        self.avg_degree = avg_degree
        self.rho_anomaly = rho_anomaly 

        self.flag_node_anomalies = flag_node_anomalies

        # Set verbosity flag
        if verbose > 2 and not isinstance(verbose, int):
            raise ValueError('The verbosity parameter can only assume values in {0,1,2}!')
        self.verbose = verbose

        # Set Bernoullis parameters
        # if mu < 0 or mu > 1:
            # raise ValueError('The Binomial parameter mu has to be in [0, 1]!')

        if pi < 0 or pi > 1:
            raise ValueError('The Binomial parameter pi has to be in [0, 1]!')
        if pi == 1: pi = 1 - EPS
        if pi == 0: pi = EPS
        self.pi = pi
        
        if rho_anomaly < 0 or rho_anomaly > 1:
            raise ValueError('The rho anomaly has to be in [0, 1]!')
        
        self.ExpM = self.avg_degree * self.N * 0.5

        if self.flag_node_anomalies == True:
            if rho_node < 0 or rho_node > 1:
                raise ValueError('The rho node has to be in [0, 1]!')
            if rho_node == 1: rho_node = 1 - EPS
            if rho_node == 0: rho_node = EPS

            self.rho_node = rho_node # probability that a node is NOT anomalous
            self.mu = 1. - self.rho_node**2
            self.pi = self.rho_anomaly * self.ExpM / (self.mu * (self.N**2-self.N))
            print(self.pi,self.rho_anomaly,self.mu)
        else: # anomalies on edges
            mu = self.rho_anomaly * self.ExpM / (self.pi * (self.N**2-self.N))
            if mu == 1: mu = 1 - EPS
            if mu == 0: mu = EPS
            assert mu > 0. and mu < 1.
            self.mu = mu

        
        ### Set MT inputs
        # Set the affinity matrix structure
        if structure not in ['assortative', 'disassortative', 'core-periphery', 'directed-biased']:
            raise ValueError('The available structures for the affinity matrix w '
                             'are: assortative, disassortative, core-periphery '
                             'and directed-biased!')
        self.structure = structure

        # Set eta parameter of the  Dirichlet distribution
        if eta <= 0 and L1:
            raise ValueError('The Dirichlet parameter eta has to be positive!')
        self.eta = eta
        # Set alpha parameter of the Gamma distribution
        if ag <= 0 and not L1:
            raise ValueError('The Gamma parameter alpha has to be positive!')
        self.ag = ag
        # Set beta parameter of the Gamma distribution
        if bg <= 0 and not L1:
            raise ValueError('The Gamma parameter beta has to be positive!')
        self.bg = bg
        # Set u,v generation preference
        self.L1 = L1
        # Set correlation between u and v synthetically generated
        if (corr < 0) or (corr > 1):
            raise ValueError('The correlation parameter has to be in [0, 1]!')
        self.corr = corr
        # Set fraction of nodes with mixed membership
        if (over < 0) or (over > 1):
                raise ValueError('The overlapping parameter has to be in [0, 1]!')
        self.over = over


    def anomaly_network_PB(self, parameters = None):
        """
            Generate a directed, possibly weighted network by using the anomaly model Poisson-Bernoulli
            Steps:
                1. Generate or load the latent variables Z_ij.
                2. Extract A_ij entries (network edges) from a Poisson distribution if Z_ij=0; from a Brenoulli(mu) id Z_ij=0
            INPUT
            ----------
            parameters : object
                         Latent variables z, s, u, v and w.
            OUTPUT
            ----------
            G : Digraph
                DiGraph NetworkX object. Self-loops allowed.
        """

        # Set seed random number generator
        prng = np.random.RandomState(self.prng)

        ### Latent variables
        if parameters is None:
            # Generate latent variables
            self.z, self.u, self.v, self.w = self._generate_lv(prng)
        else:
            # Set latent variables
            self.z, self.u, self.v, self.w = parameters


        ### Network generation
        G = nx.DiGraph()
        for i in range(self.N):
            G.add_node(i)

        # Compute M_ij
        M = np.einsum('ik,jq->ijkq', self.u, self.v)
        M = np.einsum('ijkq,kq->ij', M, self.w)

        # M[self.z == 1] = 0
        # Set c sparsity parameter
        # c = ( float(self.N * self.avg_degree * 0.5) - self.mu * self.pi ) / ((1-self.mu) * binaryM.sum() )
        c = brentq(eq_c, EPS, 20, args = (M, self.N,self.ExpM,self.rho_anomaly,self.mu))

        self.w *= c

        # print(c,(1 - self.mu)*((self.N**2-self.N) - np.sum(np.exp(-c*M))) , self.ExpM * (1-self.rho_anomaly))

        '''
        Build network
        '''
        A = prng.poisson(c * M)
        A[A>0] = 1 # binarize the adjacency matrix
        np.fill_diagonal(A, 0)
        G0 = nx.to_networkx_graph(A, create_using=nx.DiGraph)

        # binary anomaly
        # A[self.z.nonzero()] = prng.binomial(1,self.pi,self.z.count_nonzero())
        # weighted anomaly
        A[self.z.nonzero()] = prng.poisson(self.pi * self.z.count_nonzero())
        A[A>0] = 1 # binarize the adjacency matrix
        np.fill_diagonal(A, 0)

        G = nx.to_networkx_graph(A, create_using=nx.DiGraph)

        '''
        Network post-processing
        ''' 
        nodes = list(G.nodes())
        A = nx.to_scipy_sparse_matrix(G, nodelist=nodes, weight='weight')

        # Keep largest connected component
        Gc = max(nx.weakly_connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

        nodes = list(G.nodes())
        self.N = len(nodes)

        G0 = G0.subgraph(nodes)
        A = nx.to_scipy_sparse_matrix(G, nodelist=nodes, weight='weight') #
        A0 = nx.to_scipy_sparse_matrix(G0, nodelist=nodes, weight='weight') #
        try:
            self.z = np.take(self.z, nodes, 1)
            self.z = np.take(self.z, nodes, 0)
        except:
            self.z = self.z[:,nodes]
            self.z = self.z[nodes]

        if self.u is not None:
            self.u = self.u[nodes]
            self.v = self.v[nodes]
        self.N = len(nodes)

        if self.verbose > 0:
            ave_deg = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)
            print(f'Number of nodes: {G.number_of_nodes()} \n'
                f'Number of edges: {G.number_of_edges()}')
            print(f'Average degree (2E/N): {ave_deg}')
            print(f'rho_anomaly: { A[self.z.nonzero()].sum() / float(G.number_of_edges()) }')

        if self.output_parameters:
            self._output_results(nodes)

        if self.output_adj:
            self._output_adjacency(G, outfile = self.outfile_adj)

        if self.verbose == 2:
            self._plot_A(A)
            self._plot_A(A0,title='A before anomaly')
            self._plot_A(self.z,title='Anomaly matrix Z')
            if M is not None: self._plot_M(M)

        return G,G0

    def _generate_lv(self, prng = 42):
        """
            Generate z, u, v, w latent variables.
            INPUT
            ----------
            prng : int
                   Seed for the random number generator.
            OUTPUT
            ----------
            z : Numpy array
                Matrix NxN of model indicators (binary).

            u : Numpy array
                Matrix NxK of out-going membership vectors, positive element-wise.
                With unitary L1 norm computed row-wise.

            v : Numpy array
                Matrix NxK of in-coming membership vectors, positive element-wise.
                With unitary L1 norm computed row-wise.

            w : Numpy array
                Affinity matrix KxK. Possibly None if in pure SpringRank.
                Element (k,h) gives the density of edges going from the nodes
                of group k to nodes of group h.
        """
        # Generate z through binomial distribution
        # z = prng.binomial(1, self.mu, (self.N, self.N))
        # z = sparse.coo_matrix(z)
        if self.flag_node_anomalies == True:
            self.sigma = prng.binomial(1,self.rho_node,self.N)
            z = 1 - np.einsum('i,j->ij',self.sigma,self.sigma)
            z = sparse.csr_matrix(z)
        else:
            # if self.mu-1./self.N < 0:
            if self.mu < 0:
                density = EPS
            else:
                # density = self.mu-1./self.N
                density = self.mu
            z = sparse.random(self.N,self.N, density=density, data_rvs=np.ones)
            upper_z = sparse.triu(z) 
            z = upper_z + upper_z.T 
        # z -= sparse.diags(z.diagonal())
            # z.setdiag(0)

        # Generate u, v for overlapping communities
        u, v = membership_vectors(prng, self.L1, self.eta, self.ag, self.bg, self.K,
                                 self.N, self.corr, self.over)
        # Generate w
        w = affinity_matrix(self.structure, self.N, self.K, self.avg_degree)
        return z, u, v, w

    def _output_results(self, nodes):
        """
            Output results in a compressed file.
            INPUT
            ----------
            nodes : list
                    List of nodes IDs.
        """ 
        output_parameters = self.folder + 'theta_' + self.label + '_' + str(self.prng)
        # print(self.z.count_nonzero())
        if self.flag_node_anomalies == True:
            np.savez_compressed(output_parameters + '.npz', z=self.z.todense(), u=self.u, v=self.v,
                            w=self.w, mu=self.mu, pi=self.pi, nodes=nodes,sigma=self.sigma)
        else:
            np.savez_compressed(output_parameters + '.npz', z=self.z.todense(), u=self.u, v=self.v,
                            w=self.w, mu=self.mu, pi=self.pi, nodes=nodes)
        if self.verbose:
            print()
            print(f'Parameters saved in: {output_parameters}.npz')
            print('To load: theta=np.load(filename), then e.g. theta["u"]')

    def _output_adjacency(self, G, outfile = None):
        """
            Output the adjacency matrix. Default format is space-separated .csv
            with 3 columns: node1 node2 weight
            INPUT
            ----------
            G: Digraph
               DiGraph NetworkX object.
            outfile: str
                     Name of the adjacency matrix.
        """
        if outfile is None:
            outfile = 'syn_' + self.label + '_' + str(self.prng)  + '.dat'

        edges = list(G.edges(data=True))
        try:
            data = [[u, v, d['weight']] for u, v, d in edges]
        except:
            data = [[u, v, 1] for u, v, d in edges]

        df = pd.DataFrame(data, columns=['source', 'target', 'w'], index=None)
        df.to_csv(self.folder + outfile, index=False, sep=' ')
        if self.verbose:
            print(f'Adjacency matrix saved in: {self.folder + outfile}')

    def _plot_A(self, A, cmap = 'PuBuGn',title='Adjacency matrix'):
        """
            Plot the adjacency matrix produced by the generative algorithm.
            INPUT
            ----------
            A : Scipy array
                Sparse version of the NxN adjacency matrix associated to the graph.
            cmap : Matplotlib object
                Colormap used for the plot.
        """
        Ad = A.todense()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(Ad, cmap = plt.get_cmap(cmap))
        ax.set_title(title, fontsize = 15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()

    def _plot_Z(self, cmap = 'PuBuGn'):
        """
            Plot the anomaly matrix produced by the generative algorithm.
            INPUT
            ----------
            cmap : Matplotlib object
                Colormap used for the plot.
        """
        # assert isinstance(self.z, sparse.csr.csr_matrix)
        # z_dense = self.z.toarray()
        # assert isinstance(z_dense, sparse.csr.csr_matrix) == False
        # Ad = self.z.todense()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(self.z, cmap = plt.get_cmap(cmap))
        ax.set_title('Anomaly matrix', fontsize = 15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()


    def _plot_M(self, M, cmap = 'PuBuGn',title='MT means matrix'):
        """
            Plot the M matrix produced by the generative algorithm. Each entry is the
            poisson mean associated to each couple of nodes of the graph.
            INPUT
            ----------
            M : Numpy array
                NxN M matrix associated to the graph. Contains all the means used
                for generating edges.
            cmap : Matplotlib object
                Colormap used for the plot.
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(M, cmap = plt.get_cmap(cmap))
        ax.set_title(title, fontsize = 15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()




def membership_vectors(prng = 10, L1 = False, eta = 0.5, alpha = 0.6, beta = 1, K = 2, N = 100, corr = 0., over = 0.):
    """
        Compute the NxK membership vectors u, v using a Dirichlet or a Gamma distribution.
        INPUT
        ----------
        prng: Numpy Random object
              Random number generator container.
        L1 : bool
             Flag for parameter generation method. True for Dirichlet, False for Gamma.
        eta : float
              Parameter for Dirichlet.
        alpha : float
            Parameter (alpha) for Gamma.
        beta : float
            Parameter (beta) for Gamma.
        N : int
            Number of nodes.
        K : int
            Number of communities.
        corr : float
               Correlation between u and v synthetically generated.
        over : float
               Fraction of nodes with mixed membership.
        OUTPUT
        -------
        u : Numpy array
            Matrix NxK of out-going membership vectors, positive element-wise.
            Possibly None if in pure SpringRank or pure Multitensor.
            With unitary L1 norm computed row-wise.

        v : Numpy array
            Matrix NxK of in-coming membership vectors, positive element-wise.
            Possibly None if in pure SpringRank or pure Multitensor.
            With unitary L1 norm computed row-wise.
    """
    # Generate equal-size unmixed group membership
    size = int(N / K)
    u = np.zeros((N, K))
    v = np.zeros((N, K))
    for i in range(N):
        q = int(math.floor(float(i) / float(size)))
        if q == K:
            u[i:, K - 1] = 1.
            v[i:, K - 1] = 1.
        else:
            for j in range(q * size, q * size + size):
                u[j, q] = 1.
                v[j, q] = 1.
    # Generate mixed communities if requested
    if over != 0.:
        overlapping = int(N * over)  # number of nodes belonging to more communities
        ind_over = np.random.randint(len(u), size=overlapping)
        if L1:
            u[ind_over] = prng.dirichlet(eta * np.ones(K), overlapping)
            v[ind_over] = corr * u[ind_over] + (1. - corr) * prng.dirichlet(eta * np.ones(K), overlapping)
            if corr == 1.:
                assert np.allclose(u, v)
            if corr > 0:
                v = normalize_nonzero_membership(v)
        else:
            u[ind_over] = prng.gamma(alpha, 1. / beta, size=(N, K))
            v[ind_over] = corr * u[ind_over] + (1. - corr) * prng.gamma(alpha, 1. / beta, size=(overlapping, K))
            u = normalize_nonzero_membership(u)
            v = normalize_nonzero_membership(v)
    return u, v

def affinity_matrix(structure = 'assortative', N = 100, K = 2, avg_degree = 4., a = 0.1, b = 0.3):
    """
        Compute the KxK affinity matrix w with probabilities between and within groups.
        INPUT
        ----------
        structure : string
                    Structure of the network.
        N : int
            Number of nodes.
        K : int
            Number of communities.
        a : float
            Parameter for secondary probabilities.
        OUTPUT
        -------
        p : Numpy array
            Array with probabilities between and within groups. Element (k,h)
            gives the density of edges going from the nodes of group k to nodes of group h.
    """

    b *= a
    p1 = avg_degree * K / N

    if structure == 'assortative':
        p = p1 * a * np.ones((K,K))  # secondary-probabilities
        np.fill_diagonal(p, p1 * np.ones(K))  # primary-probabilities

    elif structure == 'disassortative':
        p = p1 * np.ones((K,K))   # primary-probabilities
        np.fill_diagonal(p, a * p1 * np.ones(K))  # secondary-probabilities

    elif structure == 'core-periphery':
        p = p1 * np.ones((K,K))
        np.fill_diagonal(np.fliplr(p), a * p1)
        p[1, 1] = b * p1

    elif structure == 'directed-biased':
        p = a * p1 * np.ones((K,K))
        p[0, 1] = p1
        p[1, 0] = b * p1

    return p

def normalize_nonzero_membership(u):
    """
        Given a matrix, it returns the same matrix normalized by row.
        INPUT
        ----------
        u: Numpy array
           Numpy Matrix.
        OUTPUT
        -------
        The matrix normalized by row.
    """

    den1 = u.sum(axis=1, keepdims=True)
    nzz = den1 == 0.
    den1[nzz] = 1.

    return u / den1

def eq_c(c,M, N,E,rho_a,mu):

    return np.sum(np.exp(-c*M)) - (N**2 -N) + E * (1-rho_a) / (1-mu)

def flt(x,d=1):
    return round(x, d)
