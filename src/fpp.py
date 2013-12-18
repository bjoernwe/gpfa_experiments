import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.spatial.distance

import mdp


class FPPBase(mdp.Node):

    data = None

    def __init__(self, output_dim, k=10, normalized_laplacian=True, neighbor_edges=True, input_dim=None, dtype=None):
        super(FPPBase, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.normalized_laplacian = normalized_laplacian
        self.neighbor_edges = neighbor_edges


    def _train(self, x):
        if self.data is None:
            self.data = x
        else:
            np.vstack((self.data, x))


    def _stop_training(self):
        """
        Calculates the graph Laplacian.
        """

        # initialize weight matrix W
        N, _ = self.data.shape
        W = scipy.sparse.dok_matrix((N, N))

        # pairwise distances of data points
        distances = scipy.spatial.distance.pdist(self.data)
        distances = scipy.spatial.distance.squareform(distances)

        # future-preserving graph
        for s in range(N-1):
            neighbors = np.argsort(distances[s])
            for t in neighbors[0:self.k+1]:
                if s != t and t+1 < N:  # no self-connections
                    W[s+1,t+1] = 1
                    W[t+1,s+1] = 1

        # k-nearest-neighbor graph for regularization
        if self.neighbor_edges:
            for i in range(N):
                neighbors = np.argsort(distances[i])
                for j in neighbors[0:self.k+1]:
                    if i != j:
                        W[i,j] = 1
                        W[j,i] = 1

        # graph Laplacian
        d = W.sum(axis=1).T
        if self.normalized_laplacian:
            d_inv = 1./d
            D_inv = scipy.sparse.dia_matrix((d_inv, 0), shape=(N, N))
            W = D_inv.dot(W)
            D = scipy.sparse.eye(N, N)
        else:
            D = scipy.sparse.dia_matrix((d, 0), shape=(N, N))

        # keep matrices
        L = D - W
        self.W = W
        self.D = D
        self.L = L
        return



class FPP(FPPBase):

    def __init__(self, output_dim, k=10, normalized_laplacian=True, neighbor_edges=True, input_dim=None, dtype=None):
        FPPBase.__init__(self, output_dim=output_dim, k=k, normalized_laplacian=normalized_laplacian, neighbor_edges=neighbor_edges, input_dim=input_dim, dtype=dtype)


    def _stop_training(self):
        FPPBase._stop_training(self)
        if self.normalized_laplacian:
            # find largest eigenvectors of connection matrix
            E, U = scipy.sparse.linalg.eigs(self.W, k=self.output_dim+1, which='LR')
            self.E = E.real
            self.U = U.real
        else:
            # find smallest eigenvectors, respectively largest (sigma - eig)
            self.E, self.U = scipy.sparse.linalg.eigsh(self.L, M=self.D, sigma=0.0, k=self.output_dim+1, which='LA')

        self.knn = []
        for i in range(self.output_dim):
            knn = mdp.nodes.KNNClassifier(k=1)
            knn.train(self.data, labels=self.U[:,i+1])
            self.knn.append(knn)


    def _execute(self, x):
        N, _ = x.shape
        result = np.zeros((N, self.output_dim))
        for i in range(self.output_dim):
            result[:,i] = self.knn[i].label(x)
        return result



class FPPLinear(mdp.Node):

    def __init__(self, output_dim, k=10, iterations=1, preserve_past=True, neighbor_graph=False, input_dim=None, dtype=None):
        super(FPPLinear, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.iterations = iterations
        self.preserve_past = preserve_past
        self.neighbor_graph = neighbor_graph
        self.L = None
        self.D = None
        self.sigma = 3.0
        return
    
    
    def _kernel(self, d):
        print d
        return np.exp(-.5*(d/self.sigma)**2) / (self.sigma * np.sqrt(2. * np.pi))
    
    
    def _train(self, x):

        # number of samples
        N, _ = x.shape
        
        # from y we calculate the euclidean distances
        # after the first iteration it contains the projected data
        y = x
        
        # run algorithm several times 
        for l in range(self.iterations):

            # initialize weight matrix W
            W = scipy.sparse.dok_matrix((N, N))
        
            # pairwise distances of data points
            distances = scipy.spatial.distance.pdist(y)
            distances = scipy.spatial.distance.squareform(distances)
    
            # future-preserving graph
            for s in range(N-1):
                neighbors = np.argsort(distances[s])
                for t in neighbors[0:self.k+1]:
                    if s != t: # no self-connections
                        if s+1 < N and t+1 < N:
                            #d = self._kernel(distances[s+1, t+1])
                            #d = distances[s+u, t+u]
                            #print d
                            W[s+1,t+1] = 1#./d
                            W[t+1,s+1] = 1#./d
    
            # past-preserving graph
            if self.preserve_past:
                for s in range(1, N):
                    neighbors = np.argsort(distances[s])
                    for t in neighbors[0:self.k+1]:
                        if s != t: # no self-connections
                            if s-1 > 0 and t-1 >= 0:
                                #d = self._kernel(distances[s-1, t-1])
                                #d = distances[s-u, t-u]
                                W[s-1,t-1] = 1#./d
                                W[t-1,s-1] = 1#./d
                            
            # k-nearest-neighbor graph for regularization
            if self.neighbor_graph:
                for i in range(N):
                    neighbors = np.argsort(distances[i])
                    for j in neighbors[0:self.k+1]:
                        if i != j:
                            W[i,j] = 1
                            W[j,i] = 1
    
            # graph Laplacian
            d = W.sum(axis=1).T
            d[d==0] = float('inf') 
            D = scipy.sparse.dia_matrix((d, 0), shape=(N, N))
            L = D - W
    
            # projected graph laplacian
            D2 = x.T.dot(D.dot(x))
            L2 = x.T.dot(L.dot(x))

            # (if not the last iteration:) solve and project
            if l < self.iterations-1:
                #E, U = scipy.sparse.linalg.eigs(L2, M=D2, k=self.output_dim, which='SR')
                #E, U = np.linalg.eigh(L2)
                E, U = scipy.linalg.eigh(a=L2, b=D2)
                (E, U) = (E.real, U.real)
                print min(E), max(E)
                assert 0 not in E
                assert float('nan') not in E
                assert float('nan') not in U 
                for i in range(len(E)):
                    U[:,i] = U[:,i] / E[i]**2
                y = x.dot(U)

        # add chunk result to global result
        if self.L is None:
            self.L = L2
            self.D = D2
        else:
            self.L += L2
            self.D += D2

        return


    def _stop_training(self):
        # calculate the eigen-vectors
        #E, U = scipy.sparse.linalg.eigs(self.L, M=self.D, sigma=0.0, k=self.output_dim, which='LR')
        E, U = scipy.sparse.linalg.eigs(self.L, M=self.D, k=self.output_dim, which='SR')
        self.E = E.real
        self.U = U.real
        return


    def _execute(self, x):
        return x.dot(self.U)



class GraphSFA(mdp.Node):

    def __init__(self, output_dim, k=10, normalized_laplacian=True, neighbor_edges=False, input_dim=None, dtype=None):
        super(GraphSFA, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.normalized_laplacian = normalized_laplacian
        self.neighbor_edges = neighbor_edges
        self.L = None
        self.D = None
        return
    
    
    def _train(self, x):

        # initialize weight matrix W
        N, _ = x.shape
        W = scipy.sparse.dok_matrix((N, N))

        # pairwise distances of data points
        distances = scipy.spatial.distance.pdist(x)
        distances = scipy.spatial.distance.squareform(distances)

        # future-preserving graph
        for s in range(N-1):
            neighbors = np.argsort(distances[s])
            for t in neighbors[0:self.k+1]:
                if t+1 < N:  # no self-connections
                    W[s,t+1] = 1
                    W[t+1,s] = 1

        # k-nearest-neighbor graph for regularization
        if self.neighbor_edges:
            for i in range(N):
                neighbors = np.argsort(distances[i])
                for j in neighbors[0:self.k+1]:
                    if i != j:
                        W[i,j] = 1
                        W[j,i] = 1

        # graph Laplacian
        d = W.sum(axis=1).T
        d[d==0] = float('inf') 
        if self.normalized_laplacian:
            d_inv = 1./d
            D_inv = scipy.sparse.dia_matrix((d_inv, 0), shape=(N, N))
            W = D_inv.dot(W)
            D = scipy.sparse.eye(N, N)
        else:
            D = scipy.sparse.dia_matrix((d, 0), shape=(N, N))
        L = D - W

        # projected graph laplacian
        D2 = x.T.dot(D.dot(x))
        L2 = x.T.dot(L.dot(x))

        # add chunk result to global result
        if self.L is None:
            self.L = L2
            self.D = D2
        else:
            self.L += L2
            self.D += D2

        return


    def _stop_training(self):
        # calculate the eigen-vectors
        #E, U = scipy.sparse.linalg.eigs(self.L, M=self.D, sigma=0.0, k=self.output_dim, which='LR')
        E, U = scipy.sparse.linalg.eigs(self.L, M=self.D, k=self.output_dim, which='SR')
        self.E = E.real
        self.U = U.real
        return


    def _execute(self, x):
        return x.dot(self.U)

