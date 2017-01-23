import numpy as np

class HMM():
    def __init__(self, Obs, G, p):
        self.Obs = Obs
        self.G = G
        self.p = p

    def genC(self, sigma):
        """
        Creates the matrix C, where C[i,j] = c(s=s_i, t=j) given the current
        switch setting sigma.
        """
        T = self.Obs.shape[0]
        N = self.G.shape[0]
        C = np.zeros((3 * N, T))
        C[:,0] = 1 / (3*N)
        for j in range(1, T): # columns
            for i in range(0, N): # rows
                # Departure label of edges.
                g_0 = np.nonzero(self.G[i, ]==1)[0][0]
                g_L = np.nonzero(self.G[i, ]==2)[0][0]
                g_R = np.nonzero(self.G[i, ]==3)[0][0]

                # Index in C-matrix for the vertex and edge.
                zero = 3*g_0 + self.G[g_0, i] - 1 # G[g_0, i] = 1, 2, 3 depending on 0/L/R
                left = 3*g_L + self.G[g_L, i] - 1
                right = 3*g_R + self.G[g_R, i] - 1

                sw = sigma[i] # switch setting

                # iteratively filling the matrix depending on switch settings and obs.
                if self.Obs[j] == 1: # o_j = 0
                    C[3*i, j] = (1 - self.p)*(C[right, j - 1] + C[left, j - 1])
                    if sw == 2: # sw = L
                        C[3*i + 1, j] = self.p*C[zero, j - 1]
                        C[3*i + 2, j] = 0
                    elif sw == 3: # sw = R
                        C[3*i + 1, j] = 0
                        C[3*i + 2, j] = self.p*C[zero, j - 1]
                elif self.Obs[j] == 2: # o_j = L
                    C[3*i, j] = self.p*(C[right, j - 1] + C[left, j - 1])
                    if sw == 2:
                        C[3*i + 1, j] = (1 - self.p)*C[zero, j - 1]
                        C[3*i + 2, j] = 0
                    elif sw == 3:
                        C[3*i + 1, j] = 0
                        C[3*i + 2, j] = self.p*C[zero, j - 1]
                elif self.Obs[j] == 3: # o_j = R
                    C[3*i, j] = self.p*(C[right, j - 1] + C[left, j - 1])
                    if sw == 2:
                        C[3*i + 1, j] = self.p*C[zero, j - 1]
                        C[3*i + 2, j] = 0
                    elif sw == 3:
                        C[3*i + 1, j] = 0
                        C[3*i + 2, j] = (1 - self.p)*C[zero, j - 1]
        return C

    def genD(self, sigma):
        """
        returns d, and d_index matrices given the switch setting sigma.
        d = probability of the most probable previous state given the observations and switches
        d_index = indexes of the most probable previous state given the observations and switches.
        key to the index:
            e = (index%3) == (0, 1, 2) where 0 = 0, 1= L, 2=R i.e. labeling of edge
            (index-e)/3 = (0, 1, ... N-1) = vertex number in graph
        """
        T = self.Obs.shape[0]
        N = self.G.shape[0]
        d = np.zeros([3 * N, T], float)
        d[:, 0] = 1 / (3 * N) # Initialized values. Multiply with the probability of observing them?
        d_index = np.zeros([3*N, T], float) # Indexes of optimal d:s

        for j in range(1, T):  # columns
            for i in range(0, N):  # rows
                # Index in G-matrix for the corresponding vertex
                g_0 = np.nonzero(self.G[i, ]==1)[0][0]
                g_L = np.nonzero(self.G[i, ]==2)[0][0]
                g_R = np.nonzero(self.G[i, ]==3)[0][0]

                # Index in C-matrix for the vertex and edge
                zero = 3*g_0 + self.G[g_0, i] - 1  # G[g_0, i] = 1, 2, 3 depending on 0/L/R
                left = 3*g_L + self.G[g_L, i] - 1
                right = 3*g_R + self.G[g_R, i] - 1

                sw = sigma[i]

                if self.Obs[j] == 1:
                    d[3*i, j] = (1 - self.p)*np.maximum(d[right, j - 1], d[left, j - 1] )
                    d_index[3*i, j] = np.where(d[right, j - 1] >= d[left, j - 1] ,right , left)
                    if sw == 2:
                        d[3*i + 1, j] = self.p*d[zero, j - 1]
                        d[3*i + 2, j] = 0
                        d_index[3*i + 1, j] = zero
                        d_index[3*i + 2, j] = np.NaN
                    elif sw == 3:
                        d[3*i + 1, j] = 0
                        d[3*i + 2, j] = self.p*d[zero, j - 1]
                        d_index[3*i + 1, j] = np.NaN
                        d_index[3*i + 2, j] = zero
                elif self.Obs[j] == 2:
                    d[3*i, j] = self.p*np.maximum(d[right, j - 1], d[left, j - 1])
                    d_index[3*i, j] = np.where(d[right, j - 1] >= d[left, j - 1] ,right , left)
                    if sw == 2:
                        d[3*i + 1, j] = (1 - self.p)*d[zero, j - 1]
                        d[3*i + 2, j] = 0
                        d_index[3*i + 1, j] = zero
                        d_index[3*i + 2, j] = np.NaN
                    elif sw == 3:
                        d[3*i + 1, j] = 0
                        d[3*i + 2, j] = self.p*d[zero, j - 1]
                        d_index[3*i + 1, j] = np.NaN
                        d_index[3*i + 2, j] = zero
                elif self.Obs[j] == 3:
                    d[3*i, j] = self.p*np.maximum(d[right, j - 1], d[left, j - 1])
                    d_index[3*i, j] = np.where(d[right, j - 1] >= d[left, j - 1] ,right , left)
                    if sw == 2:
                        d[3*i + 1, j] = self.p*d[zero, j - 1]
                        d[3*i + 2, j] = 0
                        d_index[3*i + 1, j] = zero
                        d_index[3*i + 2, j] = np.NaN
                    elif sw == 3:
                        d[3*i + 1, j] = 0
                        d[3*i + 2, j] = (1 - self.p)*d[zero, j - 1]
                        d_index[3*i + 1, j] = np.NaN
                        d_index[3*i + 2, j] = zero
        return d, d_index
