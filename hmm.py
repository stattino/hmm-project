import numpy as np

class HMM:
    def genC(self, G, Obs, p):
        """
        Creates the matrix C, where C[i,j] = c(s=s_i, t=j)
        G = graph structure
        Obs = [o_1, ..., o_T]
        p = noise level on switch
        """
        T = Obs.shape[0]
        N = G.shape[0]
        C = np.zeros([3*N, T], float)
        C[:,0] = 1/(3*N)
        for j in range(1,T): # columns
            for i in range(0,N): # rows
                # Find indexes in graph for vertices from where 0/L/R comes.
                temp_g = np.copy(G[i,])
                temp_g[i] = 0

                # Index in G-matrix for the corresponding vertex
                g_0 = np.nonzero(temp_g==1)[0][0] #departure label of edge
                g_L = np.nonzero(temp_g==2)[0][0]
                g_R = np.nonzero(temp_g==3)[0][0]

                # Index in C-matrix for the vertex and edge
                zero = 3*g_0 + G[g_0, i] - 1 # G[g_0, i] = 1, 2, 3 depending on 0/L/R
                left = 3*g_L + G[g_L, i] - 1
                right = 3*g_R + G[g_R, i] - 1

                # switch setting of vertex
                sw = G[i, i]

                # iteratively filling the matrix depending on switch settings and obs.
                if Obs[j] == 1: # o_j = 0
                    C[3 * i, j] = (1 - p) * (C[right, j-1] + C[left, j-1])
                    if sw == 2: # sw = L
                        C[3 * i + 1, j] = p * C[zero, j - 1]
                        C[3 * i + 2, j] = 0
                    elif sw == 3: # sw = R
                        C[3 * i + 1, j] = 0
                        C[3 * i + 2, j] = p * C[zero, j - 1]
                elif Obs[j] == 2: # o_j = L
                    C[3 * i, j] = p * (C[right, j-1] + C[left, j-1])
                    if sw == 2:
                        C[3 * i + 1, j] = (1 - p) * C[zero, j - 1]
                        C[3 * i + 2, j] = 0
                    elif sw == 3:
                        C[3 * i + 1, j] = 0
                        C[3 * i + 2, j] = p * C[zero, j - 1]
                elif Obs[j] == 3: # o_j = R
                    C[3 * i, j] = p * (C[right, j-1] + C[left, j-1])
                    if sw == 2:
                        C[3 * i + 1, j] = p * C[zero, j - 1]
                        C[3 * i + 2, j] = 0
                    elif sw == 3:
                        C[3 * i + 1, j] = 0
                        C[3 * i + 2, j] = (1 - p) * C[zero, j - 1]
        return C

    def genD(self, G, Obs, p):
        """
        returns d, and d_index matrices.
        G = graph structure
        Obs = [o_1, ..., o_T]
        p = noise level on switch

        d = probability of the most probable previous state given the observations and switches
        d_index = indexes of the most probable previous state given the observations and switches.
        key to the index:
            e = (index%3) == (0, 1, 2) where 0 = 0, 1= L, 2=R i.e. labeling of edge
            (index-e)/3 = (0, 1, ... N-1) = vertex number in graph

        """
        T = Obs.shape[0]
        N = G.shape[0]
        d = np.zeros([3 * N, T], float)
        d[:, 0] = 1 / (3 * N) # Initialized values. Multiply with the probability of observing them?
        d_index = np.zeros([3*N, T], float) # Indexes of optimal d:s

        for j in range(1, T):  # columns
            for i in range(0, N):  # rows
                temp_g = np.copy(G[i,])
                temp_g[i] = 0

                # Index in G-matrix for the corresponding vertex
                g_0 = np.nonzero(temp_g == 1)[0][0]  # departure label of edge
                g_L = np.nonzero(temp_g == 2)[0][0]
                g_R = np.nonzero(temp_g == 3)[0][0]

                # Index in C-matrix for the vertex and edge
                zero = 3 * g_0 + G[g_0, i] - 1  # G[g_0, i] = 1, 2, 3 depending on 0/L/R
                left = 3 * g_L + G[g_L, i] - 1
                right = 3 * g_R + G[g_R, i] - 1

                # switch setting of vertex
                sw = G[i, i]

                if Obs[j] == 1:  # o_j = 0
                    d[3 * i, j] = (1 - p) * np.maximum(d[right, j - 1], d[left, j - 1] )
                    d_index[3 * i, j] = np.where( d[right, j - 1] >= d[left, j - 1] , right, left)
                    if sw == 2:  # sw = L
                        d[3 * i + 1, j] = p * d[zero, j - 1]
                        d[3 * i + 2, j] = 0
                        d_index[3 * i +1, j] = zero
                        d_index[3 * i +2 , j] = np.NaN
                    elif sw == 3:  # sw = R
                        d[3 * i + 1, j] = 0
                        d[3 * i + 2, j] = p * d[zero, j - 1]
                        d_index[3 * i +1, j] = np.NaN
                        d_index[3 * i +2 , j] = zero
                elif Obs[j] == 2:  # o_j = L
                    d[3 * i, j] = p * np.maximum(d[right, j - 1], d[left, j - 1])
                    d_index[3 * i, j] = np.where( d[right, j - 1] >= d[left, j - 1] , right, left)
                    if sw == 2:
                        d[3 * i + 1, j] = (1 - p) * d[zero, j - 1]
                        d[3 * i + 2, j] = 0
                        d_index[3 * i +1, j] = zero
                        d_index[3 * i +2, j] = np.NaN
                    elif sw == 3:
                        d[3 * i + 1, j] = 0
                        d[3 * i + 2, j] = p * d[zero, j - 1]
                        d_index[3 * i +1, j] = np.NaN
                        d_index[3 * i +2 , j] = zero
                elif Obs[j] == 3:  # o_j = R
                    d[3 * i, j] = p * np.maximum(d[right, j - 1], d[left, j - 1])
                    d_index[3 * i, j] = np.where( d[right, j - 1] >= d[left, j - 1] , right, left)
                    if sw == 2:
                        d[3 * i + 1, j] = p * d[zero, j - 1]
                        d[3 * i + 2, j] = 0
                        d_index[3 * i +1, j] = zero
                        d_index[3 * i +2 , j] = np.NaN
                    elif sw == 3:
                        d[3 * i + 1, j] = 0
                        d[3 * i + 2, j] = (1 - p) * d[zero, j - 1]
                        d_index[3 * i +1 , j] = np.NaN
                        d_index[3 * i +2 , j] = zero

        return d, d_index