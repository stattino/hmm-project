import numpy as np

class HMM:

    def generateC(self, G, Obs, p):
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
                if Obs[j] == 0: # o_j = 0
                    C[3 * i, j] = (1 - p) * (C[right, j-1] + C[left, j-1])
                    if sw == 2: # sw = L
                        C[3 * i + 1, j] = p * C[zero, j - 1]
                        C[3 * i + 2, j] = 0
                    elif sw == 3: # sw = R
                        C[3 * i + 1, j] = 0
                        C[3 * i + 2, j] = p * C[zero, j - 1]
                elif Obs[j] == 1: # o_j = L
                    C[3 * i, j] = p * (C[right, j-1] + C[left, j-1])
                    if sw == 2:
                        C[3 * i + 1, j] = (1 - p) * C[zero, j - 1]
                        C[3 * i + 2, j] = 0
                    elif sw == 3:
                        C[3 * i + 1, j] = 0
                        C[3 * i + 2, j] = p * C[zero, j - 1]
                elif Obs[j] == 2: # o_j = R
                    C[3 * i, j] = p * (C[right, j-1] + C[left, j-1])
                    if sw == 2:
                        C[3 * i + 1, j] = p * C[zero, j - 1]
                        C[3 * i + 2, j] = 0
                    elif sw == 3:
                        C[3 * i + 1, j] = 0
                        C[3 * i + 2, j] = (1 - p) * C[zero, j - 1]
        return C


