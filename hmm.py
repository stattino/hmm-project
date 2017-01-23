import numpy as np
from Graph import *

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

 # Changes one random switch i, sigma[i], to a new setting.
def newSigma(sigma, proposal):
    N = sigma.shape[0]
    if proposal==0:
        change = np.random.randint(0, N)
        new_sigma = sigma
        new_sigma[change] = 2 if sigma[change]==3 else 3
    elif proposal==1:
        new_sigma = np.random.randint(2, 4, size=N)
    return new_sigma

# Rreturn a random set of sigmas.
def genSigma(g):
    N = g.shape[0]
    sigmas = np.random.randint(2, 4, size=N)
    return sigmas

# MCMC - Metropolis hastings sampling of sigmas
# Decorrelate samples by skipping skip_rate samples between each sample with skip_rate
# add later. Maybe have a proposal that changes >1 switch instead.
def sampleSigma(hmm, no_samples=2000, burn_in=100, skip_rate=1, proposal=0):
    N = hmm.G.shape[0]
    sigmas = np.zeros((N, no_samples))
    sigma_probabilities = np.zeros(no_samples)
    sigma = genSigma(hmm.G)     # Random starting sigma
    for i in range(0, skip_rate*(burn_in + no_samples)):
        proposed_sigma = newSigma(sigma, proposal)
        prob_sigma = computeProbability(hmm, sigma)
        prob_proposed_sigma = computeProbability(hmm, proposed_sigma)

        acceptance_rate = np.minimum(1, prob_sigma / prob_proposed_sigma)

        if acceptance_rate >= np.random.random():
            sigma = proposed_sigma
            prob_sigma = prob_proposed_sigma

        if i >= burn_in and i%skip_rate==0:
            sigmas[:, i] = sigma
            sigma_probabilities[i] = prob_sigma

    return sigmas, sigma_probabilities

 # p(O| sigma, G)
def computeProbability(hmm, sigma):
    C = hmm.genC(sigma)
    prob = sum(C[:, -1])
    return prob


 # p(s| O, G)
def computeTarget(hmm, sigmas):
    """
    takes HMM, and its observations and graph structure, and a vector of sigmas
    returns p_vec = p(s| O, G)
    """
    N = hmm.G.shape[0]
    prob_joint = np.zeros(3*N)
    for row in sigmas:
        c = hmm.genC(row)
        prob_joint += (c[:, -1])  # p(s, O| sigma, G)
    prob_joint = np.divide(prob_joint, sum(prob_joint))     # Normalization
    return prob_joint

 # Check convergence for different amount of samples
def convergenceCheck(hmm, sigmas, step_size=100):
    N, T = sigmas.shape
    prob_joint = np.zeros(3*N)
    prob_joint_steps = np.zeros((T/step_size, 3*N))
    for i in range(0,T):
        c = hmm.genC(sigmas[:, i])
        prob_joint += (c[:, -1])  # p(s, O| sigma, G)
        if (i + 1)%step_size == 0:
            prob_joint_steps[(i + 1)/step_size - 1, :] = np.divide(prob_joint, sum(prob_joint))
    return prob_joint_steps

# Returns p(s| O) for all states, for one to T observations, O_1:i, where i = 1, ..., T
def probabilitySteps(hmm, no_samples=2000, burn_in=100, skip_rate=1, proposal=0):
    T = hmm.Obs.shape[0]
    N = hmm.G.shape[0]
    probability_matrix = np.zeros((3*N, T))
    obs_copy = np.copy(hmm.Obs)
    # Computes p(s| O) for all T observations, then for T-1, etc...
    for i in range(0, T):
        if i>0: hmm.Obs = hmm.Obs[0:-1]
        sigmas, _ = sampleSigma(hmm, no_samples, burn_in, skip_rate, proposal)
        probability_matrix[:, (T - 1 - i)] = computeTarget(hmm, sigmas)
    hmm.Obs = obs_copy
    return probability_matrix