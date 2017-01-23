import numpy as np

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

        if i > burn_in and i%skip_rate==0:
            sigmas[:, i-skip_rate*burn_in] = sigma
            sigma_probabilities[i-skip_rate*burn_in] = prob_sigma

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
def convergenceCheck(hmm, sigmas, step_size=10):
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

def saveData(p_matrix, graph_size, no_observations, no_sample, data_type):
    np.savetxt('../data/data_states={}_observations={}_samples={}_data={}.txt'.format(3*graph_size, no_observations, no_sample, data_type), p_matrix)

def loadData(graph_size, no_observations, no_sample, data_type):
    return np.loadtxt('../data/data_states={}_observations={}_samples={}_data={}.txt'.format(3*graph_size, no_observations, no_sample, data_type))
