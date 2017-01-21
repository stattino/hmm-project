from Graph import *
from hmm import *
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# graph setitng
plt.style.use('ggplot')
font = {'family' : 'meiryo'}
matplotlib.rc('font', **font)

# define init variable
N = 8
p = 0.05
no_sample = 5000
observation = 2

#some code to test it
graph = Graph()
G3 = graph.genEvenGraph(N, 0)
sigmas = genSigma(G3)
print(G3)
print(sigmas)
[A, B] = graph.genSignals(G3, sigmas, observation, p)
print(G3)
print(B)
print(A)

hmm = HMM(B[1,], G3, p)
C = hmm.genC(sigmas)
print(C)

[d, a] = hmm.genD(sigmas)

print(a)
print(d)
#print(sum(C[:,0]))
#print(sum(C[:,5]))
#print(sum(C[0,:]))


[sigmas, sigma_prob]= sampleSigma(N, hmm, 0, no_sample)

p_vec = computeTarget(hmm, sigmas)
print(p_vec)

#plot probability (bar)
data = pd.Series(p_vec)
data.plot(kind="bar", color="k", alpha=0.7)
plt.savefig('./plots/bar/nodes_{}-observation_{}-no_sample_{}.png'.format(N, observation, no_sample))


