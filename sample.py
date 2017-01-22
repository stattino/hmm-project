from Graph import *
from hmm import *
import pandas as pd
from Plot import *

# initial definition
N = 8
p = 0.05
observation = 8
no_sample = 2000
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
# print(C)

[d, a] = hmm.genD(sigmas)

# print(a)
# print(d)
#print(sum(C[:,0]))
#print(sum(C[:,5]))
#print(sum(C[0,:]))


[sigmas, sigma_prob]= sampleSigma(hmm, 2000, 0)
p_vec = computeTarget(hmm, sigmas)
# print(p_vec)

p_matrix = probabilitySteps(hmm, no_sample, 0)

p_vec2 = convergenceCheck(hmm, sigmas)
print(p_matrix)
np.savetxt("trial.txt", p_matrix)

plot = Plot(N, no_sample, observation)
plot.barGraph(p_vec)
plot.lineGraphObservation(p_matrix)
plot.lineGraphSample(p_vec2)

