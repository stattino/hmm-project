from Graph import *
from hmm import *
import pandas as pd

#some code to test it
graph = Graph()
G3 = graph.genEvenGraph(8, 0)
sigmas = genSigma(G3)
print(G3)
print(sigmas)
p = 0.05
[A, B] = graph.genSignals(G3, sigmas, 8, p)
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


[sigmas, sigma_prob]= sampleSigma(8, hmm, 0, 5000)

# p_vec = computeTarget(hmm, sigmas)
# print(p_vec)
p_vec = convergenceCheck(hmm, sigmas)
print(p_vec)
np.savetxt("trial.txt", p_vec)