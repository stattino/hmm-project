from Graph import *
from hmm import *

#some code to test it
graph = Graph()
G3 = graph.genEvenGraph(8, 0)
G3 = graph.setSwitches(G3)
#print(G3)
[A, B] = graph.genSignals(G3, 6)
print(B)
print(A)

hmm = HMM()
C = hmm.generateC(G3, B[1,], 0.05)
print(C)

#print(sum(C[:,0]))
#print(sum(C[:,5]))
#print(sum(C[0,:]))
