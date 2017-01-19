from Graph import *
from hmm import *

#some code to test it
graph = Graph()
G3 = graph.genEvenGraph(8, 0)
G3 = graph.setSwitches(G3)
#print(G3)
p = 0.05
[A, B] = graph.genSignals(G3, 6, p)
#print(G3)
print(B)
print(A)

hmm = HMM()
C = hmm.genC(G3, B[1,], p)
print(C)

[d, a] = hmm.genD(G3, B[1,], p)

print(a)
print(d)
#print(sum(C[:,0]))
#print(sum(C[:,5]))
#print(sum(C[0,:]))
