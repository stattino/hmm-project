from Graph import * 
#some code to test it
graph = Graph()
G3 = graph.genEvenGraph(12, 0)
G3 = graph.setSwitches(G3)
#print(G3)
[A, B] = graph.genSignals(G3)
print(B)
