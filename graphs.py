import numpy as np

def genEvenGraph(n, shuffle=0): #        (shuffle not yet completed.. leads to self loops)
    """
    Generates an even graph with degree 3 for every vertex.
    Maximum of one edge between 2 vertices. Returns a matrix.
    Labeling:
        0 - no edge
        1 - edge "0"
        2 - edge "L"
        3 - edge "R"
    Parameters:
        n - number of vertices
        shuffle - if 1, randomly shuffles graph.
  """
    if (np.fmod(n,2)!=0):
        n += 1
    if (n < 5):
        n = 6
    g = np.zeros((n,n))
    for i in range(0,n):
        u = np.random.choice([1, 2, 3], (1, 3), False)
        if (i%2 == 0):
            g[i, np.fmod(i+1, n)] = u[0,0]
            g[i, np.fmod(i+2, n)] = u[0,1]
            g[i, np.fmod(i-2, n)] = u[0,2]
        else:
            g[i, np.fmod(i+2, n)] = u[0,0]
            g[i, np.fmod(i-2, n)] = u[0,1]
            g[i, np.fmod(i-1, n)] = u[0,2]

    if (shuffle==1):
        np.random.shuffle(g)
    return g

def setSwitches(g):
    # Set switches for vertices at random. (L=2 and R=3)
    for i in range(0,g.shape[0]):
        g[i,i] = np.random.randint(2,4)
    return g

def genSignals(g, N = 100, p=0.05):
    """Generates N observations from a train traversing the graph g.
        Rules?
    Input:
      g - graph
      N - number of observations
      p - noise in observation
    Returns:
      true - trajectory made by the train
      observed - trajectory observed (with noise)
  """
    n = g.shape[0]
    x_0 = np.random.randint(n)
    true_obs = np.zeros((3, N))
    observed = np.zeros((2, N))

    true_obs[0,0] = x_0
    true_obs[1,0] = np.random.randint(1, 4)
    true_obs[2,0] = g[x_0, x_0]

    for i in range(0,N-1):
        origin = true_obs[0, i]
        edge = true_obs[1, i]

        temp_g = np.copy(g[origin,])
        temp_g[origin] = 0
        dep_indx = np.nonzero((temp_g)==edge)[0][0]
        arr_indx = g[dep_indx, origin]

        true_obs[0, i+1] = dep_indx
        true_obs[2, i+1] = g[dep_indx, dep_indx]
        if arr_indx==1:
            true_obs[1, i+1] = g[dep_indx, dep_indx]
        else:
            true_obs[1, i+1] = 1
    return true_obs

#some code to test it
G3 = genEvenGraph(8,0)
G3 = setSwitches(G3)
#print(G3)
A = genSignals(G3)
print(A)