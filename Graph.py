import numpy as np

class Graph:
    def genEvenGraph(self, n, shuffle=0):  # (shuffle not yet completed.. leads to self loops)
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
        if (np.fmod(n, 2) != 0):
            n += 1
        if (n < 5):
            n = 6
        g = np.zeros((n, n))
        for i in range(0, n):
            u = np.random.choice([1, 2, 3], (1, 3), False)
            if (i % 2 == 0):
                g[i, np.fmod(i + 1, n)] = u[0, 0]
                g[i, np.fmod(i + 2, n)] = u[0, 1]
                g[i, np.fmod(i - 2, n)] = u[0, 2]
            else:
                g[i, np.fmod(i + 2, n)] = u[0, 0]
                g[i, np.fmod(i - 2, n)] = u[0, 1]
                g[i, np.fmod(i - 1, n)] = u[0, 2]

        if (shuffle == 1):
            np.random.shuffle(g)
        return g

    def setSwitches(self, g):
         # Set switches for vertices at random. (L=2 and R=3)
         # These are the true switches.
        N = g.shape[0]
        sigmas = np.random.randint(2, 4, size=N)
        return sigmas

    def genSignals(self, g, sigmas, T=100, p=0.05):
        """Generates N observations from a train traversing the graph g.
            Rules?
        Input:
          g - graph structure
          sigmas - switch
          T - number of observations desired
          p - noise level in observation
        Returns:
          true_path -   true trajectory made by the train.
                        1st row: index of vertices
                        2nd row: arrival labels
                        3rd row: departure labels
          observed - trajectory observed with noise from switches
                        1st row: arrival labels
                        2nd row: departure labels
      """
        N = g.shape[0]
        x_0 = np.random.randint(N)
        true_path = np.zeros((3, T))
        observed = np.zeros((2, T))

        true_path[0, 0] = x_0  # vertex
        true_path[1, 0] = 0  # not observed
        true_path[2, 0] = np.random.randint(1, 3)

        # true_path[1, 0] = np.random.choice([1, true_path[2,0]], 1, False, [1/3, 2/3])

        observed[0, 0] = 0 # By definition, doesnt observe this.
        if true_path[2, 0] == 1: # noise only if L/R
            observed[1, 0] = 1
        else:
            observed[1, 0] = np.random.choice([sigmas[x_0], 2 if sigmas[x_0] == 3 else 3], 1, False, [1 - p, p])

        for i in range(1, T):
            x_prev = true_path[0, i - 1]
            e_prev = true_path[2, i - 1]

            temp_g = np.copy(g[x_prev,])
            temp_g[x_prev] = 0
            vertex_idx = np.nonzero(temp_g == e_prev)[0][0]  # index of next vertex
            arr_label = g[vertex_idx, x_prev]  # arrival label of edge
            sw = sigmas[vertex_idx]

            true_path[0, i ] = vertex_idx
            true_path[1, i ] = arr_label

            observed[0, i] = arr_label
            if arr_label == 1:
                true_path[2, i] = sw
                observed[1, i] = np.random.choice([sw, 2 if sw == 3 else 3], 1, False, [1 - p, p])
            else:
                true_path[2, i] = 1
                observed[1, i] = 1

        return true_path, observed
