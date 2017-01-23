import numpy as np

class Graph:
    def genEvenGraph(self, n, shuffle=0):  # (shuffle not yet completed, it leads to self loops)
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
            shuffle - if 1, randomly shuffles graph. (doesnt work)
        """
        if (np.fmod(n, 2) != 0):
            raise ValueError('WrongInput')
        if (n < 5):
            raise ValueError('SmallInput')

        g = np.zeros((n, n))
         # Heuristic, creates two circles of points, one outside and one inside.
         # Each point is connected to its two neighbours on the same circle, and
         # to the adjacent point in the inner circle.
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
         # Set switches for vertices at random, L=2 and R=3.
        N = g.shape[0]
        sigmas = np.random.randint(2, 4, size=N)
        return sigmas

    def genSignals(self, g, sigmas, T=100):
        """Generates N observations from a train traversing the graph g.
        Input:
          g - graph structure
          sigmas - switch
          T - number of observations desired
        Returns:
          true_path -   true trajectory made by the train.
                        1st row: index of vertices
                        2nd row: arrival labels
                        3rd row: departure labels
      """
        N = g.shape[0]
        x_0 = np.random.randint(N)
        true_path = np.zeros((3, T))

        true_path[0, 0] = x_0
        true_path[1, 0] = 0  # not defined
        true_path[2, 0] = np.random.randint(1, 3)
        # true_path[1, 0] = np.random.choice([1, true_path[2,0]], 1, False, [1/3, 2/3])

         # Deterministically traverse the graph to create the true path,  following the
         # current switches, and adds noise to the observations.
        for i in range(1, T):
            x_prev = true_path[0, i - 1]
            e_prev = true_path[2, i - 1]

            vertex_idx = np.nonzero(g[x_prev,] == e_prev)[0][0] # index of next vertex
            arr_label = g[vertex_idx, x_prev]               # arrival label of edge
            sw = sigmas[vertex_idx]

            true_path[0, i ] = vertex_idx
            true_path[1, i ] = arr_label
            if arr_label == 1:
                true_path[2, i] = sw
            else:
                true_path[2, i] = 1

        return true_path

    def truePathToStates(self, true_path):
        T = true_path.shape[1]
        state_path = np.zeros(T)
        for i in range(0, T):
            label = true_path[2, i] - 1
            state_path[i] = 3*true_path[0, i] + label
        return state_path

    def genAltObservations(self, true_path, p):
        T = true_path.shape[1]
        observed = np.zeros(T)
        for i in range(0, T):
            truth = true_path[2, i]
            # truth is in {1, 2, 3}
            observed[i] = np.random.choice([truth, (truth+1)%3 +1, (truth+2)%3 +1], 1, False, [1 - p, p/2, p/2])
        return observed

