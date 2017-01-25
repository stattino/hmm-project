import pandas as pd
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
#graph setitng
plt.style.use('ggplot')
font = {'family' : 'meiryo'}
matplotlib.rc('font', **font)
params = {'legend.fontsize': 6}
plt.rcParams.update(params)

class Plot:
    def __init__(self, N, no_sample, observation, p):
        self.N = N
        self.no_sample = no_sample
        self.observation = observation
        self.p = p

    def barGraph(self, p_vec):
        #plot probability (bar)
        data = pd.Series(p_vec)
        data.plot(kind="bar", color="k", alpha=0.7)
        plt.savefig('../plots/bar/nodes_{}-observation_{}-no_sample_{}_p={}.png'.format(self.N, self.observation, self.no_sample, self.p))

    def lineGraphObservation(self, p_matrix):
        df = pd.DataFrame(np.transpose(p_matrix))
        df.plot(grid=True, linewidth = 0.4)
        plt.savefig('../plots/observation/nodes_{}-observation_{}-no_sample_{}_p={}.png'.format(self.N, self.observation, self.no_sample, self.p))

    def lineGraphSample(self, p_matrix, time='unknown'):
        y = p_matrix
        df = pd.DataFrame(y)
        cmap = matplotlib.cm.get_cmap('brg')
        df.plot(grid=True, linewidth=0.3, legend=False, figsize= (8,4), fontsize=18, colormap=cmap, ylim= [0.00001, 1], yticks=[1, 0.1, 0.01, 0.001, 0.0001], xticks=[0, 100, 200, 300], logy=True)
        plt.xlabel('Iterations [100x]', fontsize=18)
        plt.ylabel('log(p(s_i|O))', fontsize=18)
        plt.savefig('../plots/samples/nodes_{}-observation_{}-no_sample_{}_p={}_time={}.png'.format(self.N, self.observation, self.no_sample, self.p, time))

    def scatterObservation(self, p_matrix, true_states):
        plt.clf()
        for i in range(0, self.observation-1):
            plt.scatter(x=[i]*self.N*3, y=np.linspace(0, self.N*3-1, self.N*3), s=p_matrix[:, i]*250, color="blue")
            plt.scatter(x=i, y=true_states[i], s=200, facecolor="none", edgecolors="r")
        plt.xlabel("Observation", fontsize=15)
        plt.ylabel("States", fontsize=15)
        plt.savefig('../plots/scatter/nodes_{}-observation_{}-no_sample_{}_p={}.png'.format(self.N, self.observation, self.no_sample, self.p))

    def absError(self, p_matrix, p_true, proposal):
        N, D = p_matrix.shape
        for i in range(0, N):
            p_matrix[i, :] = p_matrix[i, :] - p_true
        #p_matrix = np.add((p_matrix), np.ones_like(p_matrix))
        y = np.abs(p_matrix)
        #max_val = np.max(np.sum(y, 1))
        #col_ind = np.nonzero(np.sum(y, 1) == max_val)[0][0]

        #for i in range(0, D):
        #   y[:, i] = np.add(np.copy(y[:, i]), (i+1)*np.ones_like(y[:, i]))
        df = pd.DataFrame(y)
        cmap = matplotlib.cm.get_cmap('brg')
        df.plot(grid=True, linewidth=0.4, legend=False, fontsize= 18, xticks= [0, 100, 200, 300], colormap= cmap)
        plt.xlabel('Iterations [100x]', fontsize=18)
        plt.ylabel('abs. error', fontsize=18)
        plt.savefig('../plots/error/nodes_{}-observation_{}-no_sample_{}_p={}_proposal_{}.png'.format(self.N, self.observation,
                                                                                            self.no_sample, self.p, proposal))
