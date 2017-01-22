import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#graph setitng
plt.style.use('ggplot')
font = {'family' : 'meiryo'}
matplotlib.rc('font', **font)
params = {'legend.fontsize': 6}
plt.rcParams.update(params)

class Plot:
    def __init__(self, N, no_sample, observation):
        self.N = N
        self.no_sample = no_sample
        self.observation = observation

    def barGraph(self, p_vec):
        #plot probability (bar)
        data = pd.Series(p_vec)
        data.plot(kind="bar", color="k", alpha=0.7)
        plt.savefig('./plots/bar/nodes_{}-observation_{}-no_sample_{}.png'.format(self.N, self.observation, self.no_sample))

    def lineGraphObservation(self, p_matrix):
        df = pd.DataFrame(np.transpose(p_matrix))
        df.plot(grid=True, linewidth = 0.4)
        plt.savefig('./plots/observation/nodes_{}-observation_{}-no_sample_{}.png'.format(self.N, self.observation, self.no_sample))

    def lineGraphSample(self, p_matrix):
        y = np.log(p_matrix)
        df = pd.DataFrame(p_matrix[0:10, :])
        df.plot(grid=True, linewidth = 0.4, logy=True)
        plt.savefig('./plots/samples/nodes_{}-observation_{}-no_sample_{}.png'.format(self.N, self.observation, self.no_sample))

    def scatterObservation(self, p_matrix):
        plt.clf()
        for i in range(1, self.observation):
            plt.scatter(x=[i]*self.N*3, y=np.linspace(1, self.N*3, self.N*3), s=p_matrix[:, i-1]*250, color="blue")
        plt.savefig('./plots/scatter/nodes_{}-observation_{}-no_sample_{}.png'.format(self.N, self.observation, self.no_sample))
