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
    def __init__(self, N, no_sample, observation, p):
        self.N = N
        self.no_sample = no_sample
        self.observation = observation
        self.p = p

    def barGraph(self, p_vec):
        #plot probability (bar)
        data = pd.Series(p_vec)
        data.plot(kind="bar", color="k", alpha=0.7)
        plt.savefig('./plots/bar/nodes_{}-observation_{}-no_sample_{}_p={}.png'.format(self.N, self.observation, self.no_sample, self.p))

    def lineGraphObservation(self, p_matrix):
        df = pd.DataFrame(np.transpose(p_matrix))
        df.plot(grid=True, linewidth = 0.4)
        plt.savefig('./plots/observation/nodes_{}-observation_{}-no_sample_{}_p={}.png'.format(self.N, self.observation, self.no_sample, self.p))

    def lineGraphSample(self, p_matrix):
        y = np.log(p_matrix)
        df = pd.DataFrame(y)
        df.plot(grid=True, linewidth = 0.4, legend=False)
        plt.savefig('./plots/samples/nodes_{}-observation_{}-no_sample_{}_p={}.png'.format(self.N, self.observation, self.no_sample, self.p))

    def scatterObservation(self, p_matrix, true_states):
        plt.clf()
        for i in range(0, self.observation-1):
            plt.scatter(x=[i]*self.N*3, y=np.linspace(0, self.N*3-1, self.N*3), s=p_matrix[:, i]*250, color="blue")
            plt.scatter(x=i, y=true_states[i], s=200, facecolor="none", edgecolors="r")
        plt.savefig('./plots/scatter/nodes_{}-observation_{}-no_sample_{}_p={}.png'.format(self.N, self.observation, self.no_sample, self.p))
