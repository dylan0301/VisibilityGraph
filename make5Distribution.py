import numpy as np
import timeseriesEX
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from time import time
import warnings
from hurst import compute_Hc
#https://pypi.org/project/hurst/


def visibilityGraphNX(T):
    n = len(T)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for a in range(n-1):
        slopemax = float('-inf')
        for b in range(a+1, n):
            slopeab = (T[b]-T[a])/(b-a)
            if slopeab > slopemax:
                G.add_edge(a,b)
                slopemax = slopeab
    return G


#https://networkx.org/documentation/stable/auto_examples/drawing/plot_degree.html
def plotDegree(G, title, hurstexp):
    tempdegree = []
    for n, d in G.degree():
        if d > 0:
            tempdegree.append(d)
    degree_sequence = sorted(tempdegree, reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure(title, figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # This was originally to show connected components
    Gcc = G
    # pos = nx.spring_layout(Gcc, seed=10396953)
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20, node_color=range(len(G)))
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4, width= 1)
    ax0.set_title(title)
    ax0.set_axis_off()

    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax0.text(0.05, 0.95, 'HurstExponent = '+ str(hurstexp), transform=ax0.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()

    # plt.show()

def plotPathLength(G, title):
    temppath = []
    p = dict(nx.shortest_path_length(G)) #shortest length
    for s in p.keys():
        for t in p[s].keys():
            temppath.append(p[s][t])
    
    path_sequence = sorted(temppath, reverse=True)
    dmax = max(path_sequence)

    fig = plt.figure(title, figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    # ax0 = fig.add_subplot(axgrid[0:3, :])
    # # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # # This was originally to show connected components
    # Gcc = G
    # # pos = nx.spring_layout(Gcc, seed=10396953)
    # pos = nx.circular_layout(G)
    # nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20, node_color=range(len(G)))
    # nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4, width= 1)
    # ax0.set_title(title)
    # ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(path_sequence, "b-", marker="o")
    ax1.set_title("Shortest Path Length Rank Plot")
    ax1.set_ylabel("Shortest Path Length")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    pathLengths, numOfPaths = np.unique(path_sequence, return_counts=True)
    n = len(path_sequence)
    ratioOfPaths = numOfPaths/n
    fit_y, r2score, mu, std = FitFuncs.gaussFitting(path_sequence, pathLengths, ratioOfPaths)
    # fit_y, r2score = gaussianFit(pathLengths, ratioOfPaths)
    ax2.bar(pathLengths, ratioOfPaths)
    ax2.set_title("Shortest Path Length Histogram")
    ax2.set_xlabel("Shortest Path Length")
    ax2.set_ylabel("Ratio of Nodes")
    ax2.plot(pathLengths, fit_y, '-', label='fit', color='r')
  

    ax0 = fig.add_subplot(axgrid[0:3, :])
    plt.text(0.1, 0.9, "r2score = "+str(r2score))
    plt.text(0.1, 0.6, 'mean = '+str(mu))
    plt.text(0.1, 0.3, 'std = '+str(std))

    fig.tight_layout()

    # plt.show()

def plotDiffDegree(G, H, title):
    tempdegree = []
    Gd = G.degree
    Hd = H.degree
    for n, d in G.degree():
        dd = Gd[n] - Hd[n]
        tempdegree.append(dd)
        
    degree_sequence = sorted(tempdegree, reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure(title, figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    # ax0 = fig.add_subplot(axgrid[0:3, :])
    # # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # # This was originally to show connected components
    # Gcc = G
    # # pos = nx.spring_layout(Gcc, seed=10396953)
    # pos = nx.circular_layout(G)
    # nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20, node_color=range(len(G)))
    # nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4, width= 1)
    # ax0.set_title(title)
    # ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Difference of Degree Rank Plot")
    ax1.set_ylabel("Difference of Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    diffDegrees, numOfNodes = np.unique(degree_sequence, return_counts=True)
    n = len(degree_sequence)
    ratioOfNodes = numOfNodes/n
    fit_y, r2score, mu, std = FitFuncs.gaussFitting(degree_sequence, diffDegrees, ratioOfNodes)
    #fit_y, r2score, parameters = FitFuncs.fitting(FitFuncs.generalGauss, diffDegrees, numOfNodes)
    ax2.bar(diffDegrees, ratioOfNodes)
    ax2.set_title("Difference of Degree Histogram")
    ax2.set_xlabel("Difference of Degree")
    ax2.set_ylabel("ratio of Nodes")
    ax2.plot(diffDegrees, fit_y, '-', label='fit', color='r')

    ax0 = fig.add_subplot(axgrid[0:3, :])
    plt.text(0.1, 0.9, "r2score = "+str(r2score))
    plt.text(0.1, 0.6, 'mean = '+str(mu))
    plt.text(0.1, 0.3, 'std = '+str(std))
    

    fig.tight_layout()

    # plt.show()

class FitFuncs:

    def Gauss(x, A, B):
        y = A*np.exp(-1*B*x**2)
        return y
    
    def generalGauss(x, mean, stddev):
        return 1/(stddev*np.sqrt(2*np.pi)) * np.exp(-1/2*((x - mean) / stddev)**2)

    def fitting(func, xdata, ydata):
        # Recast xdata and ydata into numpy arrays so we can use their handy features
        xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)
        # plt.plot(xdata, ydata, 'o')
        
        # # Define the Gaussian function
        # def Gauss(x, A, B):
        #     y = A*np.exp(-1*B*x**2)
        #     return y
        parameters, covariance = curve_fit(func, xdata, ydata)
        

        fit_A = parameters[0]
        fit_B = parameters[1]
        # perr = np.sqrt(np.diag(covariance))

        fit_y = func(xdata, fit_A, fit_B)

        r2score = r2_score(fit_y, ydata)
        return fit_y, r2score, parameters
        # plt.plot(xdata, ydata, 'o', label='data')
        # plt.plot(xdata, fit_y, '-', label='fit')
        # plt.legend()

    def gaussFitting(data, xdata, ydata):
        # Fit a normal distribution to the data:
        mu, std = norm.fit(data)
        fit_y = norm.pdf(xdata, mu, std)
        r2score = r2_score(fit_y, ydata)
        return fit_y, r2score, mu, std

        
    
        

def make3File(T, titled):
    # customizing runtime configuration stored
    # in matplotlib.rcParams

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(T)
    ax.set(title=titled, xlabel='Time', ylabel='Value')
    
    # compute_Hc returns a tuple of 3 values 
    H_hurst, c_hurst, val_hurst = compute_Hc(T)
    
    G = visibilityGraphNX(T)
    plotDegree(G, titled, H_hurst)
    plotPathLength(G,titled + ', PathLength')

    n = len(T)
    revT = np.zeros(n)-T
    revG = visibilityGraphNX(revT)

    # compute_Hc returns a tuple of 3 values 
    H_hurst, c_hurst, val_hurst = compute_Hc(revT)

    plotDegree(revG, titled+', Reversed', H_hurst)
    plotPathLength(revG, titled+', Reversed, PathLength')

    plotDiffDegree(G, revG, titled+', DiffDegree')
     

def save_image(filename):
    
    # PdfPages is a wrapper around pdf 
    # file so there is no clash and create
    # files with no error.
    p = PdfPages(filename)
    
    # get_fignums Return list of existing 
    # figure numbers
    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]
    
    # iterating over the numbers in list
    for fig in figs: 
        
        # and saving the files
        fig.savefig(p, format='pdf') 
    
    # close the object
    p.close() 

#####################################################################################

def makeAll(n, trials, func, funcname):
    for seedd in range(trials):
        starttime = time()
        print('trial:', seedd, '/// start')
        T = func(n, seedd)
        title = funcname + ', n=' + str(n) + ', seed=' +str(seedd)
        make3File(T, title)
        endtime = time()
        print('trial:', seedd, '/// time:', endtime-starttime)
    
    starttime = time()
    print('makefile start')

    # name your Pdf file
    filename = funcname + ".pdf"  
    
    # call the function
    save_image(filename) 
    
    endtime = time()
    print('makefile time:', endtime-starttime)


if __name__ == "__main__":
    #suppress warnings
    warnings.filterwarnings('ignore')
    
    #makeAll(n=1000, trials=3, func=timeseriesEX.uniformG, funcname='uniformG1')
    makeAll(n=1000, trials=3, func=timeseriesEX.arma(p=3,q=5, p_Param=[-0.25, 0.3, 0.2]).sequence, funcname='arma, p=3, q=5')
    #makeAll(n=100, trials=1, func=timeseriesEX.arma(p=4,q=4).sequence, funcname='arma, p=4, q=4')
