import numpy as np
import timeseries
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
import arimaEx
from collections import defaultdict

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
def plotThings(G, title):
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

def plotDiff(G, H, title):
    # tempdict = defaultdict(int)
    # for n, d in G.degree():
    #     tempdict[n] += d
    # for n, d in H.degree():
    #     tempdict[n] -= d
    # tempdegree = []
    # for n, d in tempdict.items():
    #     tempdegree.append(d)
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


def plotShortestDist(G, title):
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

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(path_sequence, "b-", marker="o")
    ax1.set_title("Path Length Rank Plot")
    ax1.set_ylabel("Path Length")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(path_sequence, return_counts=True))
    ax2.set_title("Path Length histogram")
    ax2.set_xlabel("Path Length")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()

    # plt.show()

def makeFileT(T, titled):
    # customizing runtime configuration stored
    # in matplotlib.rcParams


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(T)
    ax.set(title=titled, xlabel='Time', ylabel='Value')
    
    G = visibilityGraphNX(T)
    plotThings(G, titled)

    plotShortestDist(G,titled + ' path')
    
    
    # for i in range(3, min(15, len(T))):
    #     plotSubgraphs(G, titled, i)

def makeDiffFileT(T, titled):
    # customizing runtime configuration stored
    # in matplotlib.rcParams


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(T)
    ax.set(title=titled, xlabel='Time', ylabel='Value')
    
    G = visibilityGraphNX(T)
    n = len(T)
    revT = np.zeros(n)-T
    H = visibilityGraphNX(revT)
    plotDiff(G,H, titled)
     

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



def reversedUniformG3():
    n = 10000
    trials = 3
    for seedd in range(trials):
        T = timeseries.uniformG(n, seedd)
        title = 'uniform, n=' + str(n) + ', seed=' +str(seedd)
        makeFileT(T, title)
        revT = np.ones(n)-T
        title = 'uniform, n=' + str(n) + ', seed=' +str(seedd) + ', reversed'
        makeFileT(revT, title)

    # name your Pdf file
    filename = 'reversedUniformG3' + ".pdf"  
    
    # call the function
    save_image(filename) 

def reversedArmaaG2():
    n = 10000
    T = makeAllDistribution.armaaa()
    title = 'armaaa, n=' + str(n)
    makeFileT(T, title)
    revT = np.zeros(n)-T
    title = 'armaaa, n=' + str(n) + ', reversed'
    makeFileT(revT, title)

    # name your Pdf file
    filename = 'reversedArmaaaG2' + ".pdf"  
    
    # call the function
    save_image(filename) 

def reversedArmaaP1G2():
    n = 1000
    trials = 3
    for seedd in range(trials):
        T = makeAllDistribution.armaaaP1(n, seedd)
        title = 'armaaa, n=' + str(n) + ', p=1'
        makeFileT(T, title)
        revT = np.zeros(n)-T
        title = 'armaaa, n=' + str(n) + ', p=1' + ', reversed'
        makeFileT(revT, title)


    # name your Pdf file
    filename = 'reversedArmaaaP1G2' + ".pdf"  
    
    # call the function
    save_image(filename) 

def diffUniformG1():
    n = 10000
    trials = 3
    for seedd in range(trials):
        T = timeseries.uniformG(n, seedd)
        title = 'diffUniform, n=' + str(n) + ', seed=' +str(seedd)
        makeDiffFileT(T, title)
        

    # name your Pdf file
    filename = 'diffUniformG1' + ".pdf"  
    
    # call the function
    save_image(filename) 

def diffArmaaG1():
    n = 10000
    trials = 3
    for seedd in range(trials):
        T = makeAllDistribution.armaaaP1(n, seedd)
        title = 'diffArmaaP1, n=' + str(n) + ', seed=' +str(seedd)
        makeDiffFileT(T, title)
        

    # name your Pdf file
    filename = 'diffArmaaP1G1' + ".pdf"  
    
    # call the function
    save_image(filename) 

def randomWG1():
    n = 1000
    trials = 3
    for seedd in range(trials):
        T = timeseries.uniformG(n, seedd)
        title = 'randomwalk, n=' + str(n) + ', seed=' +str(seedd)
        makeFileT(T, title)
        revT = np.ones(n)-T
        title = 'randomwalk, n=' + str(n) + ', seed=' +str(seedd) + ', reversed'
        makeFileT(revT, title)

    # name your Pdf file
    filename = 'randomwalkG1' + ".pdf"  
    
    # call the function
    save_image(filename) 



def makeAll(n, trials, func):
    for seedd in range(trials):
        T = timeseries.uniformG(n, seedd)
        title = 'randomwalk, n=' + str(n) + ', seed=' +str(seedd)
        makeFileT(T, title)
        revT = np.ones(n)-T
        title = 'randomwalk, n=' + str(n) + ', seed=' +str(seedd) + ', reversed'
        makeFileT(revT, title)

    # name your Pdf file
    filename = 'randomwalkG1' + ".pdf"  
    
    # call the function
    save_image(filename) 


if __name__ == "__main__":
    randomWG1()