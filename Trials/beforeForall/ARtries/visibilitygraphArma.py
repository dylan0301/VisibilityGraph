import numpy as np
import timeseries
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
import arimaEX

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

def plotSubgraphs(G, mainTitle, intervalLen):
    n = len(G)
    H = nx.Graph()
    H.add_nodes_from(G.nodes) 
    for i in range(n-intervalLen+1):
        subG = G.subgraph(range(i, i+intervalLen)).copy()
        for j in range(n):
            if j < i or j >= i + intervalLen:
                subG.add_node(j)
        H.add_edges_from(subG.edges)
    HTitle = mainTitle + ', intervalLen=' + str(intervalLen)
    plotThings(H, HTitle)


def makeFileT(T, titled):
    # customizing runtime configuration stored
    # in matplotlib.rcParams


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(T)
    ax.set(title=titled, xlabel='Time', ylabel='Value')
    
    G = visibilityGraphNX(T)
    plotThings(G, titled)
    # for i in range(3, min(15, len(T))):
    #     plotSubgraphs(G, titled, i)

     
    
     

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



def ARtry5():
    n = 10**4
    for p in range(1,5):
        for trial in range(3):
            np.random.seed(trial)
            T = armaEx.ar(p,n)
            title = 'T'+str(trial)+', n=' + str(n) + ', p=' + str(p)
            makeFileT(T, title)

    # name your Pdf file
    filename = 'ARtry5' + ".pdf"  
    
    # call the function
    save_image(filename) 

def ARIMAtry2(seedd):
    np.random.seed(seedd)
    T = makeAllDistribution.armaaa()
    title = 'arma, n=' + str(500) + ', p=' + str(2) + ', q=' + str(3)
    makeFileT(T, title)
    for d in range(1,6):
        newT = makeAllDistribution.arima(T)
        T = newT[:]
        title = 'arima, n=' + str(500) + ', p=' + str(2) + ', q=' + str(3)+ ', d=' + str(d)
        makeFileT(newT, title)
    # # name your Pdf file
    # filename = 'ARIMAtry2' + ".pdf"  
    
    # # call the function
    # save_image(filename) 

def ARIMAmoregraphs1():
    for i in range(5):
        ARIMAtry2(i)
    
    # name your Pdf file
    filename = 'ARIMAmoregraphs1' + ".pdf"  
    
    # call the function
    save_image(filename) 

if __name__ == "__main__":
    #ARIMAmoregraphs1()
    ARtry5()