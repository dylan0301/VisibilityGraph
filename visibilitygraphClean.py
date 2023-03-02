from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import timeseries
import matplotlib.pyplot as plt

#slope n^2
def visibilityGraphVer2(T):
    n = len(T)
    graph = defaultdict(list)
    for a in range(n-1):
        slopemax = float('-inf')
        for b in range(a+1, n):
            slopeab = (T[b]-T[a])/(b-a)
            if slopeab > slopemax:
                graph[a].append(b)
                graph[b].append(a)
                slopemax = slopeab
    return graph

def probCoef(graph):
    n = len(graph)
    frequency = defaultdict(int)
    for i in range(n):
        for j in graph[i]:
            if j > i:
                frequency[j-i] += 1
    divide = frequency[1]
    X = []
    y = []
    for i in range(len(frequency)):
        frequency[i] /= divide
        logfreq = np.log(frequency[i])
        if logfreq != float('-inf'):
            X.append([i])
            y.append(logfreq)
    X = np.array(X)
    y = np.array(y)
    reg = LinearRegression().fit(X, y)
    print('reg score:',reg.score(X,y))
    return reg.coef_[0]




def forecastVer3(T, graph, coef):
    space = 0.001
    n = len(T)
    probLimit = 10**(-4)
    newconnected = []
    probnow = 1
    dist1 = 0
    while probnow > probLimit:
        ran = random.random()    
        if ran <= probnow:
            newconnected.append(1)
        else:
            newconnected.append(0)
        dist1 += 1
        probnow = np.exp(coef*dist1)

    def newnodeVisibles2(T, newnode):
        visiblelist = []
        slopeSteep = float('inf')
        for b in range(n-1, -1, -1):
            slopeab = (newnode-T[b])/(n-b)
            if slopeab < slopeSteep:
                visiblelist.append(1)
                slopeSteep = slopeab
            else:
                visiblelist.append(0)
        return visiblelist
    
    globalmin = min(T)
    globalmax = max(T)
    candidates = np.arange(globalmin - 10*space, globalmax + 11*space, space)
    maxscore = 0
    maxscoreNewnode = 0
    maxscoreVisibles = []
    for newnode in candidates:
        scorenow = 0
        visibles = newnodeVisibles2(T, newnode)
        for i in range(len(newconnected)):
            if newconnected[i] == visibles[i]:
                scorenow += 1
        if scorenow > maxscore:
            maxscore = scorenow
            maxscoreNewnode = newnode
            maxscoreVisibles = visibles
    for i in maxscoreVisibles:
        graph[i].append(n)
        graph[n].append(i)
    newT = np.append(T, maxscoreNewnode)
    return newT, graph

def totalforecastVer2(T, graph = None):
    if graph == None:
        graph = visibilityGraphVer2(T)
    coef = probCoef(graph)
    newT, graph = forecastVer3(T, graph, coef)
    return newT, graph


if __name__ == "__main__":
    W = timeseries.wiener()

    for i in range(100):
        W, graph = totalforecastVer2(W)
    plt.plot(W)