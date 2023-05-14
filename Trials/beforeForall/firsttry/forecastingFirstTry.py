#https://github.com/MeenaMe0/time-series-and-graph-theory
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import timeseries
import matplotlib.pyplot as plt

#naive n^3
def visibilityGraphVer1(T):
    n = len(T)
    graph = defaultdict(list)
    for i in range(n):
        for j in range(i+1, n):
            visible = True
            for k in range(i+1, j):
                if T[k] > (T[j]-T[i])/(j-i)*(k-i)+T[i]:
                    visible = False
            if visible:
                graph[i].append(j)
                graph[j].append(i)
    return graph

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


#ln y = ax로 그냥 e^ax로만 해보자. a를 return 함
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

def forecastVer1(T, graph, coef):
    n = len(T)
    probLimit = 10**(-3.5)
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
    upperbound = float('inf')
    lowerbound = float('-inf')
    #only visible to self
    for i in range(len(newconnected)):
        node = n-1-i
        if newconnected[i] == 0: #currently not visible from node
            newupperbound = float('inf')
            for j in graph[node]:
                if j > node:
                    newupperboundTemp = (T[j]-T[node])/(j-node)*(n-node)+T[node]
                    newupperbound = min(newupperbound, newupperboundTemp)
            if newupperbound >= lowerbound:
                upperbound = min(upperbound, newupperbound)
        else:
            newlowerbound = float('-inf')
            for j in graph[node]:
                if j > node:
                    newlowerboundTemp = (T[j]-T[node])/(j-node)*(n-node)+T[node]
                    newlowerbound = max(newlowerbound, newlowerboundTemp)
            if newlowerbound <= upperbound:
                lowerbound = max(lowerbound, newlowerbound)
    return (upperbound + lowerbound)/2

#failed
def forecastVer2(T, graph, coef):
    space = 0.02
    n = len(T)
    def newnodeVisibles(T, newnode):
        visibles = []
        slopeSteep = float('inf')
        for b in range(n-1, -1, -1):
            slopeab = (newnode-T[b])/(n-b)
            if slopeab < slopeSteep:
                visibles.append(b)
                slopeSteep = slopeab
        return visibles
    
    globalmin = min(T)
    globalmax = max(T)
    candidates = np.arange(globalmin - 10*space, globalmax + 11*space, space)
    maxscore = 0
    maxscoreNewnode = 0
    maxscoreVisibles = []
    scoreboard = [np.exp(coef*i) for i in range(n)]
    for newnode in candidates:
        scorenow = 0
        visibles = newnodeVisibles(T, newnode)
        for i in visibles:
            scorenow += scoreboard[n-i]
        if scorenow > maxscore:
            maxscore = scorenow
            maxscoreNewnode = newnode
            maxscoreVisibles = visibles
    for i in maxscoreVisibles:
        graph[i].append(n)
        graph[n].append(i)
    newT = np.append(T, maxscoreNewnode)
    return newT, graph

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





def totalforecastVer1(T):
    graph = visibilityGraphVer2(T)
    coef = probCoef(graph)
    nextnode = forecastVer1(T, graph, coef)
    newT = np.append(T, nextnode)
    print(nextnode)
    return newT

def totalforecastVer2(T, graph = None):
    if graph == None:
        graph = visibilityGraphVer2(T)
    coef = probCoef(graph)
    newT, graph = forecastVer3(T, graph, coef)
    return newT, graph





W = timeseries.wiener()

for i in range(100):
    W, graph = totalforecastVer2(W)
plt.plot(W)
plt.show()


