# import networkx as nx
# import matplotlib.pyplot as plt

# G = nx.complete_graph(10)
# # G.subgraph: Node [0, 1, 2]가 존재하는 subgraph를 리턴 
# subG = nx.subgraph(G, nbunch = [2,5,7]).copy()
# # G.edge_subgraph: Edge [(0, 1), (1, 2)]가 존재하는 subgraph를 리턴. 
# EdgeSubG = G.edge_subgraph([(0, 1), (1, 2)])

# print(G.edges(nbunch = [2,5,7]))

# for i in range(10):
#     if i != 2 and i!= 5 and i!=7:
#         subG.add_node(i)
# pos = nx.circular_layout(subG)

# nx.draw(subG, pos)
# plt.show()








# import numpy as np

# b = (np.array([2, 3, 4, 5, 7]), np.array([0, 1, 2, 3, 2, 4, 2, 0, 1]))

# a=np.array([[2,3,4],
#             [5,4,7],
#            [4,2,3]])

# unique_array=np.unique(a,return_counts=True)

# print(unique_array)




# from __future__ import print_function
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# xdata = [ -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# ydata = [1.2, 4.2, 6.7, 8.3, 10.6, 11.7, 13.5, 14.5, 15.7, 16.1, 16.6, 16.0, 15.4, 14.4, 14.2, 12.7, 10.3, 8.6, 6.1, 3.9, 2.1]

# # Recast xdata and ydata into numpy arrays so we can use their handy features
# xdata = np.asarray(xdata)
# ydata = np.asarray(ydata)
# plt.plot(xdata, ydata, 'o')

# # Define the Gaussian function
# def Gauss(x, A, B):
# 	y = A*np.exp(-1*B*x**2)
# 	return y
# parameters, covariance = curve_fit(Gauss, xdata, ydata)

# fit_A = parameters[0]
# fit_B = parameters[1]


# print(covariance)
# perr = np.sqrt(np.diag(covariance))
# print(perr)
# fit_y = Gauss(xdata, fit_A, fit_B)
# plt.plot(xdata, ydata, 'o', label='data')
# plt.plot(xdata, fit_y, '-', label='fit')
# plt.legend()
# plt.show()





# import numpy as np
# from scipy.stats import norm
# import matplotlib.pyplot as plt


# # Generate some data for this demonstration.
# data = norm.rvs(10.0, 2.5, size=500)
# print(data)
# # Fit a normal distribution to the data:
# mu, std = norm.fit(data)

# # Plot the histogram.
# plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

# # Plot the PDF.
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)

# plt.show()




import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

data = np.random.normal(loc=5.0, scale=2.0, size=1000)
mean,std=norm.fit(data)
print(data)
print(mean,std)

plt.hist(data, bins=30, density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
y = norm.pdf(x, mean, std)
plt.plot(x, y)
plt.show()