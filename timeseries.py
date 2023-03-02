import numpy as np
import matplotlib.pyplot as plt

def wiener():
    np.random.seed(0)
    
    # Set the number of time steps and the time step size
    num_steps = 500
    dt = 0.01
    sigma = np.sqrt(dt)

    # Initialize the Wiener process
    # Make every element in list = 0
    x = np.zeros(num_steps)

    # Generate the Wiener process by summing up the increments
    for i in range(1, num_steps):
        # value = previous value + random from normal distribution 
        # normal : mean = 0 , sd = sigma
        x[i] = x[i-1] + np.random.normal(0, sigma)

    # Plot the Wiener process
    plt.plot(x)
    plt.show()
    return x

def sde():
    # Set the number of time steps ,the noise and parameters
    num_steps = 500 
    dt = 0.01
    speed = 10
    sigma = 3
    dwiener = np.random.normal(0, np.sqrt(dt),num_steps)    # difference of weiner process is normal distribution

    # Generate the example of SDE and set the initial value
    sde_ex = []
    sde_ex.append(0)

    for i in range (1,num_steps) :
        # Add the random value to list of random variable : sde_ex
        sde_ex.append(sde_ex[i-1] - speed*(sde_ex[i-1])*dt + sigma*(dwiener[i-1]))

    # Plot the SDE model
    plt.plot(sde_ex)
    plt.show()

def randomwalk():
    # Set the number of time steps and the time step size
    num_steps = 500 

    # Set the probability : random value from a prob list 
    prob = [-1,1,1,1,1] # p = 0.8
    #prob = [-1,1] # p= 0.5 

    # Generate the Random walk by random for num_steps times
    random_w = [0]
    for i in range (1,num_steps) :
        # Add the random value to list of random variable : random_w
        random_w.append(random_w[i-1] + np.random.choice(prob))

    # Plot the Random walk
    plt.plot(random_w)

    # Draw the mean or expected value line : 𝑎+𝑡(2𝑝−1)
    startline = random_w[0] # a
    mean = 2*0.8 -1 # 2p -1
    plt.plot((0,num_steps),(startline,startline + num_steps*(mean)) ,color = 'grey')

    plt.show()

def binary():
    # Set the number of time steps and the time step size
    num_steps = 500 

    # Set the probability : random value from a prob list 
    prob = [-1,1,1,1,1] # p = 0.8
    #prob = [-1,1] # p= 0.5 

    # Generate the Binary process by random for num_steps times
    binary_p = []
    for i in range (0,num_steps) :
        # Add the random value to list of random variable : binary_p 
        binary_p.append(np.random.choice(prob))

    # Plot the Binary process
    plt.plot(binary_p)

    # Draw the mean or expected value line : 2*p = 1
    mean = 2*0.8 -1
    plt.plot((0,num_steps),(mean,mean) ,color = 'grey')

    # Set the graph frame range -2 to 2
    plt.ylim([-2,2])
    plt.show()