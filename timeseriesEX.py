import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class arma:
    def __init__(self, p = 1, q = 0, p_Param = [], q_Param = []):
        self.p = p
        self.q = q

        # np.random.seed(0)
        if not p_Param:
            p_Param = np.random.rand(p)
        if not q_Param:
            q_Param = np.random.rand(q)
        self.p_Param = np.asarray(p_Param)
        self.q_Param = np.asarray(q_Param)


    def sequence(self, n, seedd):
        p = self.p
        q = self.q

        np.random.seed(seedd)
        # Set the number of time steps and the white noise
        white_noise = np.random.randn(n+self.q+1) 

        seq = np.zeros(self.p)
        q_idx = q
        p_idx = p
        revP_Param = np.flip(self.p_Param)
        revQ_Param = np.flip(self.q_Param)
        while len(seq) < n:
            val = white_noise[q_idx]
            val += np.inner(seq[p_idx-p : p_idx], revP_Param)
            val += np.inner(white_noise[q_idx-q : q_idx], revQ_Param)
            seq = np.append(seq,val)
            q_idx += 1
            p_idx += 1
        
        return seq





def wiener(num_steps, seedd):
    np.random.seed(seedd)
    
    # Set the number of time steps and the time step size
    # num_steps = 5
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
    # plt.plot(x)
    # plt.show()
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

def randomwalk(num_steps, seedd):
    np.random.seed(seedd)

    # Set the number of time steps and the time step size
    # num_steps = 500 

    # Set the probability : random value from a prob list 
    prob = [-1,1,1,1,1] # p = 0.8
    #prob = [-1,1] # p= 0.5 

    # Generate the Random walk by random for num_steps times
    random_w = [0]
    for i in range (1,num_steps) :
        # Add the random value to list of random variable : random_w
        random_w.append(random_w[i-1] + np.random.choice(prob))

    return random_w
    # # Plot the Random walk
    # plt.plot(random_w)

    # # Draw the mean or expected value line : 𝑎+𝑡(2𝑝−1)
    # startline = random_w[0] # a
    # mean = 2*0.8 -1 # 2p -1
    # plt.plot((0,num_steps),(startline,startline + num_steps*(mean)) ,color = 'grey')

    # plt.show()

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

def uniformG(n, seedd):
    np.random.seed(seedd)
    return np.random.rand(n)