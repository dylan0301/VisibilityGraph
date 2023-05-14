import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_process.ArmaProcess.html
def ar(n, seedd):
    np.random.seed(seedd)
    p = 5
    randarr = np.random.normal(0, 1, size=p)
    arparams = randarr / np.linalg.norm(randarr)
    ar = np.r_[1, -arparams] # add zero-lag and negate
    ma = 1 # add zero-lag
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    sample = arma_process.generate_sample(n)
    return sample

def arima(before):
    # Generate model that the difference is arma model
    arima_model= [0]
    num_steps = len(before)


    for i in range (1,num_steps) :
        arima_model.append(arima_model[i-1] + before[i])
    return arima_model


def armaaa(num_steps, seedd):
    np.random.seed(seedd)
    # Set the number of time steps and the white noise
    # num_steps = 10000 
    white_noise = np.random.randn(num_steps+2)      #normal random numsteps+2 value

    # Generate the arma model at thie T ,by adding value from time T-1, T-2 and white_noise from time T ,T-1 , T-2
    arma_model = [0,0]
    for i in range (2,num_steps) :
        # Add the random value to list of random variable : arma_model
        # -0.5 * X_t-2 + 0.75 * X_t-1 + W_t + W_t-1 + W_t-2
        arma_model.append(-0.5*arma_model[i-2]+0.75*arma_model[i-1]+white_noise[i]+white_noise[i-1]+white_noise[i-2])
    return arma_model

def armaaaP1(num_steps, seedd):
    np.random.seed(seedd)
    # Set the number of time steps and the white noise
    white_noise = np.random.randn(num_steps+1)      #normal random numsteps+2 value

    # Generate the arma model at thie T ,by adding value from time T-1, T-2 and white_noise from time T ,T-1 , T-2
    arma_model = [0]
    for i in range (1,num_steps) :
        # Add the random value to list of random variable : arma_model
        # -0.5 * X_t-2 + 0.75 * X_t-1 + W_t + W_t-1 + W_t-2
        arma_model.append(0.75*arma_model[i-1]+white_noise[i])
    return arma_model