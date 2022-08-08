# this is a short version of gillespie_ode_likelihood_220703 
# the functions are modified and they take shorter time to execute, I dont' run the simultion for the whole tend time
# only first and last time points are calcualted in ODE simulation
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import quad
import matplotlib.pyplot as plt


def ode_sim(variables,newt,params):
    # lambda = beta*S
    # mu = gamma
    S = variables[0]
    I = variables[1]
    R = variables[2]
    E = variables[3]

    beta = params[0]
    gamma = params[1]
    psi = params[2]

    dSdt = -beta * S * I

    dIdt = beta * S * I -  (gamma + psi)* I

    dRdt = (gamma + psi) * I 
    
    dEdt = -(beta * S + gamma + psi)*E + (beta * S)*E**2 + gamma  

    return [dSdt, dIdt, dRdt, dEdt]


def ode_calc(ti,betap,gammap,psip):
    global S0, I0, R0, E0
    y0 = [S0,I0,R0,E0]
    params = [betap,gammap,psip] 
    time_of_simulation = np.linspace(0,ti,num=2)
    y = odeint(ode_sim,y0, time_of_simulation, args=(params,))
    return y
    
def calc_lamda(ti,betap,gammap,psip):
    y = ode_calc(ti,betap, gammap, psip)
    # lamda is S(ti) which is last time point of column 0 (S)
    lamda = betap*y[-1,0]
    return lamda

# this function takes ti: time from matrix
def phi_integrand(ti,betap,gammap,psip):
    global S0,E0
    mu = gammap
    if ti == 0:
        return(float(-(betap*S0 + mu + psip)+2*E0))
    else:
        y = ode_calc(ti,betap, gammap, psip)
        lamda = betap*y[-1,0]
        phi_value = -(lamda + mu + psip) + 2*y[-1,3]
        return float(phi_value)
    
def prior(x):
    #x[0] = mu, x[1]=sigma (new or current)
    #returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    #returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    #It makes the new sigma infinitely unlikely.
    if(x[0] <= 0 or x[1] <= 0): # for case of three should be or x[2] <= 0
        return 0
    else:
        return 1

def loglikelihood(x,data):
    betap = x[0]#[0]
    gammap = x[1]#0.95 
    psip = 0.2#x[2]
    tend2 = np.max(data)
    get = quad(phi_integrand, 0, tend2,args=(betap,gammap,psip))[0]
    
    for i in np.diagonal(data):
        st = tend2 - i
        res, err = quad(phi_integrand, 0, st, args=(betap,gammap,psip))
        get = get + res + np.log(calc_lamda(st, betap, gammap, psip))
        
    for j in data[:,-1][:-1]:
        st = tend2 - j
        res, err = quad(phi_integrand, 0, st,args = (betap,gammap,psip))
        get = get + np.log(psip) - res
        
    return get

#Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new>x:
        return True
    else:
        accept=np.random.uniform(0,1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        return (accept < (np.exp(x_new-x)))

def metropolis_hastings(param_init,iterations,data):
    # likelihood_computer(x,data): returns the likelihood that these parameters generated the data
    # transition_model(x): a function that draws a sample from a symmetric distribution and returns it
    # param_init: a starting sample
    # iterations: number of accepted to generated
    # data: the data that we wish to model
    # acceptance_rule(x,x_new): decides whether to accept or reject the new sample
    x = param_init
    accepted = []
    rejected = []   
    for i in range(iterations):
        x_new =  transition_model(x)    
        x_lik = loglikelihood(x,data)
        x_new_lik = loglikelihood(x_new,data) 
        if (acceptance(x_lik + np.log(prior(x)),x_new_lik+np.log(prior(x_new)))):            
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)            
                
    return np.array(accepted), np.array(rejected)


S0 = 20 # initial value of S could be 10
I0 = 1 # initial value of I
R0 = 0 # initial value of R
E0 = 1

##################################################
# sample output from gillespie():
dist = [[0.229, 0.215, 0.215, 0.215, 0.215, 0.215, 0.215, 0.215, 0.215],
       [0.215, 0.451, 0.268, 0.268, 0.268, 0.268, 0.268, 0.268, 0.268],
       [0.215, 0.268, 0.567, 0.455, 0.456, 0.502, 0.487, 0.456, 0.456],
       [0.215, 0.268, 0.455, 0.703, 0.455, 0.455, 0.487, 0.455, 0.455],
       [0.215, 0.268, 0.456, 0.455, 0.759, 0.456, 0.487, 0.508, 0.563],
       [0.215, 0.268, 0.502, 0.455, 0.456, 0.82 , 0.487, 0.456, 0.456],
       [0.215, 0.268, 0.487, 0.487, 0.487, 0.487, 1.035, 0.487, 0.487],
       [0.215, 0.268, 0.456, 0.455, 0.508, 0.456, 0.487, 1.11 , 0.508],
       [0.215, 0.268, 0.456, 0.455, 0.563, 0.456, 0.487, 0.508, 1.754]]

dist = np.array(dist)
# here I need to take the S from ODE at the time of matrix elements

transition_model = lambda x: np.random.normal(x,[0.05,5],(2,))

accepted, rejected = metropolis_hastings([0.3,0.7], 50000,dist)

show=int(-0.5*accepted.shape[0])
hist_show=int(-0.50*accepted.shape[0])


fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(1,2,1)
ax.plot(accepted[show:,0])
ax.set_title("Figure 13: Trace for $a$")
ax.set_xlabel("Iteration")
ax.set_ylabel("a")
ax = fig.add_subplot(1,2,2)
ax.hist(accepted[hist_show:,0], bins=20, density=True)
ax.set_ylabel("Frequency (normed)")
ax.set_xlabel("a")
ax.set_title("Figure 14: Histogram of $a$")
fig.tight_layout()

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(1,2,1)
ax.plot(accepted[show:,1])
ax.set_title("Figure 15: Trace for $b$")
ax.set_xlabel("Iteration")
ax.set_ylabel("b")
ax = fig.add_subplot(1,2,2)
ax.hist(accepted[hist_show:,1], bins=20, density=True)
ax.set_ylabel("Frequency (normed)")
ax.set_xlabel("b")
ax.set_title("Figure 16: Histogram of $b$")
fig.tight_layout()

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1,1,1)
xbins, ybins = np.linspace(0.8,1.2,30), np.linspace(75,90,30)
counts, xedges, yedges, im = ax.hist2d(accepted[hist_show:,0], accepted[hist_show:,1], density=True, bins=[xbins, ybins])
ax.set_xlabel("a")
ax.set_ylabel("b")
fig.colorbar(im, ax=ax)
ax.set_title("2D histogram showing the joint distribution of $a$ and $b$")


x = [180,9e+4]
loglikelihood(x, dist)    

