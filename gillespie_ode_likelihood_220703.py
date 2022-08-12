import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.integrate import quad
import math

# a function that caluclates the distnace matrix

def dist_calc(e):
    # varaibles are defined global, the function doesn't have a return value
    global dist, child, diagonal, sampled, tau
    branch = -1
    if e == 1:
        if child[-1] == 1:
            dist.append(tau)
        elif child[-1] == 2:
            diagonal = dist[0]+tau
            dist = [[dist[0]+tau,dist[0]],[dist[0],dist[0]+tau]]
        elif child[-1] > 2:
            # choosing one of the branches randomly
            branch = random.choice(list(set(child[:-2])-set().union(recovered, sampled)))
            #print(branch)
            # adding column of branch to the end of the matrix         
            for j in range(child[-1]-1):
                dist[j].append(dist[j][branch])
            # adding branch row to the last 
            last_row =  dist[branch][slice(0,child[-1]-1)] # last row is all elements of the first row except its last
            diagonal += tau
            last_row.append(diagonal)
            dist.append(last_row) 
            # add delta t to diagonal of dist for those who have not been sampled     
            for i in range(child[-1]-1):
                if i not in sampled and i not in recovered:
                   dist[i][i] = dist[i][i] + tau
    else:
        for i in range(child[-1]):
            if i not in sampled and i not in recovered:
               dist[i][i] = dist[i][i] + tau
        
    print("sampled is: ",sampled)
    print("recovered is: ",recovered)
    print("branch is: ",branch)
    print("tau is: ",tau)
    print("dist is: ",dist)   


def gillespie():
    global t,beta,gamma,psi,S,I,R,child,recovered,sampled,tau
    # main loop to do the gillespie  
    while t[-1] < tend: # the loop iterates until it reaches the last time point
     
            # propensities (rates)
            props = [beta * S[-1] * I[-1] , gamma * I[-1], psi*I[-1]] 

            # sum of propensities
            prop_sum = sum(props)

            if prop_sum == 0:
                    break
            # drawing a tau time from an exponential distribution 
            tau = round(np.random.exponential(scale=1/prop_sum),3)
            #tau = np.random.exponential(scale=1/prop_sum)
            #tau = round(random.random(),3)

            # updating time list 
            t.append(t[-1] + tau)

            # drawing a ranodom number from a unifrom dist.
            rand = random.uniform(0,1)
              
            # the first event is infection of a susceptible 
            if rand * prop_sum <= props[0]:
                
                    S.append(S[-1] - 1) # updating S
                    I.append(I[-1] + 1) # updating I
                    
                    child.append(child[-1] + 1) # number of children is 1 less than infected
                    dist_calc(1) # calling the dist_calc() updates the distance matrix without a return value
                    
                    R.append(R[-1]) # updating R
                    
            # second event is recovery of an infected 
            elif rand * prop_sum > props[0] and rand * prop_sum <= props[0] + props[1] and rand * prop_sum < props[1] + props[2]:

                    S.append(S[-1])  
                    R_ind = random.choice(list(set(child[:-2]) - set().union(recovered, sampled)))
                    recovered.append(R_ind)
                    I.append(I[-1] - 1)
                    R.append(R[-1] + 1)
                    dist_calc(2)
                    
            # third event is sampling of an infected
            elif rand * prop_sum >  props [1] and rand * prop_sum <= props[0] + props[1] + props[2]:
         
                    S.append(S[-1])  
                    Samp_ind = random.choice(list(set(child[:-2]) - set().union(recovered, sampled)))
                    sampled.append(Samp_ind) # adding the sampled individual to the sampled list
                    I.append(I[-1] - 1)
                    R.append(R[-1] + 1)
                    dist_calc(3)
    
# a recursive matrix to break the matrix 
def convert_newick(mat):
    if np.shape(mat)[0] == 1:
        #return(":"+str(mat[0][0]))
        return("xAz:" + str(mat[0][0]))
    elif np.shape(mat)[0] == 2:
        new_mat = mat - np.amin(mat)
        # dv collects non zero elements of the new mat 
        dv = new_mat[np.nonzero(new_mat)]
        #return("(:"+str(dv[0])+",:"+str(dv[1])+"):"+str(np.amin(mat)))
        return("(xAz:" + str(dv[0]) + ",xAz:" + str(dv[1]) + "):" + str(np.amin(mat)))
    elif np.shape(mat)[0] > 2:
        branch_length =  np.amin(mat)
        # substracting min value of all elements
        newm = mat - branch_length
        out = break_matrix(newm)
        print("out is:",out[0],"and: ",out[1])
        try:
            return("("+convert_newick(out[0])+","+convert_newick(out[1])+"):"+str(branch_length))
        except:
            print("operation for",out,"is not possible")
        #return("(" + convert_newick(out[0])  + "," + convert_newick(out[1]) + "):" + str(branch_length))


# break matrix breaks the matrix to two matrices
def break_matrix(mat):
    mat2 = copy.deepcopy(mat)
    k = []
    for i in range(np.shape(mat2)[0]):
        if mat2[0][i] == 0:
            k.append(i)
        #print(i)
    m1 = np.delete(mat2,k,1)
    m1 = np.delete(m1,k,0)
    m2 = mat[np.ix_(k,k)]
    output = [m1,m2]
    return(output)

# toNweick outputs the final result
def toNewick(dis_matrix):
    out = convert_newick(dis_matrix)
    return("("+out+")xA0z;")


def add_label(textf):
    j = 1
    textl = list(textf)
    for i in range(0,len(textl)):
        #print(i)
        if textl[i] == 'A':
            textl.insert(i+1,j)
            label_list.append("A"+str(j))
            j += 1
            
    label_list.append("A0")
    text2 = ''.join(map(str, textl))
    return(text2)

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

    return([dSdt, dIdt, dRdt, dEdt])


# Our integral approximation function
def integral_approximation(f, a, b):
    return((b-a)*np.mean(f))


def ode_calc(ti,betap,gammap,psip):
    global S0, I0, R0, E0
    y0 = [S0,I0,R0,E0]
    params = [betap,gammap,psip] 
    time_of_simulation = np.linspace(0,ti,num=1000)
    y = odeint(ode_sim,y0, time_of_simulation, args=(params,))
    return(y)
    
def calc_lamda(ti,betap,gammap,psip):
    global newt2,tend2
    index_S = min(range(len(newt2)), key=lambda i: abs(newt2[i]-ti))
    y = ode_calc(tend2,betap, gammap, psip)
    lamda = betap*y[index_S,0]
    return(lamda)

# this function takes ti: time from matrix
def phi_integrand(ti,betap,gammap,psip):
    global newt2,tend2
    y = ode_calc(tend2,betap, gammap, psip)
    intpl = interp1d(newt2, y[:,3], kind = 'cubic')
    lamda = calc_lamda(ti,betap,gammap,psip) 
    mu = gammap
    phi_value = -(lamda + mu + psi) + 2*intpl(ti)
    return(float(phi_value))

# a function to calculate the likelihood
def likelihood(distp,betap,gammap,psip):
    tend2 = np.max(distp)
    prd = math.exp(quad(phi_integrand, 0, tend2,args=(betap,gammap,psip))[0])
    
    for i in np.diagonal(distp):
        st = tend2 - i
        res, err = quad(phi_integrand, 0, st, args=(betap,gammap,psip))
        prd = prd*calc_lamda(st,betap,gammap,psip)*math.exp(res)
        #print(i)
        
    for j in distp[:,-1][:-1]:
        st = tend2 - j
        res, err = quad(phi_integrand, 0, st,args = (betap,gammap,psip))
        prd = (prd*psip)/math.exp(res)
    return(prd)
    
# a function to calculate the loglikelihood
def loglikelihood(distp,betap,gammap,psip):
    tend2 = np.max(distp)
    get = quad(phi_integrand, 0, tend2,args=(betap,gammap,psip))[0]
    
    for i in np.diagonal(distp):
        st = tend2 - i
        res, err = quad(phi_integrand, 0, st, args=(betap,gammap,psip))
        get = get + res + np.log(calc_lamda(st, betap, gammap, psip))
        
    for j in distp[:,-1][:-1]:
        st = tend2 - j
        res, err = quad(phi_integrand, 0, st,args = (betap,gammap,psip))
        get = get + np.log(psip) - res
    return(get)

S0 = 20 # initial value of S could be 10
I0 = 1 # initial value of I
R0 = 0 # initial value of R
E0 = 1

S = [S0] # this list keeps the S number in each update 
I = [I0] # this list keeps the I number in each update
R = [R0] # this list keeps the R number in each update

dist = [] # distance matrix

child = [0] # a list to keep the index of children, I don't remove any element from this list, so it's in order
sampled = [] # a list to keep the sampled indivduals 
recovered = [] # a list to keep the recovered individuals 
diagonal = 0 # a list to keep the diagonal elements of the distance matrix
tau = -1 

t = [0] # a list to keep the time

tend = 2 # last time point could be 30

beta=0.5 # beta parameter
gamma=0.95 # gamma paramater
psi = 0.2 # psi parameter 

# calling the gillespie algorithm
gillespie()
                
dist_array = np.array(dist)
# deleting un-sampled ones from distance matrix
dist_array = dist_array[np.ix_(sampled,sampled)]

dist3 = copy.deepcopy(dist)
text = toNewick(dist_array)
text

label_list = []

text2 = add_label(text)

y0 = [S0,I0,R0,E0]

newt = np.linspace(0,tend,num=1000)

params = [beta,gamma,psi] 

y = odeint(ode_sim,y0, newt, args=(params,))

S_plot, = plt.plot(t,S, label="S")
I_plot, = plt.plot(t,I, label="I")
R_plot, = plt.plot(t,R, label="R")

plt.legend(handles=[S_plot, I_plot, R_plot])

# plot of the ODE simulation 
plt.xlabel("Time")
plt.ylabel("Population")

plt.plot(newt,y[:,0], color = 'blue') # S
plt.plot(newt,y[:,1], color = 'orange') # I
plt.plot(newt,y[:,2], color = 'green') # R


plt.show()

# interpolation of E(t) 
intpl = interp1d(newt, y[:,3], kind = 'cubic')
plt.plot(newt,y[:,3], color = 'black') #E
plt.plot(newt,intpl(newt), color = 'red') #E
plt.show()

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

tend2 = np.max(dist)
#tend2 = tend
newt2 = np.linspace(0,tend2,num=1000)

betap=beta # beta parameter
gammap=gamma # gamma paramater
psip = psi# psi parameter 

# using defined integral_approximation():
# Generate function values
x_range = np.arange(0,tend+0.0001,.0001)
fx = [phi_integrand(x,betap,gammap,psip) for x in x_range]
approx = integral_approximation(fx,0,tend)
approx

LL = likelihood(dist, betap, gammap, psip)        
LLL= loglikelihood(dist, betap, gammap, psip)


from scipy import optimize
def sampfunc(a,b,c):
    return(np.square(a) + np.cos(b) - np.sqrt(c))

# extra two 0's as dummy equations as root solves a system of equations 
# rather than single multivariate equation
def func(x):                                        # A,B,C represented by x ndarray
    return [np.square(x[0]) + np.cos(x[1]) - np.sqrt(x[2]) - 1.86, 0, 0]


# optimzing for one parameter
def func(x):                                        # A,B,C represented by x ndarray
    return [np.square(1.09) + np.cos(x[0]) - np.sqrt(0.069) - 1.86]

result = optimize.root(func , x0 = [0.1])
x = result.x
A, B, C = x                       
x

def loglikelihood2(x):
    global dist
    betap = x#[0]
    gammap = 0.95#x[1]
    psip = 0.2#x[2]
    tend2 = np.max(dist)
    get = quad(phi_integrand, 0, tend2,args=(betap,gammap,psip))[0]
    
    for i in np.diagonal(dist):
        st = tend2 - i
        res, err = quad(phi_integrand, 0, st, args=(betap,gammap,psip))
        get = get + res + np.log(calc_lamda(st, betap, gammap, psip))
        
    for j in dist[:,-1][:-1]:
        st = tend2 - j
        res, err = quad(phi_integrand, 0, st,args = (betap,gammap,psip))
        get = get + np.log(psip) - res
    #return[get - LLL]
    return get


#import GPyOpt
from GPyOpt.methods import BayesianOptimization

domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]


myBopt_1d = BayesianOptimization(f=loglikelihood2, domain=domain)
print("The minumum value obtained by the function was %.4f (x = %.4f)" % (myBopt_1d.fx_opt, myBopt_1d.x_opt))
myBopt_1d.run_optimization(max_iter=5)
myBopt_1d.plot_acquisition()

# this is working
result = optimize.root(loglikelihood2 , x0 = [0.4])
x = result.x
A, B, C = x                       
x
from scipy.optimize import fsolve

def vf(x):
    return [loglikelihood2(x), 0]

root = fsolve(vf, x0=[0.4,0.8])
# this is working
root = fsolve(loglikelihood2, x0=[0.4])

from scipy.optimize import minimize
x0 = np.array([0.4,0.7])
res = minimize(loglikelihood2, x0, method='nelder-mead')
res = minimize(loglikelihood2, x0, method='basinhopping')


LLL2 = loglikelihood2(x0)

from scipy.optimize import basinhopping

ret = basinhopping(loglikelihood2, x0, niter=200)

