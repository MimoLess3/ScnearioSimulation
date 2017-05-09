
# coding: utf-8

# In[1]:

import numpy as np
import csv
import pandas 

csv = pandas.read_csv(filepath_or_buffer="DataDaily.csv", sep=";")
csv


# In[2]:

mean = csv.mean()


# In[3]:

covmatr = csv.cov()


# In[4]:

matrixCov = covmatr.as_matrix()

matrixCov.shape
eigenvalues, eigenvectors = np.linalg.eigh(matrixCov)


# In[7]:

size = eigenvalues.size

eigenvalues[size - 3:size]


# In[9]:

#соответсвующие собственные векора eignvertor[]

b3 = eigenvectors[:, size-3]
lambda3 = eigenvalues[size-3]

b2 = eigenvectors[:, size-2]
lambda2 = eigenvalues[size-2]

b1 = eigenvectors[:, size-1]
lambda1 = eigenvalues[size-1]

print(b1)
print(lambda1)
print("------------")
print(b2)
print(lambda2)
print("------------")
print(b3)
print(lambda3)
print("------------")


# In[10]:

fac1 = [-2.45, -1.63, -0.82, 0, 0.82, 1.63, 2.45]
prob1 = [0.016, 0.094, 0.234, 0.313, 0.234, 0.094, 0.016]

fac2 = [-2, -1.000, 0.000 ,1.000 ,2.000]
prob2 = [0.062, 0.250, 0.370, 0.250, 0.062]

fac3 = [-1.410, 0.000, 1.410]
prob3 = [0.25, 0.5, 0.25]


# In[13]:

#iter method
import scipy.stats as stats

delta = 0.1
eps = 0.001

def iterStep(x, n, val):
    flag = False
    x[0] = val
    x[1] = invF(2*F(x[0]))
    for i in range(1, n-1):
        odd(i, x)
        even(i, x, flag)
    odd(n-1, x)
    print(G(x,n))
    
def IterMet(n, delta):
    x = np.zeros(2*n-1)
    flag = False
    iterStep(x, n, invF(2 ** (-n)))
    valueG = G(x, n)
    prevG = valueG
    while (abs(valueG) > eps or delta < 0.001):
        valueG = G(x, n)
        if (prevG*valueG < 0):
            delta /= 2
        prevG = valueG
        if (valueG > 0):
            iterStep(x, n, x[0] + delta)
        else:
            iterStep(x, n, x[0] - delta)
    print("Итоговая точность")
    print(G(x,n))
    print()
    return x

def invF(x):
    return stats.norm.ppf(x)

def F(x):
    return stats.norm.cdf(x)
    
def odd(i, x):
    x[2*i] =  2 * x[2*i-1] - x[2*i - 2]
    
def even(i, x, flag):
    value = -F(x[2*i-1]) + 2 * F(x[2*i])
    if (value > 1):
        print("error")
        print(2 * i + 1)
        print(value)
        flag = True
    if (value < 0):
        print("error")
        print(2 * i + 1)
        print(value)
        flag = True
    x[2*i + 1] = invF(value)
    
def G(x, n):
    return F(x[2*n-3]) - 2*F(x[2*n-2]) + 1

x = IterMet(2, delta)
a3 = x[:3:2]

x = IterMet(4, delta)
a2 = x[:7:2]

x = IterMet(6, delta)
a1 = x[:11:2]

print()
print(a1)
print(a2)
print(a3)


# In[14]:

def probV(a, n):
    prob = np.zeros(n+1)
    prob[0] = F(a[0])
    for i in range(1,n):
        prob[i] = F(a[i]) - F(a[i-1])
    prob[n] = 1 - F(a[n-1])
    return prob


# In[15]:

probvecUpd1 = probV(a1, 6)


# In[16]:

probvecUpd2 = probV(a2, 4)


# In[17]:

probvecUpd3 = probV(a3, 2)


# In[18]:

import scipy.integrate as integrate
from scipy.integrate import quad

def integrand(x):
    return x*stats.norm.pdf(x)

def findvalue(n, a, b):
    x = np.zeros(n+1)
    x[0] , err = quad(integrand, a=-np.inf, b=a[0]) / b[0]
    for i in range (1,n):
        x[i], err =  quad(integrand, a=-a[i-1], b=a[i]) / b[i]
    x[n] , err =  quad(integrand, b=np.inf, a=a[n-1]) / b[n]
    return x

val1 = findvalue(6, a1, probvecUpd1)
val2 = findvalue(4, a2, probvecUpd2)
val3 = findvalue(2, a3, probvecUpd3)

val1 = val1.tolist()
probvecUpd1 = probvecUpd1.tolist()

val2 = val2.tolist()
probvecUpd2 = probvecUpd2.tolist()

val3 = val3.tolist()
probvecUpd3 = probvecUpd3.tolist()

print(val1)
print(probvecUpd1)
print()
print(val2)
print(probvecUpd2)
print()
print(val3)
print(probvecUpd3)


# In[19]:

csvmodel = pandas.read_csv(filepath_or_buffer="2k15day.csv", sep=";", index_col=0)
csvmodel


# In[20]:

times = csvmodel.index.size - 1
times


# In[21]:

import matplotlib.pyplot as plt
matrModel = csvmodel.as_matrix()

d = np.zeros(matrModel[0].size)
for i in range(0, len(d)):
    d[i] = 1000 / matrModel[0][i]
d


# In[24]:

P = np.zeros(len(matrModel))
deltaP = np.zeros(len(P) - 1)

for idx in range(0, len(matrModel)):
    for pos in range(0, len(matrModel[idx])):
        P[idx] += matrModel[idx][pos] * d[pos]

        
for idx in range(0, len(P) - 1):
    deltaP[idx] = round(P[idx+1] - P[idx])
    
deltaP

df = pandas.DataFrame({'DeltaP' : deltaP}, index=csvmodel.index[:times])


# In[25]:

#монте-карло
import math as math

def monteF():
    w1 = np.random.normal()
    w2 = np.random.normal()
    w3 = np.random.normal()
    arr = np.zeros(mean.size)
    for indx in range(0, mean.size):
        arr[indx] = montefunc(indx, w1, w2, w3)
    return arr
        

def montefunc(idx, w1, w2, w3):
    return math.exp(mean[idx] + math.sqrt(lambda1) * b1[idx] * w1 + 
                   math.sqrt(lambda2) * b2[idx] * w2 +
                   math.sqrt(lambda3) * b3[idx] * w3)


# In[26]:

def func(prob1, fac1):
    rand = np.random.uniform()
    sum1 = 0
    for prob in prob1:
        sum1 += prob
        if rand < sum1:
            return fac1[prob1.index(prob)]
    return fac1[len(prob1) - 1]


# In[27]:

def funcScenario(prob1, fac1, prob2, fac2, prob3, fac3):
    w1 = func(prob1, fac1)
    w2 = func(prob2, fac2)
    w3 = func(prob3, fac3)
    arr = np.zeros(mean.size)
    for indx in range(0, mean.size):
        arr[indx] = montefunc(indx, w1, w2, w3)
    return arr


# In[28]:

#for Var
count = 500
shortfall = np.zeros(times)
alpha = 0.05
p =round(count * alpha)

varScenario = np.zeros(times)
for j in range(0,times):
    valueofP = []
    for i in range(0, count):
        valueofP.append(funcScenario(prob1, fac1,prob2,fac2,prob3,fac3))
    startv = matrModel[j] * d
    
    startp = np.dot(startv, np.ones(matrModel[j].size))

    value = np.zeros(count)

    for i in range(0, count):
        value[i] = np.dot(valueofP[i], startv)

    value.sort()
    varScenario[j] = value[p] - startp
    for i in range(0, p):
        shortfall[j] += value[i] - startp
    shortfall[j] /= p


# In[29]:

#for Var Updated
countf2 = 500

shortfallUpd = np.zeros(times)

varScenarioUpd = np.zeros(times)
for j in range(0,times):
    valueofP = []
    for i in range(0, count):
        valueofP.append(funcScenario(probvecUpd1, val1,probvecUpd2,val2,probvecUpd3,val3))
    startv = matrModel[j] * d
    
    startp = np.dot(startv, np.ones(matrModel[j].size))

    value = np.zeros(count)

    for i in range(0, count):
        value[i] = np.dot(valueofP[i], startv)

    value.sort()
    varScenarioUpd[j] = value[p] - startp
    for i in range(0, p):
        shortfallUpd[j] += value[i] - startp
    shortfallUpd[j] /= p

    


# In[30]:

var1 = -varScenario
var2 = -varScenarioUpd
deltap1 = -deltaP
for i in range(0, len(deltap1)):
    if deltap1[i] < 0:
        deltap1[i] = 0
title = "Var " + str(alpha)
df1 = pandas.DataFrame({'varScenario' : var1, 'varScenarioUpd' : var2, 'Loses' : deltap1 }, index=csvmodel.index[:times])
df1
df1.plot(kind="line", title=title)
plt.show()


# In[31]:

var1sh = -shortfall
var2sh = -shortfallUpd
deltap1 = -deltaP
for i in range(0, len(deltap1)):
    if deltap1[i] < 0:
        deltap1[i] = 0

df1 = pandas.DataFrame({'shortfall' : var1sh, 'shorfallUpd' : var2sh, 'Loses' : deltap1 }, index=csvmodel.index[:times])
df1
df1.plot(kind="line", title="Shortfall " + str(alpha))
plt.show()



# In[32]:

m2 =df1.shorfallUpd.mean()


# In[33]:

m1 = df1.shortfall.mean()


# In[34]:

dict1 = {'mean' : [m1, m2]}


# In[35]:

df3 = pandas.DataFrame(dict1, index=["shortfall", "shortfallUpd"] )
df3


# In[ ]:



