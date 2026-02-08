import numpy as np
import matplotlib.pyplot as plt
x = np.loadtxt("Ice_cream selling data.csv",delimiter=",",usecols=0,skiprows=1)
y = np.loadtxt("Ice_cream selling data.csv",delimiter=",",usecols=1,skiprows=1)
x_powered = np.column_stack((x,x**2,x**3,x**4))
w = [-4.30770763 , 14.40630922 , 2.26006056, -3.09766957]
b = 15.905307840911915
iterations = 100000
alpha = 0.01
def normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    x_scaled = (x-mu) / sigma
    return x_scaled
x_normalized = normalization(x_powered)
def compute_cost(x,y,w,b):
    m = len(x)
    f_wb = x @ w + b
    error = (f_wb - y)**2
    cost = np.sum(error) / (2*m)
    return cost
def compute_gradient(x,y,w,b):
    m = len(x)
    f_wb = x @ w + b
    error = (f_wb - y)
    dj_dw = (x.T @ error) / m
    dj_db = np.sum(error)
    return dj_dw,dj_db
# for i in range (iterations):
#     dj_dw,dj_db = compute_gradient(x_normalized,y,w,b)
#     w -= (alpha * dj_dw)
#     b -= (alpha * dj_db)
#     print(compute_cost(x_normalized,y,w,b))
f_new = x_normalized @ w +b
plt.plot(x,f_new)
plt.scatter(x,y)
plt.show()