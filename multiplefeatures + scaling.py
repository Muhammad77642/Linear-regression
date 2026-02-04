import matplotlib.pyplot as plt
import numpy as np
x = np.array([
[50,   2,   1,   2000],
[60,   3,   2,   2500],
[75,   4,   2,   3000],
[90,   5,   3,   3500],
[120,  6,   3,   4000],
[150,  7,   4,   5000],
[180,  8,   4,   6500],
[210,  9,   5,   8000],
[250, 10,   6,   10000],
[300, 12,   7,   12000],
[350, 14,   8,   15000],
[420, 16,   9,   18000],
[500, 18,  10,   22000],
[600, 20,  11,   26000],
[750, 22,  12,   30000],
[900, 25,  13,   35000],
[1100, 28, 14,   40000],
[1300, 30, 15,   45000],
[1600, 33, 16,   52000],
[2000, 36, 18,   60000]
], dtype=np.float64)

y = np.array([
100, 120, 150, 180, 220,
260, 310, 360, 420, 500,
580, 670, 780, 900, 1050,
1200, 1400, 1650, 1900, 2200
], dtype=np.float64)
w = np.zeros((4,))
b = 0
alpha = 0.01
iterations  = 100000

def normalization(x):
    mu= np.mean(x ,axis=0)
    sigma = np.std(x,axis=0)
    x_scaled = (x-mu) / sigma
    return x_scaled
x_scaled = normalization(x)
def compute_cost(x,y,w,b):
    m = len(x)
    cost = 0.0
    for i in range (m):
        f_wb = np.dot(x[i],w) + b
        error = (f_wb - y[i])**2
        cost+= error
    cost = cost / (2 * m)
    return cost

def compute_gradient(x,y,w,b):
    m = len(x)
    n = len(x[1])
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range (m):
        f_wb = np.dot(x[i],w) + b
        error = (f_wb - y[i])
        for j in range (n):
            dj_dw [j] += error * x[i,j]
        dj_db += error
    dj_dw = dj_dw / m
    dj_db =dj_db / m
    return dj_dw,dj_db
for i in range (iterations) :
    dj_dw,dj_db = compute_gradient(x_scaled,y,w,b)
    w = w - (alpha * dj_dw)
    b  = b - (alpha * dj_db)
    print(compute_cost(x_scaled,y,w,b))
