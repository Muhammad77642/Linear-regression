import matplotlib.pyplot as plt
import numpy as np
x = np.array([
[1, 2, 1],
[2, 1, 3],
[3, 2, 2],
[4, 3, 1],
[5, 2, 4],
[6, 3, 3],
[7, 4, 2],
[8, 5, 1],
[9, 4, 5],
[10, 5, 4],
[11, 6, 3],
[12, 7, 2],
[13, 6, 6],
[14, 7, 5],
[15, 8, 4],
[16, 9, 3],
[17, 8, 7],
[18, 9, 6],
[19, 10, 5],
[20, 11, 4],
[21, 10, 8],
[22, 11, 7],
[23, 12, 6],
[24, 13, 5],
[25, 12, 9],
[26, 13, 8],
[27, 14, 7],
[28, 15, 6],
[29, 14, 10],
[30, 15, 9]
], dtype=np.float64)

y = np.array([
15, 16, 20, 23, 27, 30, 33, 36, 40, 43,
46, 49, 53, 56, 59, 62, 66, 69, 72, 75,
79, 82, 85, 88, 92, 95, 98, 101, 105, 108
], dtype=np.float64)
w = [2.71697906 ,0.83919046 ,  0.50227729]
b = 9.467578410818566
f_new = np.dot(x,w) + b
alpha = 0.001
rmse = np.sqrt(np.mean(f_new - y )**2)
print("RMSE" , rmse)
iterations = 1000000
def compute_cost(x,y,w,b):
    m = len(x)
    cost = 0.0
    for i in range(m) :
        f_wb = np.dot(x[i],w) + b
        error = (f_wb - y[i])**2
        cost += error
    cost = cost / (2*m)
    return cost

def compute_gradient (x,y,w,b):
    m = len(x)
    n = len(x[1])
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        f_wb = np.dot(x[i],w) + b
        error = (f_wb - y[i] )
        for j in range (n) :
            dj_dw[j] +=  error * x[i,j]
        dj_db += error
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return  dj_dw,dj_db
for i in range (iterations) :
    dj_dw , dj_db = compute_gradient(x,y,w,b)
    w = w - (alpha * dj_dw)
    b = b - (alpha * dj_db)
    print(compute_cost(x,y,w,b))
print(f"optimal value of w : {w} optimal value of b : {b}")
