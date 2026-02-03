import matplotlib.pyplot as plt
import numpy as np
x = np.array([
5, 8, 12, 15, 18, 22, 25, 30, 34, 38,
42, 47, 53, 58, 63, 69, 74, 80, 86, 93,
101, 110, 120, 131, 143, 156, 170, 185, 201, 218,
236, 255, 275, 296, 318, 341, 365, 390, 416, 443
], dtype=np.float64)
y = np.array([
28, 35, 41, 45, 52, 60, 63, 70, 78, 83,
90, 97, 105, 111, 118, 125, 132, 140, 149, 157,
166, 175, 186, 198, 209, 222, 235, 249, 263, 278,
294, 310, 327, 345, 364, 384, 405, 427, 450, 474
], dtype=np.float64)
alpha = 0.00001
iterations = 1000000

def compute_cost(x,y,w,b):
    m = len(x)
    cost = 0
    for i in range (m) :
        f_wb = w * x[i] + b
        cost+= (f_wb - y[i])**2
    total_cost = cost / (2 * m)
    return total_cost

def compute_gradient(x,y,w,b):
    m = len(x)
    dj_dw = 0
    dj_db = 0
    for i in range (m) :
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw,dj_db
for i in range (iterations):
    dj_dw,dj_db = compute_gradient(x,y,w,b)
    w = w - (alpha * dj_dw)
    b = b - (alpha * dj_db)
    print(compute_cost(x,y,w,b))
print(f"final w : {w} \n final b : {b}")





