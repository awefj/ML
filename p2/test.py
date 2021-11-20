import datetime

import numpy as np
try:
    data = np.loadtxt('./sps.csv', delimiter=',', dtype=np.float32)
    x_data = data[:, 1:]
    t_data = data[:, [0]]

    #data shape
    print("data.ndim = ", data.ndim, ", data.shape = ", data.shape)
    print("x_data.ndim = ", x_data.ndim, ", x_data.shape = ", x_data.shape)
    print("t_data.ndim = ", t_data.ndim, ", t_data.shape = ", t_data.shape)

except FileNotFoundError as err:
    print(str(err))
except IndexError as err:
    print(str(err))
except Exception as err:
    print(str(err))

W = np.random.rand(x_data.shape[-1], 1)
b = np.random.rand(1)
print("W = ", W, ", W.shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)

def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    # debug 1, 2, 3 -- print initial x, grad values
    #print("debug 1. initial input = ", x)
    #print("debug 2. initial grad = ", grad)
    #print("===========================================")

    it=np.nditer(x,flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        # debug 3 -- print grad values
        #print("debug 3. idx = ", idx, "x[idx] = ",x[idx])
        tmp_val = x[idx] #temp val for swap
        x[idx] = float(tmp_val) + delta_x
        fx1=f(x) #f(x+delta_x)
        x[idx] = float(tmp_val) - delta_x
        fx2=f(x) #f(x-delta_x)
        grad[idx] = (fx1-fx2)/(delta_x*2)
        # debug 4,5
        #print("debug 4. grad[idx] = ", grad[idx])
        #print("debug 5. grad = ", grad)
        #print("===========================================")
        x[idx] = tmp_val #data restore
        it.iternext()
    return grad

def loss_func(x, t):
    y = np.dot(x, W) + b
    return (np.sum((t-y)**2))/(len(x))

def predict(x):
    y = np.dot(x,W) + b
    return y

learning_rate = 1e-3
f = lambda x : loss_func(x_data, t_data)
print("initial error value = ", loss_func(x_data, t_data), "initial W = ", W, "\n", ", b = ", b)
start_time = datetime.datetime.now()

for step in range(20001):
    W-=learning_rate*numerical_derivative(f,W)
    b-=learning_rate*numerical_derivative(f,b)
    if(step%1000 == 0):
        print("step = ", step, "error_value = ", loss_func(x_data, t_data))

end_time = datetime.datetime.now()
print("")
print("Elapsed time = ", end_time - start_time)

print("W = ", W, ", b = ", b)

ex_data_01 = np.array([4,4,4,4])
ex_data_02 = np.array([-3,0,9,-1])
ex_data_03 = np.array([-7,-9,-2,8])
ex_data_04 = np.array([1,-2,3,-2])
ex_data_05 = np.array([19,-12,0,-76])
print("predict([4,4,4,4]) = ", predict(ex_data_01))
print("predict([-3,0,9,-1]) = ", predict(ex_data_02))
print("predict([-7,-9,-2,8]) = ", predict(ex_data_03))
print("predict([1,-2,3,-2]) = ", predict(ex_data_04))
print("predict([19,-12,0,-76]) = ", predict(ex_data_05))