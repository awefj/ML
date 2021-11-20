import numpy as np

x_data = np.array([1,2,3,4,5]).reshape(5,1)
t_data = np.array([2,3,4,5,6]).reshape(5,1)

W = np.random.rand(1,1)
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
    y = np.dot(x,W) + b
    return (np.sum((t-y)**2))/(len(x))

def error_val(x, t):
    y = np.dot(x,W)+b
    return (np.sum((t-y)**2))/(len(x))

def predict(x):
    y = np.dot(x, W) + b
    return y

learning_rate = 1e-2

f = lambda x : loss_func(x_data, t_data)
#print("initial error value = ", error_val(x_data, t_data), "initial W = ", W, "\n", ", b = ", b)
print("initial error value = ", loss_func(x_data, t_data), "initial W = ", W, "\n", ", b = ", b)
for step in range(8001):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)

    if (step%400==0):
        #print("step = ", step, "error value = ", error_val(x_data, t_data), "W = ", W, ",b = ", b)
        print("step = ", step, "error value = ", loss_func(x_data, t_data), "W = ", W, ",b = ", b)

print("predict(43) = ", predict(43))

