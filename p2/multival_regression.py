import numpy as np

data = np.loadtxt('./data-01-test_score.csv', delimiter=',', dtype=np.float32)
x_data = data[:, 0:-1]
t_data = data[:, [-1]]

W = np.random.rand(3,1)
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

def loss_func(x,t):
    y = np.dot(x,W)+b
    return (np.sum((t-y)**2))/(len(x))

def predict(x):
    y = np.dot(x,W)+b
    return y

learning_rate = 1e-5
f = lambda x: loss_func(x_data, t_data)
print("initial error value = ", loss_func(x_data, t_data), "initial W = ", W, "\n", ", b = ", b)
for step in range(10001):
    W-= learning_rate * numerical_derivative(f,W)
    b-= learning_rate * numerical_derivative(f,b)
    if(step%400==0):
        print("step = ", step, "error value = ", loss_func(x_data, t_data), "\n" ,"W = ", W, " b = ", b)

print("W = ", W, ", b = ", b)

test_data = np.array([100, 98, 81])
print("test data = ", test_data)
print("predict(test_data) = ", predict(test_data))