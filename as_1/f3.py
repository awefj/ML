import numpy as np
import nd

def func3(x):
    return x[0]*x[1] + x[1]*x[2]*x[3] + 3*x[0] + x[3]*(x[2]**2)
f = lambda x : func3(x)
x = np.array([1.0, 2.0, 3.0, 4.0])
ret = nd.numerical_derivative(f,x)
print('type(ret) = ', type(ret), ', ret_val = ', ret)