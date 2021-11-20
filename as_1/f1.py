import numpy as np
import nd

def func1(x):
    return x**2
f = lambda x : func1(x)
x = np.array([3.0])
ret = nd.numerical_derivative(f,x)
print('type(ret) = ', type(ret), ', ret_val = ', ret)