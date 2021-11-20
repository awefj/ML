import numpy as np
import nd

def func2(x):
    return 2*x[0] + 3*x[0]*x[1] + (x[1]**3)
f = lambda x : func2(x)
x = np.array([1.0, 2.0])
ret = nd.numerical_derivative(f, x)
print('type(ret) = ', type(ret), ', ret_val = ', ret)