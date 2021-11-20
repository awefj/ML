import numpy as np

def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    # debug 1, 2, 3 -- print initial x, grad values
    print("debug 1. initial input = ", x)
    print("debug 2. initial grad = ", grad)
    print("===========================================")

    it=np.nditer(x,flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        # debug 3 -- print grad values
        print("debug 3. idx = ", idx, "x[idx] = ",x[idx])
        tmp_val = x[idx] #temp val for swap
        x[idx] = float(tmp_val) + delta_x
        fx1=f(x) #f(x+delta_x)
        x[idx] = float(tmp_val) - delta_x
        fx2=f(x) #f(x-delta_x)
        grad[idx] = (fx1-fx2)/(delta_x*2)
        # debug 4,5
        print("debug 4. grad[idx] = ", grad[idx])
        print("debug 5. grad = ", grad)
        print("===========================================")
        x[idx] = tmp_val #data restore
        it.iternext()
    return grad