import numpy as np
from scipy.optimize import minimize

test = np.arange(13*4).reshape(13, 4)




test2 = test.flatten()





def objfun(x):
    x=x.reshape(13,4)
    max=0
    for i in range(13):
     if(max<sum(x[i,:])/5 ):
         max =sum(x[i,:])/5
    return max
#print(objfun(test2))
A = np.array([
[1, 1, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
[-1, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0],
[0, -1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, -1, 0, 0, 0, 1, 1, 0, 0, -1, 0, 0],
[0, 0, 0, -1, 0, 0, -1, 0, 1, 1, 0, 0, -1],
[0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 1, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0,-1, 1]])

b = np.array([
    [4, 4, 0, 4],
    [-4, 0, -4, 0],
    [0, -4, 4, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, -4]
])

def const1(x):
    x = x.reshape(13, 4)
    return np.linalg.norm(np.matmul(A, x)-b, 2)

# const2 >= 0
def const2(x):
    x = x.reshape(13, 4)
    x = np.sum(x[0, :])
    return 5-x

def const3(x):
    x = x.reshape(13, 4)
    x = np.sum(x[1, :])
    return 5-x

def const4(x):
    x = x.reshape(13, 4)
    x = np.sum(x[2, :])
    return 5-x

def const5(x):
    x = x.reshape(13, 4)
    x = np.sum(x[3, :])
    return 5-x

def const6(x):
    x = x.reshape(13, 4)
    x = np.sum(x[4, :])
    return 5-x

def const7(x):
    x = x.reshape(13, 4)
    x = np.sum(x[5, :])
    return 5-x

def const8(x):
    x = x.reshape(13, 4)
    x = np.sum(x[6, :])
    return 5-x

def const9(x):
    x = x.reshape(13, 4)
    x = np.sum(x[7, :])
    return 5-x
def const10(x):
    x = x.reshape(13, 4)
    x = np.sum(x[8, :])
    return 5-x

def const11(x):
    x = x.reshape(13, 4)
    x = np.sum(x[9, :])
    return 5-x
def const12(x):
    x = x.reshape(13, 4)
    x = np.sum(x[10, :])
    return 5-x
def const13(x):
    x = x.reshape(13, 4)
    x = np.sum(x[11, :])
    return 5-x

def const14(x):
    x = x.reshape(13, 4)
    x = np.sum(x[12, :])
    return 5-x


cons = (
    {'type': 'eq', 'fun': const1},
    {'type': 'ineq', 'fun': const2},
    {'type': 'ineq', 'fun': const3},
    {'type': 'ineq', 'fun': const4},
    {'type': 'ineq', 'fun': const5},
    {'type': 'ineq', 'fun': const6},
    {'type': 'ineq', 'fun': const7},
    {'type': 'ineq', 'fun': const8},
    {'type': 'ineq', 'fun': const9},
    {'type': 'ineq', 'fun': const10},
    {'type': 'ineq', 'fun': const11},
    {'type': 'ineq', 'fun': const12},
    {'type': 'ineq', 'fun': const13},
    {'type': 'ineq', 'fun': const14},
)



# 设置每一个点的范围，最小是0， 没有最大值，所以是None
bnds = [(0, None) for i in range(13*4)]



res = minimize(objfun, x0 =np.zeros(shape=(13*4)), method='SLSQP', bounds=bnds, constraints=cons,options={'maxiter':10000, 'ftol':0.0001})
k=np.around(res['x'],decimals=3).reshape(13,4)
print('the solution is\n',np.around(res['x'],decimals=3).reshape(13,4))
print(np.linalg.norm(np.matmul(A, k)-b, 2))
print("the final value is ",np.around(res['fun'],decimals=3))
