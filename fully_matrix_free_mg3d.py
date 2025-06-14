import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, cgs
from scipy.ndimage import convolve
import time 
## python >= 3.11, scipy>=1.41.1，numpy>=2.02 
## A fully matrix free method to solve $-\Delta u=f$ with 0-Dirichlet boundary conditions

## Mesh 
CYCLE = 'N' ## {V, W, N, F, B}, B is backslash-cycle
kk = 8
lev = kk-4

n = 2**kk
N = n+1
NN = n-1
Dof = NN**3
print(f'kk={kk}, n=2^{kk}={n}, lev={lev}, Dof={NN}^3={Dof}, {CYCLE}-cycle')
pts1d = np.linspace(0, 1, N)[1:-1]
h = pts1d[1] - pts1d[0]
r, s, t = np.meshgrid(pts1d, pts1d, pts1d, indexing='ij')

## R P
def PRxyz(x):
    w = np.array([[[1./8, 1./4, 1./8], [1./4, 1./2, 1./4], [1./8, 1./4, 1./8]],
                  [[1./4, 1./2, 1./4], [1./2, 1.  , 1./2], [1./4, 1./2, 1./4]],
                  [[1./8, 1./4, 1./8], [1./4, 1./2, 1./4], [1./8, 1./4, 1./8]]])
    return convolve(x, w)
def R(x):
    v = PRxyz(x)
    return v[1::2, 1::2, 1::2]/8
def P(x):
    a1,a2,a3 = x.shape
    y = np.zeros((a1*2+1, a2*2+1, a3*2+1))
    y[1::2, 1::2, 1::2] = x
    return PRxyz(y)*4  # right-hand h^2  ~ (2h)^2, then 4 times 

## stencil
def AA(x):
    Ax = 6*x
    Ax[1:] -= x[:-1]
    Ax[:-1] -= x[1:]
    Ax[:,1:] -= x[:,:-1]
    Ax[:,:-1] -= x[:,1:]
    Ax[:,:,1:] -= x[:,:,:-1]
    Ax[:,:,:-1] -= x[:,:,1:]
    return Ax
def AA2(x):
    sz = x.shape
    g = lambda v: AA(v.reshape(sz)).flatten()
    df = sz[0]*sz[1]*sz[2]
    LA2 = LinearOperator((df, df), g)
    return cg(LA2, x.flatten(), atol=1e-12)[0].reshape(sz)

## mg
def vcycle(f, k=1, kH=lev):
    if k == kH:
         return AA2(f)
    wd = (0.8/6)  # 0.6 ~ sor coef,   /6 ~ diag(A)^{-1}
    u = wd * f 
    u += P(mg(R(f -AA(u)), k+1, kH))
    u += wd * (f - AA(u))
    return u

def mg(f, k=1, kH=lev):
    if k == kH:
         return AA2(f)
    wd = (0.8/6)  # 0.6 ~ sor coef,   /6 ~ diag(A)^{-1}
    if CYCLE == 'B':
        u = wd * f 
        u += P(mg(R(f -AA(u)), k+1, kH))
        return u
    if CYCLE == 'N':
        u = P(mg(R(f), k+1, kH))
        u += wd * (f - AA(u))
        u += P(mg(R(f -AA(u)), k+1, kH))
        u += wd * (f - AA(u))
        return u
    if CYCLE == 'V':
        u = wd * f 
        u += P(mg(R(f -AA(u)), k+1, kH))
        u += wd * (f - AA(u))
        return u
    if CYCLE == 'W':
        u = wd * f 
        u += P(mg(R(f -AA(u)), k+1, kH))
        u += wd * (f - AA(u))
        u += P(mg(R(f -AA(u)), k+1, kH))
        u += wd * (f - AA(u))
        return u
    if CYCLE == 'F':
        u = wd * f 
        u += P(vcycle(R(f -AA(u)), k+1, kH))
        u += wd * (f - AA(u))
        u += P(vcycle(R(f -AA(u)), k+1, kH))
        u += wd * (f - AA(u))
        return u
    
## Linear Operator
class count:
    def __init__(self):
        self.step = 0
ct = count()
def AA0(x):
    ct.step += 1
    x = x.reshape((NN, NN, NN))
    return mg(AA(x)).flatten()
LA = LinearOperator((Dof, Dof), AA0)

## Solve 
u = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) + x*(1-x)*y*(1-y)*z*(1-z)
f = lambda x, y, z: 3*np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) + 2*y*(1-y)*z*(1-z) + 2*x*(1-x)*z*(1-z) + 2*x*(1-x)*y*(1-y)
ui = u(r, s, t)
fi = f(r, s, t)
pfi = mg(fi)
tt = time.time()
solution, exitCode = cgs(LA, (h*h)*pfi.flatten(),  maxiter=50, atol=1e-11)
tt -= time.time()
err = np.max(abs(ui.flatten() - solution.flatten()))
print("loo error :", err)
print("step      :", ct.step)
print("time      :", -tt)
