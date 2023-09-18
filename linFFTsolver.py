import numpy as np
import scipy.sparse.linalg as sp
import itertools
import time
import sys
import pandas as pd
from multiprocessing import Pool, Process,Lock
#from mayavi import mlab #scientific visualization library

################# STANDARD FFT-HOMOGENIZATION by Moulinec & Suquet (1994) ###############
# ----------------------------------- GRID ------------------------------------
ndim   = 3            # number of dimensions
N      = 31           # number of voxels (assumed equal for all directions, needs to be an odd number)
ndof   = ndim**2*N**3 # number of degrees-of-freedom

# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------
print(1)
# tensor operations/products: np.einsum enables index notation, avoiding loops
trans2 = lambda A2 : np.einsum('ijxyz          ->jixyz  '     ,A2)
ddot42 = lambda A4,B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ',A4,B2)
dot22  = lambda A2,B2: np.einsum('ijxyz  ,jkxyz  ->ikxyz  ',A2,B2)
dyad22 = lambda A2,B2: np.einsum('ijxyz  ,klxyz  ->ijklxyz',A2,B2)
print(2)
# identity tensor                                               [single tensor]
i      = np.eye(ndim)
# identity tensors                                            [grid of tensors]
I      = np.einsum('ij,xyz'           ,                  i   ,np.ones([N,N,N]))
I4     = np.einsum('ijkl,xyz->ijklxyz',np.einsum('il,jk',i,i),np.ones([N,N,N]))
I4rt   = np.einsum('ijkl,xyz->ijklxyz',np.einsum('ik,jl',i,i),np.ones([N,N,N]))
I4s    = (I4+I4rt)/2. # symm. 4th order tensor
II     = dyad22(I,I)  # dyadic product of 2nd order unit tensors
print(3)
# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------

# phase indicator: cubical inclusion of volume fraction 
# here: inclusion has cylinder form
phase  = np.zeros([N,N,N])
r = (0.181*N**2)/np.pi #radius of cylinder (20% volume fraction)
for i in range(N):
    for j in range(N):
        for k in range(N):
            if ((i-int(N/2))**2 + (k-int(N/2))**2) <= r:
                phase[i,j,k]=1.

print(4)
## Visualization with Mayavi
# X, Y, Z = np.mgrid[0:1:N*1j, 0:1:N*1j, 0:1:N*1j]
# mlab.points3d(X, Y, Z, phase, color=(0.86, 0.24, 0.22), scale_factor=0.1, mode='cube')
# mlab.outline(color=(0.24, 0.56, 0.71), line_width=2.7)
# mlab.show()

# material parameters + function to convert to grid of scalars
param   = lambda M1,M2: M1*np.ones([N,N,N])*(1.-phase)+M2*np.ones([N,N,N])*phase
lambda1 = 4442
lambda2 = 22742
#lambda1 = 22742
#lambda2 = lambda1
lambdas = param(lambda1, lambda2)  # Lamé constants (material1, material2)  [grid of scalars]
mu1     = 975
mu2     = 31404
#mu1     = 31404
#mu2     = mu1
mu      = param(mu1, mu2)          # shear modulus [grid of scalars]
## stiffness tensor                [grid of scalars]  
C4      = lambdas*II+2.*mu*I4s 
print(5)
# ------------------------------------------------------------------------------

## projection operator                            [grid of tensors]
delta   = lambda i,j: np.cfloat(i==j)              # Dirac delta function
freq    = np.arange(-(N-1)/2.,+(N+1)/2.)          # coordinate axis -> freq. axis
lambda0 = (lambda1 + lambda2)/2                   # Lamé constant for isotropic reference material
mu0     = (mu1 + mu2)/2                           # shear modulus for isotropic reference material
const   = (lambda0 + mu0)/(mu0*(lambda0 + 2*mu0))
#Greens  = np.zeros([ndim,ndim,ndim,ndim,N,N,N])   # Green's function in Fourier space
print(6)
def lame_1(E,v):
    return 0
def lame_2(E,v):
    return 0

def a_Greens(cores):
    Greens  = np.zeros([ndim,ndim,ndim,ndim,N,N,N])   # Green's function in Fourier space
    args=[]
    for k,h,i,j in itertools.product(range(ndim), repeat=4):
        #print(k)
        #print(h)
        #print(i)
        #print(j)
        #print("")
        args.append([k,h,i,j])
        """
        for x,y,z in itertools.product(range(N), repeat=3):
            q = np.array([freq[x], freq[y], freq[z]]) # frequency vector
            if not q.dot(q) == 0: # zero freq. -> mean
                Greens[k,h,i,j,x,y,z] = (1/(4*mu0*q.dot(q))*\
                (delta(k,i)*q[h]*q[j]+delta(h,i)*q[k]*q[j]+\
                delta(k,j)*q[h]*q[i]+delta(h,j)*q[k]*q[i]))-\
                (const*((q[i]*q[j]*q[k]*q[h])/(q.dot(q))**2))
        """
    with Pool(cores) as pool:
        Results=pool.starmap(assign_Greens,args)
    for r in Results:
        #print(r)
        [k,h,i,j]=r[1]
        for x,y,z in itertools.product(range(N), repeat=3):
            Greens[k,h,i,j,x,y,z]=r[0][x,y,z]
    return Greens
   
#a_Greens()
def assign_Greens(k,h,i,j):
    GR=np.zeros([N,N,N])
    print(k)
    print(h)
    print(i)
    print(j)
    print("")
    for x,y,z in itertools.product(range(N), repeat=3):
        q = np.array([freq[x], freq[y], freq[z]]) # frequency vector
        if not q.dot(q) == 0: # zero freq. -> mean
            GR[x,y,z] = (1/(4*mu0*q.dot(q))*\
             (delta(k,i)*q[h]*q[j]+delta(h,i)*q[k]*q[j]+\
              delta(k,j)*q[h]*q[i]+delta(h,j)*q[k]*q[i]))-\
              (const*((q[i]*q[j]*q[k]*q[h])/(q.dot(q))**2))
    return GR, [k,h,i,j]
# (inverse) Fourier transform (for each tensor component in each direction)
fft  = lambda x: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[N,N,N]))
ifft = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[N,N,N]))
print(8)
# inner functions to produce IB matrix, IB = I - F^{-1} *Gamma*F*C --> eps_i+1 = IB*eps_i 
#G           = lambda x: np.real(ifft(ddot42(Greens,fft(x)))).reshape(-1)
G           = lambda x: np.real(ifft(ddot42(Greens,fft(x))))
Stiff_Mat   = lambda x: ddot42(C4,x.reshape(ndim,ndim,N,N,N))
G_Stiff_Mat = lambda x: G(Stiff_Mat(x))
#Id          = lambda x: ddot42(I4,x.reshape(ndim,ndim,N,N,N)).reshape(-1)
Id          = lambda x: ddot42(I4,x.reshape(ndim,ndim,N,N,N))
IB          = lambda x: np.add(Id(x),-1.*G_Stiff_Mat(x))
print(9)
# # ----------------------------- NEWTON ITERATIONS -----------------------------

# initialize stress and strain for each 6 macroscopic strain E  
sig = [np.zeros([ndim,ndim,N,N,N]) for _ in range(6)]
eps = [np.zeros([ndim,ndim,N,N,N]) for _ in range(6)]

# set macroscopic strains, total:6 (for each direction)

E   = [np.zeros([ndim,ndim,N,N,N]) for _ in range(6)]
print(10)
E[0][0][0] = 1.0 # loading in 1,1 direction 
E[1][1][1] = 1.0 # loading in 2,2 direction
E[2][2][2] = 1.0 # loading in 3,3 direction
E[3][0][1] = 1.0 # loading in 1,2 direction
E[3][1][0] = 1.0 # loading in 2,1 direction (due to symmetry)
E[4][1][2] = 1.0 # loading in 2,3 direction
E[4][2][1] = 1.0 # loading in 3,2 direction (due to symmetry)
E[5][0][2] = 1.0 # loading in 1,3 direction
E[5][2][0] = 1.0 # loading in 3,1 direction (due to symmetry)
"""
E=np.zeros([ndim,ndim,N,N,N])
sig=np.zeros([ndim,ndim,N,N,N])
eps = np.zeros([ndim,ndim,N,N,N])

E[0][0]=1.0
E[1][1]=1.0
E[2][2]=1.0
E[0][1]=1.0
E[1][0]=1.0
E[1][2]=1.0
E[2][1]=1.0
E[0][2]=1.0
E[2][0]=1.0

iter=0
"""
iiter = [0 for _ in range(6)]
print(11)
# --------------- for convergence criteria ----------------

freqMat = np.zeros([ndim, 1, N, N, N])      #[grid of scalars]
for j in range(ndim):
    for x in range(N):
        for y in range(N):
            for z in range(N):
                if j==0:
                    freqMat[j,0,x,y,z] = freq[x]
                elif j==1:
                    freqMat[j,0,x,y,z] = freq[y]
                elif j==2:
                    freqMat[j,0,x,y,z] = freq[z] 
print(12)
freqMat_T = trans2(freqMat)
c         = int((N-1)/2)                    # center of frequency grid
# ---------------------------------------------------------
count=0
"""
start=time.time()
for i in range(6):
    sigma = np.zeros([ndim,ndim,N,N,N])
    eps[i] += E[i]
    while True:
        #eps[i]   = IB(eps[i])
        sigma    = Stiff_Mat(eps[i])
        # ---------------- (equilibrium-based) convergence criteria -------------------------
        fft_sig=fft(sigma)
        fou_sig  = fft_sig.reshape(ndim, ndim, N,N,N)
        delta_eps = -1.*np.real(ifft(ddot42(Greens,fft_sig))).reshape(-1)
        nom      = np.sqrt(np.mean(np.power(dot22(freqMat_T, fou_sig),2))) #nominator
        denom    = np.sqrt(np.real(fou_sig[0,0,c,c,c]**2 + fou_sig[1,1,c,c,c]**2 +\
                    fou_sig[2,2,c,c,c]**2 + fou_sig[0,1,c,c,c]**2 +\
                    fou_sig[1,2,c,c,c]**2 + fou_sig[0,2,c,c,c]**2)) # denominator
        # ---------------------------------------------------------------
        #eps[i]=np.add(Id(eps[i]),delta_eps)
        #print(count)
        #print(nom/denom)
        #print("")
        #count=count+1
        if nom/denom <1.e-8 and iiter[i]>0: break
        iiter[i] += 1
end=time.time()
print("Old approach:")
print(str(end-start)+"s")
"""

def solve(i,Greens):
    
    count=0
    sig[i] = np.zeros([ndim,ndim,N,N,N])
    eps[i] += E[i]
    while True:
        #eps[i]   = IB(eps[i])
        sig[i]    = Stiff_Mat(eps[i])
        # ---------------- (equilibrium-based) convergence criteria -------------------------
        fft_sig=fft(sig[i])
        fou_sig  = fft_sig.reshape(ndim, ndim, N,N,N)
        delta_eps = -1.*np.real(ifft(ddot42(Greens,fft_sig)))
        #delta_eps = -1.*np.real(ifft(ddot42(Greens,fft_sig))).reshape(-1)
        nom      = np.sqrt(np.mean(np.power(dot22(freqMat_T, fou_sig),2))) #nominator
        denom    = np.sqrt(np.real(fou_sig[0,0,c,c,c]**2 + fou_sig[1,1,c,c,c]**2 +\
                    fou_sig[2,2,c,c,c]**2 + fou_sig[0,1,c,c,c]**2 +\
                    fou_sig[1,2,c,c,c]**2 + fou_sig[0,2,c,c,c]**2)) # denominator
        # ---------------------------------------------------------------
        eps[i]=np.add(Id(eps[i]),delta_eps)
        #print(count)
        print(nom/denom)
        print("")
        #count=count+1
        if nom/denom <1.e-8 and iiter[i]>0: break
        iiter[i] += 1
    #if eps[i].any():
    #    print(eps[i])
    return eps[i]
    """
    sig=np.zeros([ndim,ndim,N,N,N])
    eps+=E
    while True:
        #eps[i]   = IB(eps[i])
        sig    = Stiff_Mat(eps)
        # ---------------- (equilibrium-based) convergence criteria -------------------------
        fft_sig=fft(sig)
        fou_sig  = fft_sig.reshape(ndim, ndim, N,N,N)
        delta_eps = -1.*np.real(ifft(ddot42(Greens,fft_sig)))
        #delta_eps = -1.*np.real(ifft(ddot42(Greens,fft_sig))).reshape(-1)
        nom      = np.sqrt(np.mean(np.power(dot22(freqMat_T, fou_sig),2))) #nominator
        denom    = np.sqrt(np.real(fou_sig[0,0,c,c,c]**2 + fou_sig[1,1,c,c,c]**2 +\
                    fou_sig[2,2,c,c,c]**2 + fou_sig[0,1,c,c,c]**2 +\
                    fou_sig[1,2,c,c,c]**2 + fou_sig[0,2,c,c,c]**2)) # denominator
        # ---------------------------------------------------------------
        eps=np.add(Id(eps),delta_eps)
        #print(count)
        print(nom/denom)
        print("")
        #count=count+1
        if nom/denom <1.e-8 and iter>0: break
        iter +=1
    return eps
    """
if __name__=='__main__':
    start=time.time()
    Greens_parallel  = a_Greens(6)
    print(f"Greens needs{sys.getsizeof(Greens_parallel)}")
    
    with Pool(6) as pool:
        results=pool.starmap(solve,((i,Greens_parallel)for i in range(6)))
    for i in range(6):
        eps[i]=results[i]
        
    #eps=solve(1,Greens_parallel)
    #if Greens.any()==Greens_parallel.any():
    #    print("I am okay :)")
    #results=pool.map(solve,args)
    end=time.time()
    print("New approach:")
    print(str(end-start)+"s")
    # homogenized stiffness
    homStiffness = np.zeros([6, 6])

    # homogenization operation <f> = 1/N Σ f_i
    for i in range(6):
        sig[i] = Stiff_Mat(eps[i])
        #print(len(eps[i]))
        #print(len(sig[i][0][0]))
        #print(round(np.sum(eps[i][2][2])/(N**3),4))
        homStiffness[0][i] = round((1.0/(N**3))*np.sum(sig[i][0][0]),4)
        homStiffness[1][i] = round((1.0/(N**3))*np.sum(sig[i][1][1]),4)
        homStiffness[2][i] = round((1.0/(N**3))*np.sum(sig[i][2][2]),4)
        homStiffness[3][i] = round((1.0/(N**3))*np.sum(sig[i][0][1]),4)
        homStiffness[4][i] = round((1.0/(N**3))*np.sum(sig[i][1][2]),4)
        homStiffness[5][i] = round((1.0/(N**3))*np.sum(sig[i][2][0]),4)

    print("Homogenized Stiffness: \n", homStiffness)
    X=[]
    Y=[]
    Z=[]
    #print(eps[0][0])
    for i in range(N):
        X.append(i)
        Y.append(i)
        Z.append(i)
    e11=[]
    e12=[]
    e13=[]
    e22=[]
    e23=[]
    e33=[]
    s11=[]
    s12=[]
    s13=[]
    s22=[]
    s23=[]
    s33=[]
    pos=[]
    for x in X:
        for y in Y:
            for z in Z:
                e11.append(eps[0][0][0][x][y][z])
                e12.append(eps[3][0][1][x][y][z])
                e13.append(eps[5][0][2][x][y][z])
                e22.append(eps[1][1][1][x][y][z])
                e23.append(eps[4][1][2][x][y][z])
                e33.append(eps[2][2][2][x][y][z])
                s11.append(sig[0][0][0][x][y][z])
                s12.append(sig[3][0][1][x][y][z])
                s13.append(sig[5][0][2][x][y][z])
                s22.append(sig[1][1][1][x][y][z])
                s23.append(sig[4][1][2][x][y][z])
                s33.append(sig[2][2][2][x][y][z])
                pos.append([x,y,z])
    print(e11)
    Data = pd.DataFrame({"position": pos,
                         "e11": e11,
                         "e12": e12,
                         "e13": e13,
                         "e22": e22,
                         "e23": e23,
                         "e33": e33,
                         "s11": s11,
                         "s12": s12,
                         "s13": s13,
                         "s22": s22,
                         "s23": s23,
                         "s33": s33})
    Data.to_csv("Data.csv",index=False)

