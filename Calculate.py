import numpy as np
import scipy
import time
from matplotlib import pyplot as plt
from decimal import *
from scipy.linalg import sqrtm
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import os.path
import torch

from Basics import *
from Decs import *


# Time evolution gate. U(dt) = exp(-i*dt*H)
def gateH(H,dt, Ry, Rotate = True):
	
	ds2 = H.shape[0]
	ds = int(np.sqrt(ds2))
	
	H1 = tsum('ijkl ->ikjl', H.reshape(ds,ds,ds,ds)).reshape(ds*ds,ds*ds)
	exp1 = scipy.linalg.expm(1j*H1*dt)	
	
	G = tsum('ijkl ->ikjl', exp1.reshape(ds,ds,ds,ds))#.reshape(ds*ds,ds*ds)
	
	if Rotate:
		G = tsum('ijkl, xj, yl -> ixky', G, Ry,Ry).reshape(ds*ds, ds*ds)
	
#	print('Gate(tau =',dt,') ::', G.reshape(ds*ds,ds*ds).real)
	
	return G.reshape(ds*ds,ds*ds)#tsum('ijkl ->ikjl', exp1.reshape(ds,ds,ds,ds)).reshape(ds*ds,ds*ds)	


# Environment routines:
def CTEnvironment(C,T):
	
	C1 = tsum('ij, abj -> iab', C, T)		
	C2 = tsum('abc, iaj-> jibc', C1, T)
	C3 = tsum('ijkl,imn -> lkjmn', C2, C1)
	
	return C3


def Onsite(C,T,E1):
	
	C1 = tsum('ij, abj -> iab', C, T)		
	
	C2 = tsum('abc, iaj-> jibc', C1, T)
	
	Ctemp0 = tsum('ijkl, mnkj-> imnl', C2, E1)
	
	C3 = tsum('ijkl, ijm -> mkl', Ctemp0, C1)
	
	res = tsum('iaj, ix, jax', C3, C, C1)
	
	return res

def FullUEnv(C,T,A, A1):
	ds = A.shape[0]
	Ddk0=A.shape[1]
	Ddk= A1.shape[1]
	D  = A.shape[2]
	D2 = D*D
	
	E  = DoubleLayer(A,A1)#EAcustom(A, A1)#tsum('aijkl, amnop -> imjnkolp', A, np.conjugate(A1)).reshape(Ddk0*Ddk,D2,D2,D2)
	
	C1 = tsum('ij, abj -> iab', C, T)		
	
	C2 = tsum('abc, iaj-> jibc', C1, T)
	
	Ctemp0 = tsum('ijkl, mnkj-> imnl', C2, E)
	
	Ctop   = tsum('ijkl, lkm -> ijm', Ctemp0, C1)
	
	Rho    = tsum('iaj, ibj  -> ab',  Ctop, Ctop)#.reshape(ds*D*ds*D, ds*D*ds*D)	
	
	return Rho.reshape(Ddk0,Ddk,Ddk0,Ddk)

def FullUEnvunit(C,T,E):
	
	#EAcustom(A, A1)#tsum('aijkl, amnop -> imjnkolp', A, np.conjugate(A1)).reshape(Ddk0*Ddk,D2,D2,D2)
	
	C1 = tsum('ij, abj -> iab', C, T)		
	
	C2 = tsum('abc, iaj-> jibc', C1, T)
	
	Ctemp0 = tsum('ijkl, mnkj-> imnl', C2, E)
	
	Ctop   = tsum('ijkl, lkm -> ijm', Ctemp0, C1)
	
	Rho    = tsum('iaj, ibj  -> ab',  Ctop, Ctop)#.reshape(ds*D*ds*D, ds*D*ds*D)	
	
	return Rho#.reshape(Ddk0,Ddk,Ddk0,Ddk)

def FullUEnvds(C,T,A, A1):
	ds = A.shape[0]
	Ddk0=A.shape[1]
	Ddk= A1.shape[1]
	D  = A.shape[2]
	D2 = D*D
	
	E  = tsum('aijkl, bpqrs -> abipjqkrls', A, np.conjugate(A1)).reshape(np.array(A.shape)*np.array(A1.shape))
	#DoubleLayer(A,A1)#EAcustom(A, A1)#tsum('aijkl, amnop -> imjnkolp', A, np.conjugate(A1)).reshape(Ddk0*Ddk,D2,D2,D2)
	
	E  = E.reshape(ds,ds, E.shape[1],E.shape[2],E.shape[3],E.shape[4])
	
	C1 = tsum('ij, abj -> iab', C, T)		
	
	C2 = tsum('abc, iaj-> jibc', C1, T)
	
	Ctemp0 = tsum('ijkl, xymnkj-> xyimnl', C2, E)
	
	Ctop   = tsum('xyijkl, lkm -> xyijm', Ctemp0, C1)
	
	Rho    = tsum('uviaj, xyiaj  -> uxvy',  Ctop, Ctop)#.reshape(ds*D*ds*D, ds*D*ds*D)	
	
	return Rho.reshape(ds,ds,ds,ds)

def Calculate2(Corner, T, E2Au,E2Bu, C1, C2):			

	#C1 = tsum('ij, abj -> iab', Corner, T)		
	
	#C2 = tsum('abc, iaj-> jibc', C1, T)
	
	Ctemp1 = tsum('ijkl, omnkj-> oimnl', C2, E2Au)
	
	Ctemp1 = tsum('oijkl,ijm ->olkm', Ctemp1, C1)
	
	Ctemp2 = tsum('ijkl, omnkj-> oimnl', C2, E2Bu)
	
	Ctemp2 = tsum('oijkl,ijm ->olkm', Ctemp2, C1)

	Result = tsum('abcd, abcd', Ctemp1, Ctemp2)

	return Result
	
def Calculate2norm(Corner, T, E):			

	C1 = tsum('ij, abj -> iab', Corner, T)		
	
	C2 = tsum('abc, iaj-> jibc', C1, T)
	
	Ctemp0 = tsum('ijkl, mnkj-> imnl', C2, E)
	
	Ctemp0 = tsum('ijkl,ijm ->lkm', Ctemp0, C1)
	
	#Ctemp2 = tsum('ijkl, mnkj-> imnl', C2, E)
	
	#Ctemp1 = tsum('ijkl,ijm ->lkm', Ctemp1, C1)
	
	Result = tsum('bcd, bcd', Ctemp0, Ctemp0)

	return Result, C1, C2

	
# Correlation S.S(r)
def Correlations(Corner, T, A, E,H, H0, tol):
	ds = A.shape[0]
	D  = A.shape[1]
	D2 = D*D
	
	Oos = sqrtm(H).reshape(ds,ds,ds*ds)
	hd,ho = Takagi(H)
	htak = ho @ sqrtm(np.diag((hd)))
	h1  = np.reshape(htak, (ds,ds,ds*ds)) 
	
	h1  = sqrtm(H).reshape(ds,ds,ds*ds)
	#Oes = sqrtm(Oe)
	
	A1   = tsum('yxk, xabcd -> ykabcd', h1,A)
	
	E1   = tsum('aijkl, bpqrs -> abipjqkrls', A, np.conjugate(A)).reshape(ds*ds, D2, D2, D2, D2)
	
	L = tsum('ij, aik, kl -> jal', Corner, T, Corner)
	
	L = tsum('iaj, cip, xdbca, djq -> xpbq', L, T,E1,T)
	
	End = L
	
	rvals= []
	Correlations = []
	
	for i in range(30):
		
		Corr = tsum('xiaj, yiaj -> xy', L, End).reshape(ds,ds,ds,ds)
		
		CorrH = tsum('aibj, iajb', Corr, H.reshape(Corr.shape))
		
		SSr = CorrH/tsum('iijj', Corr)
		
		rvals.append(2*i+1)
		if abs(SSr) > tol:
			Correlations.append(SSr)
		else:
			Correlations.append(0.)
			
		#print(np.round(CorrH/tsum('iijj', Corr),12))
		L = tsum('xiaj, cip, dbca, djq -> xpbq', L, T,E,T)
		
		###########################
		"""
		Corr = tsum('xiaj, yiaj -> xy', L, End).reshape(ds,ds,ds,ds)
		
		CorrH = tsum('aibj, iajb', Corr, H0.reshape(Corr.shape))
		
		SSr = CorrH/tsum('iijj', Corr)
		
		rvals.append(2*i+2)
		if abs(SSr) > 1e-13:
			Correlations.append(SSr)
		else:
			Correlations.append(0.)
			
		print(np.round(CorrH/tsum('iijj', Corr),12))
		"""
		
		
		##############################
		L = tsum('xiaj, cip, dbca, djq -> xpbq', L, T,E,T)
			
	return Correlations
		
