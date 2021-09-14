import numpy as np
import scipy
import time
from matplotlib import pyplot as plt
from decimal import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from scipy import optimize as opt

import os.path
from Decs import *
import torch

from Basics import *
from ReadTensors import *
from Calculate import *
from Optimise import *


def Energy(C,T,A):

		ds = 2

		Sij, Sxx,Syy, Szz, Sijp, Spp, Smm, Sx, Sy, Sz = Hamiltonians(ds)

		A = tsum('aijkl -> alijk', A)

		E = DoubleLayer(A,A)

		norm, C1, C2 = Calculate2norm(C, T, E)
		
		E1, E2   = HtoE(A,A, Sijp)
		
		SSxyz = Calculate2(C, T, E1,E2,C1, C2)/norm
		
		return SSxyz

def diff_(a,b):

	return np.round(np.linalg.norm(a - b),16)

def Singlestep(C, TE, Tensors, A0, A1):	
	
	Var = [C, TE, A1, Tensors]	

	X   = XfromA(A0, Tensors)	

	Result = Optimise(X, Var)
	
	return Result.x#Afromlambda(Result.x, Tensors)
	

def checkrecursion(dev):
	l = len(dev)
	
	val = dev[0]

	for i in range(1,l-1):

		c = 1 - abs(dev[i]/val)

		diff = abs(dev[i]) - abs(val)

		#print(i,  c, dev[i], val)

		if abs(c) < 1e-11 or abs(diff) < 1e-12:

			print(dev[i], val, diff)

			return True, i
	
	return False, 0 
	
def Olap(X, Var):
	
	[C, TE, A1, Tensors] = Var
	A2 = Afromlambda(X,Tensors)
	
	return Fidvals(C, TE, A2, A1)
	
def Overlap(C1,C2,E1,E2,norm):
	
	Ctemp1 = tsum('ijkl, mnkj-> imnl', C2, E1)
	Ctemp1 = tsum('ijkl,lkm ->ijm', Ctemp1, C1)
	
	Ctemp2 = tsum('ijkl, mnkj-> imnl', C2, E2)
	Ctemp2 = tsum('ijkl,lkm ->ijm', Ctemp2, C1)
	
	return tsum('ijk, kji', Ctemp1, Ctemp2)/norm
	
	

def Fidvals(Corner0, TEdge0, A1u, A1, Rotate = 0, norm = 1.):
		for i in range(Rotate):
			A1 = tsum('aijkl -> ajkli', A1)

		Ry = np.array([[0,-1.], [1.,0.]])
		
		Enuma = DoubleLayer(A1u, A1)
		
		Edec1a = DoubleLayer(A1, A1)
		
		Edec2a = DoubleLayer(A1u, A1u)
		
		C1 = tsum('ij, abj -> iab', Corner0, TEdge0)		
	
		C2 = tsum('abc, iaj-> jibc', C1, TEdge0)
		
	
		Ctemp0   = tsum('ijkl, mnkj-> imnl', C2, Enuma)
		Ctemp0   = tsum('ijkl, lkm -> ijm', Ctemp0, C1)
		
		num      = tsum('ijk, kji', Ctemp0, Ctemp0)/norm
			
		Ctemp0   = tsum('ijkl, mnkj-> imnl', C2, Edec1a)
		Ctemp0   = tsum('ijkl, lkm -> ijm', Ctemp0, C1)
		
		den1     = tsum('ijk, kji', Ctemp0, Ctemp0)/norm
		
		
		Ctemp0   = tsum('ijkl, mnkj-> imnl', C2, Edec2a)
		Ctemp0   = tsum('ijkl, lkm -> ijm', Ctemp0, C1)
		
		den2      = tsum('ijk, kji', Ctemp0, Ctemp0)/norm
		
		
		return 1 - (np.abs(num)**2/(den1*den2)).real

def Fidnew(A1, A1u, C1, C2, norm, Rotate = 0):

		for i in range(Rotate):
			A1 = tsum('aijkl -> ajkli', A1)

		
		Enuma = DoubleLayer(A1u, A1)
		
		Edec1a = DoubleLayer(A1, A1)
		
		Edec2a = DoubleLayer(A1u, A1u)
		
	
		term1 = Overlap(C1,C2,Enuma,Enuma,norm)
		term2 = Overlap(C1,C2,Edec1a,Edec1a,norm)
		term3 = Overlap(C1,C2,Edec2a,Edec2a,norm)
		
		return 1 - (np.abs(term1)**2/(term2*term3)).real
		

def Optimise(X0, Var):
	
	#F = Olap(X0, Var)#Fidelity(X0,Var)
	X0 = X0 + np.random.random(X0.shape)*1e-2#inguess(X0, 1e-1)
	options = {'gtol':10.**(-5)}
	
	Result = opt.minimize(Olap, X0, args = (Var,), method = 'CG')#; Result = Result.x
	return Result
	
def Hop(Corner0,TEdge0, Tensors, A0,H):
	
	H = H
	
	ds = A0.A.shape[0]
	D  = A0.A.shape[1]
	Chi= Corner0.shape[0]
	D2 = D*D
	
	Xin = lambdafromA(A0.A, Tensors.TensorsC4v)#A0.Xfree
	A  = A0.A
	E  = A0.E
	
	hsq= sqrtm(H)
	hd,ho = Takagi(H)
	dk = ds*ds
	
	#dk = 1 #
	
	dk2= dk#ds*ds
	Dd2= D*dk
	D2d2= Dd2*D
	Dds = D*ds
	
	
	
	Gate_ij = tsum('ijkl->ikjl', H.reshape(ds,ds,ds,ds))
	
	sort = np.argsort(np.abs(hd))[:-dk-1:-1]
	sortD= np.arange(D)
	hd   = hd[sort]
	ho = ho[:, sort]
	
	htak = ho @ sqrtm(np.diag((hd)))#sqrtm(H)#ho @ sqrtm(np.diag((hd)))
	h1  = np.reshape(htak, (ds,ds,dk)) 
	
	sorti = np.arange(1)
	Hsimple = (ho[:,sorti] @ (np.diag(np.sqrt(hd[sorti])))).reshape(ds,ds,1)
	#h1  = hsq.reshape(ds,ds,dk)
	#print('ALL OK ::', np.allclose(H, tsum('ija, kla', h1, h1).reshape(H.shape)), hd)
	
	
	h2   = tsum(' kib,ija -> kjba', h1, h1)
	h3   = tsum('abij, bck->acijk',h2, h1)
	h3   = (h3 + h3.transpose(0,1,3,4,2) + h3.transpose(0,1,4,2,3))/3.
	
	h4   = tsum('kjdc, jiba -> kidcba', h2, h2)
	h2   = (h2 + h2.transpose(0,1,3,2))/2.
	h4   = (h4 + h4.transpose(0,1,3,4,5,2) + h4.transpose(0,1,4,5,2,3) + h4.transpose(0,1,5,2,3,4))/4.
	
	
	Rot = np.zeros((D,D)); Rot[0,1] = 1.; Rot[1,0] = 1.; Rot[2,2] = 1.
	
	#C3 = CTEnvironment(Corner0,TEdge0)
	#Env0 = tsum('ainmb, ajklb -> ijklmn', C3, C3)
	
	#LM, RM = renormers(Corner0,TEdge0,E)
	#T2     = Ttemp(TEdge0, LM, RM)
	
	X11    = Xin
	X4     = Xin
	U      = np.eye(D*dk,D)
	
	Anew = ConstructTensor(Xin,Tensors, True)
	
	start_time = time.time()
	print('\n','Time step','\n')
	
	
	Hnew = tsum('ija, kla -> ijkl', h1,h1).reshape(H.shape)
	print(np.linalg.norm(Hnew- H))
	
	print(np.round(tsum('ijkl -> ikjl', H.reshape(ds,ds,ds,ds)).reshape(H.shape),14))
	
	for i in range(1):
	
		#X = XfromA(A, Tensors); X = X/np.max(X)
		norm, C1, C2 = Calculate2norm(Corner0, TEdge0, E)
		
		Asimple = tsum('yxk, xabcd -> ykabcd', Hsimple,A).reshape(ds, D,D,D,D) 
		A1   = tsum('yxk, xabcd -> ykabcd', h1,A).reshape(ds, dk*D,D,D,D)
	
	#	A4    = tsum('yxijkl, xabcd -> yiajbkcld', h4,A).reshape(ds,Dd2, Dd2, Dd2, Dd2)
	#	E1    = DoubleLayer(A1, A)
		
		Ddk = A1.shape[1]
		print(Ddk)
		for i1 in range(D):
			lst = (1,i1,2,0,0)
			if (np.round(A[lst], 13)) > 1e-12:
				print('~',lst, A[lst])
		
		
		for i1 in range(Ddk):
			lst = (1,i1,2,0,0)
			if (np.round(A1[lst], 13)) > 1e-12:
				print(lst, A1[lst])
		for i1 in range(Ddk):
			lst = (0,i1,2,0,0)
			if (np.round(A1[lst], 13)) > 1e-12:
				
				print(lst, A1[lst])
		
		En0 = Energy(Corner0,TEdge0,A)
		
		
		
		Anew0, s0, vh0 = SimpleUpdate(A,H)
		
		print('::', diff_(Anew0, A))
		Anew = np.round(tsum('aijkl -> aklij', Anew0),16)
		
		print('::', diff_(Anew, Anew0))
		Anew, s1, vh1 = SimpleUpdate(Anew,H)
		
		Anew = tsum('aijkl -> ajkli', Anew)
		
		Anew, s2, vh2 = SimpleUpdate(Anew,H)
		
		Anew = tsum('aijkl -> aklij', Anew)
		
		Anew, s3, vh3 = SimpleUpdate(Anew,H)
		
		print(diff_(Anew, A))
	sys.exit()

def SimpleUpdate(A,H):
		ds = A.shape[0]
		D  = A.shape[1]
		Dds= D*ds
		
		dt = 0
		
		Sij, Sxx,Syy, Szz, Sijp, Spp, Smm, Sx, Sy, Sz = Hamiltonians(ds)
		Ry = np.array([[0,-1.], [1.,0.]])
		
		
		H1 = gateH(Sij,dt,Ry, True )
		H0 = gateH(Sij,dt,Ry, False)
		
		Ry = np.array([[0,-1.], [1.,0.]])
		
		Ars = A.reshape(ds*D,D**3)
		
		u,s,vh = scipy.linalg.svd(Ars)	
				
		sort = np.arange(len(s))
		vh = vh[sort,:]
		
		core = (u @ np.diag(s)).reshape(ds,D, Dds)
		
		extra = vh.reshape(Dds,D,D,D)
		
		core0 = core.reshape(ds*D, ds*D)
		
		s0, u0 = EigenDecomposition(core0)
		
		test = tsum('aix, xjkl -> aijkl', core, extra)
		
		test2 = test.reshape(Ars.shape)
		
		change = test2 - Ars
		
		print(diff_(test, A), diff_(test2, Ars))
					
		coreB = tsum('aix, ab -> bix', core, Ry)
		
		testB = tsum('aix, xjkl -> aijkl', coreB, extra)
		
		B = tsum('aijkl, ax -> xijkl', A, Ry)
		
		##################
		
		print(core.shape)
		
		hd,ho = Takagi(H)
		dk = ds*ds
	
		htak = ho @ sqrtm(np.diag((hd)))#sqrtm(H)#ho @ sqrtm(np.diag((hd)))
		
		h1  = np.reshape(htak, (ds,ds,dk)) 
		
		corenew = tsum('aix, baj -> bxij', core, h1).reshape(ds* D*ds, D*dk)
	
		corenew2= tsum('ax, bx -> ab', corenew, corenew)
		
		hd,ho = takagisvd(corenew2)#Takagi(corenew2)
		
		htak = ho @ sqrtm(np.diag((hd)))#sqrtm(H)#ho @ sqrtm(np.diag((hd)))
		
		hd,ho = Takagi(corenew2)
		
		htak2 = ho @ sqrtm(np.diag((hd)))#sqrtm(H)#ho @ sqrtm(np.diag((hd)))
		
		print(np.round(hd,13))
		
		print(np.linalg.norm(htak @ htak.T - corenew2))
		
		print(np.linalg.norm(htak - htak2), htak.shape)
		
		print(np.linalg.norm(htak @ htak.T - htak2 @ htak2.T))
		un = htak @ np.linalg.pinv(corenew, rcond = 1e-13)
		
		sys.exit()
			
		return Anew, s1, vh
		
			

def SimpleUpdate2(A,H):
		ds = A.shape[0]
		D  = A.shape[1]
		Dds= D*ds
		Ars = A.reshape(ds*D,D**3)
		
		u,s,vh = scipy.linalg.svd(Ars)
		
		sort = np.arange(len(s))
		vh = vh[sort,:]
		print(u.shape, s.shape, vh.shape)
		
		core = (u @ np.diag(s)).reshape(ds,D, Dds)
		extra = vh.reshape(Dds,D,D,D)
		
		test = tsum('aix, xjkl -> aijkl', core, extra)
		
		core2 = tsum('aix -> axi', core).reshape(ds*Dds,D)#tsum('abk, bix -> akix', h1, core).reshape(ds,dk*D, Dds)
		
		corenew = tsum('aix, biy -> axby', core, core)#
		
		corenew2= tsum('axby, abcd -> cxdy', corenew, H.reshape(ds,ds,ds,ds)).reshape(ds*Dds, ds*Dds)#core2 @ core2.T
		
		s,vh = takagisvd(corenew2)
		Dcut = D
		sort = np.arange(Dcut)
		ssq  = np.sqrt(s[sort])
		vh1  = vh[:,sort]
		
		rot = np.zeros((Dcut,Dcut))
		rot[0,0] = 1.; rot[1,2] = 1.; rot[2,1] = -1.
	#	ssq = ssq @ rot
		vh1 = vh1 @ rot
		print(np.round(s,13))
		corecut = vh1 @ np.diag(ssq)
		
		
		print(diff_(corecut @ corecut.T, corenew2))
		
		print(np.round(vh1,13).real)
		print(np.round(np.diag(ssq),13).real)
		print(np.round(core,13).real)
		
		print('#############3')
		
		print(np.round(corecut,13).real)
		
		Anew =  tsum('axi, xjkl -> aijkl', corecut.reshape(ds,Dds,Dcut), extra)
		
		print(diff_(Anew, A))
		sys.exit()
		
		
		
		cnsq = cnsq.reshape(ds,Dds,Dcut)
		
		Anew = tsum('axi, xjkl -> aijkl', cnsq, extra)
		
		En1 = Energy(Corner0,TEdge0,Anew)
		
		print(En1, En1 - En0)
		
		print(s/s[0], Anew.shape, cnsq.shape, extra.shape, core.shape)
		
		Arecons = tsum('aix, xjkl -> aijkl', core, extra)
		print(Arecons.shape, Anew.shape)
		print(np.linalg.norm(Anew - A))
		
	
