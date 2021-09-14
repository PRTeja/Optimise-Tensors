
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

def zz(*args,**kwargs):
     kwargs.update(dtype=np.complex128)
     return np.zeros(*args,**kwargs) 

def tsum(string, *args):
	args = tuple([torch.tensor(np.array(a, dtype = np.complex128)) for a in args])	
	return torch.einsum(string, *args).numpy() 

def Errorval(A, Ax):
	w12 = np.abs( tsum( 'aijkl, aijkl',A, np.conjugate(Ax)))**2
	
	w11 = ( tsum( 'aijkl, aijkl',Ax, np.conjugate(Ax)))
	w22 = ( tsum( 'aijkl, aijkl',A, np.conjugate(A)))
	
	return 1 - (w12/(w11*w22)).real

def EigenDecomposition(C, C2 = None):
	s,u = scipy.linalg.eig(C, C2)
	sort = np.argsort(np.abs(s))[:-len(s)-1:-1]
	return s[sort], u[:,sort]




def best_fit_slope(xs,ys):
	xs = np.array(xs)
	ys = np.array(ys)
	m = ( ((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /(np.mean(xs)**2 - np.mean(xs**2)))
	return m
    
def Symmetrize_A1(A):
	A= 0.5*(A + tsum('aijkl -> ailkj',A)) #A.permute(0,1,4,3,2))   # left-right reflection
	A= 0.5*(A + tsum('aijkl -> akjil',A)) #A.permute(0,3,2,1,4))   # up-down reflection
	A= 0.5*(A + tsum('aijkl -> alijk',A)) #A.permute(0,4,1,2,3))   # pi/2 anti-clockwise
	A= 0.5*(A + tsum('aijkl -> ajkli',A)) #A.permute(0,2,3,4,1))
	
	return A

def ReflectA(A):
	return tsum('aijkl -> aklij', A)
	
def ReflectE(E):
	return tsum('ijkl -> klij', E)

def lambdafromA(A, Tensors0):
	n = Tensors0.shape[0]
	X = []
	for i in range(n):
		Ten = Tensors0[i,:,:,:,:,:]
		val = tsum('aijkl, aijkl', Ten, A)/tsum('aijkl, aijkl', Ten, Ten)
		
		X.append(val)
	
	return np.array(X)

def Xc4vtoX(Xc4v):
	X = []
	# Valid only for D = 5
	
	Xc4v = Xc4vtoXfew(Xc4v)
	for i in range(10):
		a = Xc4v[i]
		X.append(a)
		X.append(a)
		X.append(a)
		X.append(a)
			
		
	return np.array(X)	

def Xc4vtoXfew(Xc4v):
	Xfew = []
	
	return np.array([Xc4v[0],Xc4v[1],Xc4v[2],Xc4v[3], Xc4v[4],Xc4v[5],Xc4v[5], Xc4v[6],Xc4v[7],Xc4v[7]])
	
	
	
def XfewtoXc4v(Xfew):
	Xc4v = []
	
	for i in range(4):
		Xc4v.append(Xfew[i])
	
	value = 0
	for i in range(3):
		value = value + Xfew[4+i]
	
	Xc4v.append(value/3.)
	
	value = 0
	for i in range(3):
		value = value + Xfew[7+i]
	
	Xc4v.append(value/3.)
	
	
	return np.array(Xc4v)
		
	
def XtoXc4v(X):
	l = len(X)
	#l  = 
	Xc40 = []
	for i in range(10):
		Xc40.append((X[4*i] + X[4*i+1] +  X[4*i+2] +  X[4*i+3] )/4.)
		
	xc4v = zz(8)
	
	for i in range(5):
		xc4v[i] = Xc40[i]
	
	xc4v[5] = (Xc40[5] + Xc40[6])/2.
	
	xc4v[6] = Xc40[7]
	
	xc4v[7] = (Xc40[8] + Xc40[9])/2.
	
	
	"""	
	value = 0.
	for i in range(12):
		value =  value + X[16+i]
		
	Xc4v.append(value/12.)
	
	value = 0.
	
	for i in range(12):
		value =  value + X[28+i]
		
	Xc4v.append(value/12.)
	"""
		
	return np.array(xc4v)
	
def CheckC4vA(A): # Checks for C4v symmetry of a 4-legged tensor 
	# Check Rotational symmetries
	check1 =  A.all() == tsum('aijkl -> ajkli',A).all()
	check2 =  A.all() == tsum('aijkl -> aklij',A).all()
	check3 =  A.all() == tsum('aijkl -> alijk',A).all()
	# Check reflection symmetries
	check4 =  A.all() == tsum('aijkl -> akjil',A).all()
	check5 =  A.all() == tsum('aijkl -> ailkj',A).all()
	if not (check1 and check2 and check3 and check4 and check5):
		print('Symmetries(3-rotational, 2 reflectional) :: ',check1,check2, check3, check4, check5)
		
	return (check1 and check2 and check3 and check4 and check5)

def CheckC4vE(A): # Checks for C4v symmetry of a 4-legged tensor 
	# Check Rotational symmetries
	check1 =  A.all() == tsum('ijkl -> jkli',A).all()
	check2 =  A.all() == tsum('ijkl -> klij',A).all()
	check3 =  A.all() == tsum('ijkl -> lijk',A).all()
	# Check reflection symmetries
	check4 =  A.all() == tsum('ijkl -> kjil',A).all()
	check5 =  A.all() == tsum('ijkl -> ilkj',A).all()
	
	if not (check1 and check2 and check3 and check4 and check5):
		print('Symmetries(3-rotational, 2 reflectional) :: ',check1,check2, check3, check4, check5)
		
	return (check1 and check2 and check3 and check4 and check5)

def DoubleLayer(A,B):
	return tsum('aijkl, apqrs -> ipjqkrls', A, np.conjugate(B)).reshape(np.array(A.shape[1:5])*np.array(B.shape[1:5]))
	
	
class Onesite(object):
	
	# Defining basic attributes of the one-site tensor
	
	def __init__(
	    self,
	    A):
	    
	    self.A  = A
	    self.ds = A.shape[0]
	    
	    self.D     = A.shape[1]
	    self.D2    = self.D**2
	    self.shape = A.shape
	    Self.Ry    = np.array([[0,-1.], [1.,0.]])
	    
	    
	    if CheckC4vA(A):
	    	self.c4v = True
	    else:
	    	self.c4v = False
	    	
	    self.E = DoubleLayer(A,A)
	    self.B   = tsum('aijkl, ax -> xijkl', self.A, self.Ry)
	    #tsum('aijkl, apqrs -> ipjqkrls', A, np.conjugate(A)).reshape(D2,D2,D2,D2)


def HtoE(A,B,H):
	
	# Takes in 2 tensors, applies a 2-site gate and returns two tensors with an extended bond dimension.
	
	#     -a-  -b-        -a-    -b-                 
	#	\  /		\     /                   -a'=     =b'-
	#	 H	->	 h= =h         ->         |       |
	#	/ \		/     \		  |       |
	#					        -a'=     =b'-
	#    -a-    -b-      -a-     -b-    	      
	#
	#
	
	D = A.shape[1]
	D2= D*D
	ds= A.shape[0]
	
	# H(1234) = H( i',i; j',j) 
	sqr = sqrtm(H)
	
	u = np.reshape(sqr,(ds,ds,ds*ds))
	vt = np.reshape(sqr,(ds*ds,ds,ds))
		
	Au = tsum('ijk,jlmno ->kilmno', u, A)
	Bu = tsum('ijk,klmno ->ijlmno', vt,B)
	
	AuA = tsum('oiabcd, ijklm -> oajbkcldm',Au, np.conjugate(A))
	BuB = tsum('oiabcd, ijklm -> oajbkcldm',Bu, np.conjugate(B))
	
	
	AuA = np.reshape(AuA, (np.array(A.shape)*np.array(A.shape)) )
	BuB = np.reshape(BuB, (np.array(B.shape)*np.array(B.shape)) )	

	return AuA, BuB	
	
def B_singlet(A, Bond): # Insert a Bond Operator on each of the four virtual indices
	
	B = tsum('jklmn, ki -> jilmn', A, Bond)
	B = tsum('jklmn, li -> jkimn', B, Bond)
	B = tsum('jklmn, mi -> jklin', B, Bond)
	B = tsum('jklmn, ni -> jklmi', B, Bond)
	
	return B

def B_singlet1(A, Bond): # Insert a Bond Operator on each of the four virtual indices
	
	B = tsum('jklmn, ki -> jilmn', A, Bond)
	B = tsum('jklmn, li -> jkimn', B, Bond)
	B = tsum('jklmn, mi -> jklin', B, Bond.T)
	B = tsum('jklmn, ni -> jklmi', B, Bond.T)
	
	return B
	
def Hamiltonians(ds):
	Sp = zz((ds,ds))
	Sm = zz((ds,ds))
	Sz = zz((ds,ds))
	#dt = 0.001
	
	if ds == 5:
		for i in range(ds):
			Sz[i][i] = i - 2.
		Sm[0][1] = np.sqrt(4.)
		Sm[1][2] = np.sqrt(6.)
		Sm[2][3] = np.sqrt(6.)
		Sm[3][4] = np.sqrt(4.)
	
		Sp[1][0] = np.sqrt(4.)
		Sp[2][1] = np.sqrt(6.)
		Sp[3][2] = np.sqrt(6.)
		Sp[4][3] = np.sqrt(4.)	
	if ds ==2:
		Sz[0,0] = -0.5
		Sz[1,1] = 0.5
		Sm[0,1] = 1.
		Sp[1,0] = 1.
	S0  = np.eye(Sz.shape[0])
	Sx  = (Sp + Sm)/2.
	Sy  = (Sp - Sm)/2. 
	
	#Sz1 = 
	
	Spm = tsum('ij,kl->ijkl', Sp,Sm)
	Spp = tsum('ij,kl->ijkl', Sp,Sp)
	Smp = tsum('ij,kl->ijkl', Sm,Sp)
	Smm = tsum('ij,kl->ijkl', Sm,Sm)
	Szz = tsum('ij,kl->ijkl', Sz,Sz)
	Sxx = tsum('ij,kl->ijkl', Sx,Sx).reshape(ds*ds,ds*ds)
	Syy = tsum('ij,kl->ijkl', Sy,Sy).reshape(ds*ds,ds*ds)
	
	
	Sz1 = tsum('ij,kl->ijkl', Sz,S0)
	Sz2 = tsum('ij,kl->ijkl', S0,Sz)
	
	#Sz  = Sz1+Sz2
	
	Sij = 0.5*(Spm + Smp) +Szz
	Sijp =  (-0.5*(Spp + Smm) -Szz).reshape((ds*ds, ds*ds ))#(0.5*(Spp + Smm) -Szz).reshape(ds*ds,ds*ds)
	
	Sij = np.reshape(Sij, (ds*ds, ds, ds))
	Sij = np.reshape(Sij, (ds*ds, ds*ds ))
	
	Szz = Szz.reshape(ds*ds,ds,ds)
	Szz = Szz.reshape(ds*ds,ds*ds)
		
	H = Sij#1./14.*(Sij + S2*7./10 + S3*7./45. + S4/90.)
			
	return Sij, Sxx,Syy, Szz, Sijp, Spp.reshape(Sijp.shape), Smm.reshape(Sijp.shape), Sx, Sy, Sz
	
def RemoveGauge(A):
	
	ds = A.shape[0]
	D  = A.shape[1]
	
	#B = tsum('jklmn, ki -> jilmn', A, Bond)
	#B = tsum('jklmn, li -> jkimn', B, Bond)
	
	
	U  =[[] for i in range(ds)]
	
	for i in range(1):
		#print(i)
		a = A[i,:,:,:,:]
		
		rho = tsum('ijik -> jk', a)
		s,u = Takagi(rho)#EigenDecomposition(rho)#
		print(np.round(rho, 14))
		#print(np.round(s,14))
		#u[:,2] = 0.*u[:,2]
		#print(np.round(rho,14))
		#print(np.round(u,14))
		
		#print(np.allclose(u @ np.diag(s) @ u.T, rho))
	print(rho[0,1])
	print(rho[1,2])
	lm2 = rho[1,2]
	lm1 = rho[0,1]
	return s[0], rho, [lm1, lm2]	
	
def lmdfromX(X):
	X   = np.array(X)
	l =  len(X)
	X1 = zz((l//2))
	for i in range(l//2):
		X1[i] = X[2*i] + X[2*i + 1]*1j
		
	return X1
	
def Afromlambda(X, Tensors):
	
	lmd = lmdfromX(X)
	Ai    = tsum('i,ijklmn -> jklmn', lmd, Tensors)		
	return Ai
	
def AfromlambdaC4v(X0, Tensors):
	
	X  = [X0[0], X0[1]]*4 + [X0[2], X0[3]]*4
	
	return Afromlambda(X, Tensors)


def X4fromA(A, Tensors):
	N = Tensors.shape[0]
	X = np.array([0.,]*2*N)
	for i in range(N):
		j = 2*i
		unit = Tensors[i,:,:,:,:,:]
		dot1 = tsum('aijkl, aijkl', A, unit)
		dot2 = tsum('aijkl, aijkl', unit, unit)
		amp  = (dot1/dot2)
		X[j] = amp.real
		X[j+1] = amp.imag
	
	X1 = np.array([0. for i in range(4)])
	
	X1[0] = X[0] + X[2] + X[4] + X[6]
	X1[1] = X[1] + X[3] + X[5] + X[7]
	X1[2] = X[8] + X[10] + X[12] + X[14]
	X1[3] = X[9] + X[11] + X[13] + X[15]
	
	return X1/np.max(abs(X1))
	
def CompleteTensor(A, Tensors):
	X = XfromA(A, Tensors)
	A1= Afromlambda(X, Tensors)
	
	return np.allclose(A, A1)
	

def XfromA(A, Tensors, c4v=True):
	
	if c4v:
		Tensors0 = Tensors.TensorsC4v
	else:
		Tensors0 = Tensors.TensorsCs
		
	
	N = Tensors0.shape[0]
	X = np.array([0.,]*1*N)
	for i in range(N):
		j = 2*i
		unit = Tensors0[i,:,:,:,:,:]
		dot1 = tsum('aijkl, aijkl', A, unit)
		dot2 = tsum('aijkl, aijkl', unit, unit)
		amp  = (dot1/dot2)
		X[j] = amp#.real
	#	X[j+1] = amp.imag
	
	return X/np.max(abs(X))
	
def ConstructTensor(X,Tensors, C4v = True):
	if C4v:
		return tsum('x, xaijkl', X, Tensors.TensorsC4v)
	else:
		return tsum('x, xaijkl', X, Tensors.TensorsCs)

def ConstructC4vTensor(XC4v,Tensors0):
	return tsum('x, xaijkl', XC4v, Tensors0)
		

def ConstructCsTensor(XCs,Tensors):
	return tsum('x, xaijkl', XCs, Tensors0)
	
