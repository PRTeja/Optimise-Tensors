
import numpy as np
import scipy
import time
import scipy.optimize as opt
from matplotlib import pyplot as plt
from decimal import *
from scipy.linalg import sqrtm
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import os.path
import torch

from Basics import *
from Decs import *
from ReadTens import *
from Calculate import *


def Bmatrix(A):
	Ry = np.array([[0,-1.], [1.,0.]])
	A  = tsum('aijkl, ax -> xijkl', A, Ry)
	return A

def ReflectA(A):
	return tsum('aijkl -> aklij', A)

def RotateX(X):
	l = len(X)
	Xn = []
	for i in range(l//4):
		block = []
		block.append(X[4*i])
		block.append(X[4*i+1])
		block.append(X[4*i+2])
		block.append(X[4*i+3])
		
		Xn.append(block)
	
	Xn = np.array(Xn)
	Xn2= []
	
	for x0 in Xn:
		#x0 = [x0[3], x0[0], x0[1], x0[2]]
		Xn2.append(x0[3])
		Xn2.append(x0[0])
		Xn2.append(x0[1])
		Xn2.append(x0[2])
		
	return np.array(Xn2)

def XtoXnon0(X,fprime):
	Xres = []
	
	for i in range(len(fprime)):
		if np.abs(fprime[i]) > 1e-14:
			Xres.append(X[i])
			
	return np.array(Xres)
	
def Xnon0toX(Xres,X0, fprime):
	X = X0#np.zeros(len(fprime))
	j=0
	for i in range(len(fprime)):
		if np.abs(fprime[i]) > 1e-14:
			X[i] = Xres[j]
			j = j+1
			
	return np.array(X)
	

def XCs_fromXc4(X0):
	#return X0
	X = []#np.zeros(len(X0))
	
	for x in X0:
		X.append(x)
	
	#print(len(X))
	
	X[6] = 'x'; X[7] = 'x'

	X[14]= 'x'; X[15]= 'x'

	X[22]= 'x'; X[23]= 'x'

	X[30]= 'x'; X[31]= 'x'

	X[38]= 'x'; X[39]= 'x'

	X[48]= 'x'; X[49]= 'x'
	X[50]= 'x'; X[51]= 'x'
	X[52]= 'x'; X[53]= 'x'
	X[54]= 'x'; X[55]= 'x'

	X[62]= 'x'; X[63]= 'x'

	X[72]= 'x'; X[73]= 'x'
	X[74]= 'x'; X[75]= 'x'
	X[76]= 'x'; X[77]= 'x'
	X[78]= 'x'; X[79]= 'x'
	
	Xcs = []
	
	for x in X:
		if x != 'x':
			Xcs.append(x)
	
	return np.array(Xcs)
	

			
def Xfree_fromXcs(X):
	#return X
	X1 = np.zeros(80)
	
	######   0-7 // 0-5
	X1[0] = X[0]	;	X1[1] = X[1]
	X1[2] = X[2]	;	X1[3] = X[3]
	X1[4] = X[4]	;	X1[5] = X[5]
	
	X1[6] = X1[2];  	X1[7] = X1[3]
	######   8-15 // 6-11
	
	X1[8]  = X[6]	;	X1[9] = X[7]
	X1[10] = X[8]	;	X1[11] = X[9]
	X1[12] = X[10]	; 	X1[13] = X[11]
	
	X1[14] = X1[10];  X1[15] = X1[11]
	######   16-23 // 12-17
	
	X1[16] = X[12];	X1[17] = X[13]
	X1[18] = X[14];	X1[19] = X[15]
	X1[20] = X[16];	X1[21] = X[17]
	
	X1[22] = X1[18];  X1[23] = X1[19]
	#######   24-31 // 18-23
	
	X1[24] = X[18];	X1[25] = X[19]
	X1[26] = X[20];	X1[27] = X[21]
	X1[28] = X[22];	X1[29] = X[23]
	
	X1[30] = X1[26];  X1[31] = X1[27]
	######################
	#######   32-39 // 24-29
	
	X1[32] = X[24];	X1[33] = X[25]
	X1[34] = X[26];	X1[35] = X[27]
	X1[36] = X[28];	X1[37] = X[29]
	
	X1[38] = X1[34];	X1[39] = X1[35]
	
	#######  40-55 // 30-37

	X1[40] = X[30];	X1[41] = X[31]
	X1[42] = X[32];	X1[43] = X[33]
	X1[44] = X[34];	X1[45] = X[35]	
	X1[46] = X[36];	X1[47] = X[37]
	
	X1[48] = X1[40];	X1[49] = X1[41]; 
	X1[50] = X1[46];	X1[51] = X1[47];
	X1[52] = X1[44];	X1[53] = X1[45];
	X1[54] = X1[42];	X1[55] = X1[43];
	
	#######   56-63 // 38 - 43
	
	X1[56] = X[38];	X1[33] = X[39]
	X1[58] = X[40];	X1[35] = X[41]
	X1[60] = X[42];	X1[37] = X[43]
	
	X1[62] = X1[58];	X1[63] = X1[59]	

	#######  64-79 // 44-51

	X1[64] = X[44];	X1[65] = X[45]
	X1[66] = X[46];	X1[67] = X[47]
	X1[68] = X[48];	X1[69] = X[49]	
	X1[70] = X[50];	X1[71] = X[51]
	
	X1[72] = X1[64];	X1[73] = X1[65]; 
	X1[74] = X1[70];	X1[75] = X1[71];
	X1[76] = X1[68];	X1[77] = X1[69];
	X1[78] = X1[66];	X1[79] = X1[67];
	
	return X1
		

def A2B2fromlambda(X, Tensors):
		A = Afromlambda(X)
		#rot = (0, 4,3)
		A1 = ReflectA(A)
		
		#Bi  = B_singlet(Ai,D,ds, Bond)#B  = B_singlet(A,D,ds, Bond)
		
		return tsum('aijkl, bmnoj -> abimnokl', A,A1)
		
def FullEnv(C1,C2):
	
	Ctemp0 = tsum('ijkl, imn ->nmjkl', C2, C1)
	
	return tsum('iabcj, idefj -> adefcb', Ctemp0, Ctemp0)
		
def Step(C, T, A0,E, Tensors, Ry, h1, c4v = False, rotate  = 0):
	
#	print(X, '~')
	
		
	for i in range(rotate):
		A0 = tsum('aijkl -> ajkli', A0)#X1 = RotateX(X1)#Xc4vtoX(X0)
		
	
	ds= A0.shape[0]
	D = A0.shape[1]
	D2= D*D
	
	dk = 4
	
	A1   = tsum('yxk, xabcd -> ykabcd', h1,A0).reshape(ds, dk*D,D,D,D)
	
	B0   = Bmatrix(A0)
	B1   = tsum('yxk, xabcd -> ykabcd', h1,B0).reshape(ds, dk*D,D,D,D)
	
	#E = DoubleLayer(A,A)
	
	#norm, C1, C2 = Calculate2norm(Corner, T, E)
	
	C1 = tsum('ij, abj -> iab', C, T)		
	
	C2 = tsum('abc, iaj-> jibc', C1, T)
	
	Ctemp0 = tsum('ijkl, mnkj-> imnl', C2, E)
	
	Ctemp0 = tsum('ijkl,lkm ->ijm', Ctemp0, C1)
	
	norm = tsum('ijk, ijk', Ctemp0, Ctemp0)
	
	#CTMEnv = FullEnv(C1,C2)/norm
	
	
	if c4v:
		initials = lambdafromA(A0, Tensors.TensorsC4v)#Xinit#
		initials = initials + np.random.random(initials.shape)*0.
		
		options ={'gtol': 1.*1e-9}
		method = 'CG'
	else:
		initials = lambdafromA(A0, Tensors.TensorsCs)
		options ={'gtol': 1.*1e-6}
		method = 'CG'
	
	initials = initials + np.random.random(initials.shape)*0.
	X_initial = ComplextoSingle(initials)
	
	#Xfree_fromXcs(Xs0)
	
	E1 = DoubleLayer(A1, A1)
	E2 = DoubleLayer(B1, B1)#DoubleLayer(tsum('aijkl -> aklji',A1), tsum('aijkl -> aklji',A1))
	
	
	Ctemp1 = tsum('ijkl, mnkj-> imnl', C2, E1)
	Ctemp1 = tsum('ijkl,lkm ->ijm', Ctemp1, C1)
	
	Ctemp2 = tsum('ijkl, mnkj-> imnl', C2, E2)
	Ctemp2 = tsum('ijkl,lkm ->ijm', Ctemp2, C1)
	
	term2  = tsum('ijk, kji', Ctemp1, Ctemp2)/norm
	
	#print(term2)
	#sys.exit()
	#print(initials)
	
	#print(FidelityValue(initials, [A0,A1, B1,C1,C2, Ry, TensorD5,norm,False, True]))
	
	Var_initial = [A0,A1, B1,C1,C2, Ry, Tensors,norm,term2, c4v, True]
	
	initval = FidCal(A0,A1,B1, C1,C2, norm,term2)
	
	fprime = scipy.optimize.approx_fprime(X_initial, FidelityValue_initial, [1e-2,]*len(X_initial),[A0,A1, B1,C1,C2, Ry, Tensors,norm,term2, c4v, True])
	
	Var = [A0,A1, B1,C1,C2, Ry, Tensors,norm,term2, c4v, True, X_initial, fprime]
	
	print(np.round(SingletoComplex(fprime),14))
	
	print(np.nonzero(SingletoComplex(fprime)))
	Xnon0 = XtoXnon0(X_initial,fprime)
	print('initial guess ::', initials)
	print('initial overlap :: ', initval)
	print('pre-optimised(guess) ::', FidelityValue_initial(X_initial, Var_initial), len(X_initial))
	print(FidelityValue(Xnon0, Var))
	f = open("CG_it",'w+')
	f.truncate(0)
	f.close()
	
	X = Xnon0
	
#	Result = opt.minimize(FidelityValue_initial, X_initial, args= Var_initial, method ='CG', options = options)
	
	Result = opt.minimize(FidelityValue, X, args= Var, method =method, options = options)
	
	print('final(optimised)  overlap :: ',Result.fun)
	
	X = Xnon0toX(Result.x, X_initial, fprime)
	
	X1 = (SingletoComplex(X))
	
	Xz = np.zeros(len(X1))
	
	if c4v:
		Xz[0] = 1.
		Xz[3] = X1[3]/X1[0]#2.50155220e-01-8.74326108e-04j
	
		Anew	= ConstructTensor(Xz,Tensors, True)
		print('with zeros ::', FidCal(Anew,A1,B1, C1,C2,norm,term2).real)
	
	print('Result ::', X1)
	SaveParameters(X1)
	
	sys.exit()
	if c4v:
		return   ConstructTensor(X1,Tensors, True), Result.fun, initval.real#(SingletoComplex(Result.x))
	else:
		
	
		return   ConstructTensor(X1,Tensors, False), Result.fun, initval.real #SingletoComplex((Result.x))
	


def FidelityValue_initial(Xs0, Variable):#A2B2, CTMEnv, D2):
		
		[A,A1,B1, C1,C2, Ry, Tensors,norm,term2, c4v, Cs]    = Variable
		
		if not c4v:
			#Tensors0 = Tensors.TensorsCv
			X	= SingletoComplex(Xs0)
			A2	= ConstructTensor(X,Tensors, False)
			
		else:
			X	= SingletoComplex(Xs0)
			A2	= ConstructTensor(X,Tensors, True)
			
			#Xs = Xs0
		#	Xs = Xc4vtoX(Xs0)
		#A       = Afromlambda(X, Tensors)
		
		return FidCal(A2,A1,B1, C1,C2,norm,term2).real

def FidelityValue(Xnon0, Variable):#A2B2, CTMEnv, D2):
		
		[A,A1,B1, C1,C2, Ry, Tensors,norm,term2, c4v, Cs, X, fprime]    = Variable
		
		Xs0 = Xnon0toX(Xnon0, X,fprime)

		return FidelityValue0(Xs0, [A,A1,B1, C1,C2, Ry, Tensors,norm,term2, c4v, Cs])

def FidelityValue0(Xs0, Variable):#A2B2, CTMEnv, D2):
		
		[A,A1,B1, C1,C2, Ry, Tensors,norm,term2, c4v, Cs]    = Variable
		
		if not c4v:
			#Tensors0 = Tensors.TensorsCv
			X	= SingletoComplex(Xs0)
			A2	= ConstructTensor(X,Tensors, False)
			
		else:
			X	= SingletoComplex(Xs0)
			A2	= ConstructTensor(X,Tensors, True)
			
			#Xs = Xs0
		#	Xs = Xc4vtoX(Xs0)
		#A       = Afromlambda(X, Tensors)
		
		return FidCal(A2,A1,B1, C1,C2,norm,term2).real	
			
def FidCal(A2,A1,B1, C1,C2,norm,term2):
			
			
		#A	= Afromlambda(Xs, Tensors)
		
		#B2 = ReflectA(B2)
		#B1 = ReflectA()
		
		B2	= Bmatrix(A2)
		
		E1 = DoubleLayer(A2, A1)
		E2 = DoubleLayer(B2, B1)
		
		
		#print(C2.shape, E1.shape, E2.shape)
		
		Ctemp1 = tsum('ijkl, mnkj-> imnl', C2, E1)
		Ctemp1 = tsum('ijkl,lkm ->ijm', Ctemp1, C1)
		Ctemp2 = tsum('ijkl, mjkn-> imnl', C2, E2)
		Ctemp2 = tsum('ijkl,lkm ->ijm', Ctemp2, C1)
		
		
		
		#Ctemp1 = tsum('ijkl, mnkj-> imnl', C2, E1)
		#Ctemp2 = Ctemp1#tsum('ijkl, mnkj-> imnl', C2, E2)
		
		term1  = tsum('ijk, ijk', Ctemp1, Ctemp2)/norm
		
		
		#term2 = 1.#norm/norm
		
		E1 = DoubleLayer(A2, A2)
		#E2 = DoubleLayer(B2, B2)#(tsum('aijkl -> aklji',A), tsum('aijkl -> aklji',A))
		
		Ctemp1 = tsum('ijkl, mnkj-> imnl', C2, E1)
		Ctemp1 = tsum('ijkl,lkm ->ijm', Ctemp1, C1)
		
		#Ctemp2 = tsum('ijkl, mjkn-> imnl', C2, E2)
		#Ctemp2 = tsum('ijkl,lkm ->ijm', Ctemp2, C1)
		
		term3  = tsum('ijk, ijk', Ctemp1, Ctemp1)/norm
		
		Fid   = 1 - term1*np.conjugate(term1)/(term2*term3)#np.abs(term1)/np.sqrt(term3.real)#term1*np.conjugate(term1)/(term2*term3)
		#np.abs(term2 + term3 - term1 - np.conjugate(term1))#
			
		#print(term1, term2, term3)
		f = open("CG_it",'a+')
		f.write(str(Fid)+'\t'+str( term1/norm)+'\t'+str( term3/norm)+'\n')
		f.close()
		
		#print(Fid.real, X)
		#print(Fid, term1 + np.conjugate(term1)- (term2+ term3))
		
		return Fid.real#1. - Fid
	
