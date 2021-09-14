
from itertools import groupby

import numpy as np

import scipy
from scipy.linalg import block_diag, sqrtm, polar, schur

import scipy.sparse.linalg as spla

from decimal import *
from Basics import *

import sys

getcontext().prec = 16

def takagieig(N, tol = 1e-14, rounding =1e-14):
	l, U = np.linalg.eigh(N)
	vals = np.abs(l)  # These are the Takagi eigenvalues
	phases = np.sqrt(np.complex128([1 if i > 0 else -1 for i in l]))
	Uc = U @ np.diag(phases)  # One needs to readjust the phases
	list_vals = [(vals[i], i) for i in range(len(vals))]
	list_vals.sort(reverse=True)
	sorted_l, permutation = zip(*list_vals)
	permutation = np.array(permutation)
	Uc = Uc[:, permutation]
	# And also rearrange the unitary and values so that they are decreasingly ordered
	return np.array(sorted_l), Uc


def group_indices(D, rounding = 1e-15):
	# Make sure the array is arranged in ascending or descending order
	pos = []
	ver = D[0]
	D1  = np.abs(D)
	if D1[0] == D1[1]:
		prevpos = 0
	else:
		pos = [[0,1]]
		prevpos = 1
	for i in range(1,len(D)):
		ver = D1[prevpos]
		if abs(D1[i]) > rounding:
			if not (abs(1 - D1[i]/ver) < 1e-4):
				pos.append([prevpos, i])
				prevpos = i
			
			
		else: 	
			
			pos.append([prevpos, i])
			prevpos = i
			break	
	
	if abs(D1[len(D1)-1]) > rounding:
		pos.append([prevpos, i+1])
		 
	return pos

def blockD(a):
	n = len(a)
	bd = block_diag(a[0])
	for i in range(1, n):
		bd = block_diag(bd, a[i])
	
	return bd

def Msqrt(Block):
	if np.allclose(Block, np.diag(np.diag(Block)) ):#Block[i] == np.diag(np.diag(Block[i])):
			return np.diag(np.sqrt(np.diag(Block)))
	else:
			return sqrtm(Block)


def blockSQ(E, D):
	pos = np.array(group_indices(D))
	n   = len(pos)
	Block = [[] for i in range(n)]
	Blocksqrt = [ Msqrt(E[pos[i,0]:pos[i,1], pos[i,0]:pos[i,1]]) for i in range(n)]#[[] for i in range(n)]
	
		
	#B = blockD(Block)
	Bsq = blockD(Blocksqrt)
	
	l = len(Bsq)
	m = len(E) - len(Bsq)
	B = block_diag(Bsq, np.zeros((m,m)))
	
	
	return pos, B

def Takagi(C):
	
	if np.allclose(C, C.T):
		msg = 'all ok'
	else:
		print('*****************Takagi failed*****************')
		sys.exit('Matrix is not symmetric')
	
	D,X = scipy.linalg.eig(C)
	sort = np.argsort(-np.abs(D))#[:-len(D)-1:-1]
	
	D   = D[sort]
	X   = X[:,sort]
	
	Ein = X.T @ X
	
	# Check if needed:
	#print(np.allclose(np.diag(D) @ Ein, Ein @ np.diag(D)))
	
	# The Z matrix in the report:
	E   = scipy.linalg.inv(Ein)
	
	
	# Square root of the Z matrix. If Z is not diagonal, use the block diag function
	
	if np.allclose(E, np.diag(np.diag(E)), rtol = 1e-10):
		Esq0 = np.diag(np.sqrt(np.diag(E)))
		#print("DIAG") # PRINT IF NEEDED 
	else:
		
		pos, Esq0 = blockSQ(E, D)#sqrtm(E)
	
	#	print("BLOCK DIAG") # PRINT IF NEEDED
		
		# if none of this works, try:
		#Esq0 = sqrtm(E)#np.diag(np.sqrt(np.diag(E)))
		
	O   = X @ Esq0
	
	
	if not np.allclose(C, O @ np.diag(D) @ O.T):
		print('TAKGI FAILED')
		Di = np.diag(D)
		Esq = sqrtm(E)
		
		print('commutative ::', np.allclose(Di @ E, E @ Di), E.shape)
		print('commutative ::', np.allclose(Di @ Esq0, Esq0 @ Di))
		print('commutative ::', np.allclose(Di @ Esq0 @ Esq0,  Di @ E))
		
		return [D,E],X
		
		print('step0 ::', np.allclose(sqrtm(E) @ sqrtm(E), E), np.allclose(Esq0  @ Esq0, E))
		
		print('step1 ::', np.allclose(C,X @ Di @ E @ (X).T))
		print('step2 ::', np.allclose(C,X @ Di @ Esq @ Esq @ (X).T))
		print(np.allclose(C, X @ Esq @ Di @ (X @ Esq).T), np.allclose(O, X @ Esq ))
		
		print('\n','*****************Takagi Failed*************','\n')
		sys.exit('TAKAGI INVERSION FAILED!!!!')
		
		
	#else:
	#	print('Takagi sucess')
	
	#sys.exit()
	return D, O
	
def Takagi2(C):
	
	if np.allclose(C, C.T):
		msg = 'all ok'
	else:
		print('*****************Takagi failed*****************')
		sys.exit('Matrix is not symmetric')
	
	D,X = scipy.linalg.eig(C)
	sort = np.argsort(np.abs(D))#[:-len(D)-1:-1]
	
	D   = D[sort]
	X   = X[:,sort]
	
	Ein = X.T @ X
	
	# Check if needed:
	#print(np.allclose(np.diag(D) @ Ein, Ein @ np.diag(D)))
	
	# The Z matrix in the report:
	E   = scipy.linalg.inv(Ein)
	
	
	# Square root of the Z matrix. If Z is not diagonal, use the block diag function
	
	if np.allclose(E, np.diag(np.diag(E)), rtol = 1e-11):
		Esq0 = np.diag(np.sqrt(np.diag(E)))
		#print("DIAG") # PRINT IF NEEDED 
	else:
		
		pos, Esq0 = blockSQ(E, D)#sqrtm(E)
	
	#	print("BLOCK DIAG") # PRINT IF NEEDED
		
		# if none of this works, try:
		#Esq0 = sqrtm(E)#np.diag(np.sqrt(np.diag(E)))
		
	O   = X @ Esq0
	
	
	if not np.allclose(C, O @ np.diag(D) @ O.T):
		print('TAKGI FAILED')
		Di = np.diag(D)
		
		return Takagi(C)
		Esq = sqrtm(E)
		
		print('commutative ::', np.allclose(Di @ E, E @ Di), E.shape)
		print('commutative ::', np.allclose(Di @ Esq0, Esq0 @ Di))
		print('commutative ::', np.allclose(Di @ Esq0 @ Esq0,  Di @ E))
		
		return [D,E],X
		
		print('step0 ::', np.allclose(sqrtm(E) @ sqrtm(E), E), np.allclose(Esq0  @ Esq0, E))
		
		print('step1 ::', np.allclose(C,X @ Di @ E @ (X).T))
		print('step2 ::', np.allclose(C,X @ Di @ Esq @ Esq @ (X).T))
		print(np.allclose(C, X @ Esq @ Di @ (X @ Esq).T), np.allclose(O, X @ Esq ))
		
		print('\n','*****************Takagi Failed*************','\n')
		sys.exit('TAKAGI INVERSION FAILED!!!!')
		
		
	#else:
	#	print('Takagi sucess')
	
	#sys.exit()
	return D, O

def takagisvd(N, tol=1e-13, rounding=15):
    (n, m) = N.shape
    if n != m:
        raise ValueError("The input matrix must be square")
    if np.linalg.norm(N - np.transpose(N)) >= tol:
        print(np.linalg.norm(N - N.T))
        raise ValueError("The input matrix is not symmetric")

    N = np.real_if_close(N)

    if np.allclose(N, 0):
        return np.zeros(n), np.eye(n)

    if np.isrealobj(N):
        print('REALLLLLLLLL')
        # If the matrix N is real one can be more clever and use its eigendecomposition
        l, U = np.linalg.eigh(N)
        vals = np.abs(l)  # These are the Takagi eigenvalues
        phases = np.sqrt(np.complex128([1 if i > 0 else -1 for i in l]))
        Uc = U @ np.diag(phases)  # One needs to readjust the phases
        list_vals = [(vals[i], i) for i in range(len(vals))]
        list_vals.sort(reverse=True)
        sorted_l, permutation = zip(*list_vals)
        permutation = np.array(permutation)
        Uc = Uc[:, permutation]
        # And also rearrange the unitary and values so that they are decreasingly ordered
        return np.array(sorted_l), Uc

    v, l, ws = np.linalg.svd(N)
    w = np.transpose(np.conjugate(ws))
    rl = np.round(l, rounding)

    # Generate list with degenerancies
    result = []
    for k, g in groupby(rl):
        result.append(list(g))

    # Generate lists containing the columns that correspond to degenerancies
    kk = 0
    for k in result:
        for ind, j in enumerate(k):  # pylint: disable=unused-variable
            k[ind] = kk
            kk = kk + 1

    # Generate the lists with the degenerate column subspaces
    vas = []
    was = []
    for i in result:
        vas.append(v[:, i])
        was.append(w[:, i])

    # Generate the matrices qs of the degenerate subspaces
    qs = []
    for i in range(len(result)):
        qs.append(sqrtm(np.transpose(vas[i]) @ was[i]))

    # Construct the Takagi unitary
    qb = block_diag(*qs)

    U = v @ np.conj(qb)
    return rl, U
    
def Takagipartial(C, m):
		
	if np.allclose(C, C.T):
		msg = 'all ok'
		#print('Takagi valid')
	else:
		print('*****************Takagi failed*****************')
		sys.exit('Matrix is not symmetric')
	
	D, X = spla.eigs(C, k=min(m, C.shape[0]-2))
	#D,X = scipy.linalg.eig(C)
	sort = np.argsort(-np.abs(D))#[:-len(D)-1:-1]
	#print(D, sort, D[sort])
	D   = D[sort]
	X   = X[:,sort]
	Ein = X.T @ X
	
	E   = scipy.linalg.inv(Ein)
	#print('Takagi ::', np.round(E, decimals = 10))
	if np.allclose(E, np.diag(np.diag(E))): #E.all() == np.diag(np.diag(E)).all():
		Esq0 = np.diag(np.sqrt(np.diag(E)))
		#print("DIAG")
	else:
		#blockSQ(E, D)
		
		pos, Esq0 = blockSQ(E, D)#sqrtm(E)
		print("BLOCK DIAG")
		#Esq0 = sqrtm(E)#np.diag(np.sqrt(np.diag(E)))
		
	O   = X @ Esq0#sqrtm(E) 
	
	
	
	# three checks. 
	# 1. Esq is the square-root of E. to avoid singular values. Check D*E == D*sqrt(E)^2
	# 2. E commutes with D
	# 3. Esq commutes with D
	# if any of the checks fail, go for full Takagi
	
	# Check-1
	if not np.allclose(np.diag(D) @ E, np.diag(D) @ Esq0 @ Esq0):
	
		
		print(np.round(D,13))
		print(np.round(Esq0,13))
		print(np.allclose(E, np.diag(np.diag(E))))
		print('\n','*****************Takagi Failed at check-1*************','\n')
		
		return Takagi(C)
		sys.exit('TAKAGI INVERSION FAILED!!!!')
		
	# Check-2
	if not np.allclose(E @ np.diag(D), np.diag(D) @ E):
		print('\n','*****************Takagi Failed at check-2*************','\n')
		
		return Takagi(C)
		sys.exit('TAKAGI INVERSION FAILED!!!!')
	
	# Check-2
	if not np.allclose(Esq0 @ np.diag(D), np.diag(D) @ Esq0):
		print('\n','*****************Takagi Failed at check-3*************','\n')
		
		return Takagi(C)
		sys.exit('TAKAGI INVERSION FAILED!!!!')
	
	#else:
	#	print('Takagi sucess')
	
	#sys.exit()
	return D, O
