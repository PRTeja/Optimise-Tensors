
import numpy as np
from numpy import sqrt 
import scipy
import time
from matplotlib import pyplot as plt
from decimal import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys
import os.path
import torch
from  Basics import *

class NewBasisTensors:
	def __init__(self,D,ds):
		self.D = D
		self.ds=ds
		self.index        = []
		self.indexC4v     = []
		self.indexfree    = []
		self.amplitude    = []

class BaseTensors:
	def __init__(self,D,ds):
		self.D = D
		self.ds=ds
		self.index        = []
		self.indexC4v     = []
		self.indexfree    = []
		self.amplitude    = []
	
	def add_indexC4v(self, value):
		self.indexC4v.append(value)
		
	def add_indexfree(self, value):
		self.indexfree.append(value)
		
	def add_amplitude(self, value):
		self.amplitude.append(value)
	
	def add_values(self, index, indx, indxfree, amp):
		self.index.append(index)
		self.indexC4v.append(indx)
		self.indexfree.append(indxfree)
		self.amplitude.append(amp)
		

class OneSite:
	def __init__(self,X, TensorD5, Ry, C4v):
		self.TensorD5 = TensorD5
		self.Ry = Ry#Bond
		#print(X,'*')
		
		if C4v:
			self.A   = ConstructC4vTensor(X,self.TensorD5)
			self.X   = X
			self.Xfree  = Xc4vtoX(X)
			#self.c4v = True
		else:
			self.A   = ConstructTensor(X,self.TensorD5)
			self.X   = X
			self.Xfree  = X#Xc4vtoX(X)
			#self.c4v = False
			
		self.c4v = CheckC4vA(self.A)#False	
		self.E   = DoubleLayer(self.A, self.A)
		self.B   = tsum('aijkl, ax -> xijkl', self.A, self.Ry)#B_singlet(self.A, self.Bond)
		
		
	def Symmetrize(self):
		self.A   = Symmetrize_A1(self.A)
		self.B   = tsum('aijkl, ax -> xijkl', self.A, self.Ry)#B_singlet(self.A, self.Bond)
		self.E   = DoubleLayer(self.A, self.A)
		self.c4v = CheckC4vA(self.A)#True	

	def Update(self, X):
		self.A   = ConstructC4vTensor(X,self.TensorD5)
		self.c4v = CheckC4vA(self.A)#False
		self.X   = X
		self.Xfree   = Xc4vtoX(X)
		self.E   = DoubleLayer(self.A, self.A)
		self.B   = tsum('aijkl, ax -> xijkl', self.A, self.Ry)#B_singlet(self.A, self.Bond)
		
		
	

def string2zeroarray(string, shape):
	globals()[string] = zz(shape)

def returnindex(lines):
	l0 = lines.split('[')
	l0 = l0[1]
	l0 = l0.split(']')
	
	l0 = l0[0]
	l0 = l0.split(',')
	l0 = np.array([int(i) for i in l0])
	
	return tuple(l0)

def returnoccupation(lines):
	
	l0 = lines.split('[')
	l0 = l0[0]
	l0 = l0.split('e_')
	l0 = l0[1]
	return int(l0)

def returnamp(lines):
	l0 = lines.split('=')
	return eval(l0[1])


def freeind(occup, index):
	if occup//100==3:
		if index[1] !=0:
			return 0
		if index[2] !=0:
			return 1
		if index[3] !=0:
			return 2
		if index[4] !=0:
			return 3
	else:
		if index[1] ==0:
			return 0
		if index[2] ==0:
			return 1
		if index[3] ==0:
			return 2
		if index[4] ==0:
			return 3	


	
def SingletoComplex(Xs):
	l = len(Xs)//2
	X = []
	for i in range(l):
		X.append(Xs[2*i]+Xs[2*i+1]*1j)
	return np.array(X)
	
def ComplextoSingle(X):
	l = len(X)
	Xs = []
	for x0 in X:
		Xs.append(x0.real)
		Xs.append(x0.imag)
	return  np.array(Xs)
	
def numtorb0(a):
	if a == 0:
		return 'h'
	if a == 1 or a == 2:
		return 'r'
	if a ==3 or a ==4:
		return 'b'

	

def ReadTensors_c4v(ds,D,N=10,Nb=1):
		
	# ds = size of physical index
	# D  = sie of virtual index
	# N  = Number of Tensors files
	# Nb = Number of Bond operator files
	
	if D==3:
		N = 2
	
	
	
	TensorD5 = BaseTensors(D,ds)
	
	Tensors = zz((N, ds,D,D,D,D))
	form = (ds,D,D,D,D)#).shape 
	
	#Tensor_names = ['T_1_A1_13','T_2_A1_31']# for D = 3
	if D ==5:
		Nc4v = 10
		Ncs  = 38
		
		namec4v = 'D5.dat'
		namecs  = 'D5_Cs.txt'
		
		Tensor_names = ['T_1_A1_310','T_2_A1_301','T_3_A1_103','T_4_A1_130',
		'T_5_A1_112','T_6_A1_112','T_7_A1_112','T_8_A1_121',
		'T_9_A1_121', 'T_10_A1_121']# for D = 5
	
		labels = ['310', '301', '103','130', '112', '112', '112', '121', '121', '121']
	if D ==3:
		N = 2
		NC4v = 2
		NCs  = 7
		
		Tensor_names = ['T_1_A1_31', 'T_2_A1_13']
		labels = ['31', '13']
	
	
	
	TensorsC4v = zz((Nc4v,ds,D,D,D,D))
	TensorsCs = zz((Ncs,ds,D,D,D,D))
	
	
	#f = open('Tensors/D5.dat','r')
	f = open('Tensors/'+namec4v,'r')
	f1 = f.readlines()
	
	for line in f1:
		#exec(line)
		if len(line) >1:
			occup = returnoccupation(line)
			value = returnamp(line)
			freeindex = freeind(occup, returnindex(line))
			
			#n0, freeindex = indexnumreturn(index)
			
			print(ocup)
			#TensorD5.add_values(returnindex(line), n0, freeindex, value)
			#print(TensorD5.index[n0])
			#indexes[i].append(returnidex(line))
		
	sys.exit()
	for i in range(N):
		line = 'Tensors['+str(i)+',:,:,:,:,:] = ' + Tensor_names[i]+'[:,:,:,:,:]'
		exec(line)
		
		
	Bonds = zz((Nb, D,D))
	#sys.exit()

	for i0 in range(Nb):
		string = 'Tensors/BondOp'+str(i0+1)+'.dat'
		if D ==3:
			string = 'Tensors/BondOp1D3.dat'
		
		f = open(string,'r')
		f1 = f.readlines()
		#details = f1[0].split()
		print('Bond Operator -' +str(i0+1) +' is defined. file_len =', len(f1))
		f.close()
		for lines in f1:
			line = lines.split()
			Di, di1, di2,eps = int(line[0]), int(line[1]), int(line[2]), eval(line[3])
			Bonds[i0, di1, di2] = eps
			
	return TensorD5, Bonds
	


def ConstructC4vTensor(XC4v,TensorD5):
	nums = len(TensorD5.amplitude)
	#print(nums)
	
	D = TensorD5.D
	ds= TensorD5.ds
	
	A  = zz((ds,D,D,D,D))
	
	
	for i in range(nums):
		
		indx = TensorD5.index[i]
		j    = TensorD5.indexC4v[i]
		amp  = TensorD5.amplitude[i]
		#print(len(XC4v), j, nums, indx)
		A[indx] = XC4v[j]*amp
		
	return A	

def ConstructTensor(X,TensorD5):
	nums = len(TensorD5.amplitude)
	
	if len(X) != 40:
		print(X)
		sys.exit('Lambda value error!!! - shape not accurate ::'+str(len(X)))
	
	D = TensorD5.D
	ds= TensorD5.ds
	
	A  = zz((ds,D,D,D,D))
	
	for i in range(nums):
		indx = TensorD5.index[i]
		j    = TensorD5.indexC4v[i]
		k    = TensorD5.indexfree[i]
		amp  = TensorD5.amplitude[i]
		A[indx] = X[j*4+k]*amp
		#print(j,j*4+k, X[j*4+k], amp)
	return A
	
def CreateA1_112(N, ds, D):
	A1 = zz((N, ds,D,D,D,D))
	
	f = open(string,'r')
	f1 = f.readlines()
	f.close()
	
	for lines in f1:
			line = lines.split()
			Di, di1, di2,eps = int(line[0]), int(line[1]), int(line[2]), eval(line[3])
			
			
