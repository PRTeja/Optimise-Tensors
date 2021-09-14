
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
from ReadD5 import readindexD5

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
		self.ds= ds
		if self.D ==3:
			self.count_index = np.array([0, 4])
			self.freq_index  = np.array([4,4])
		if self.D ==5:
			self.count_index = np.array([0, 4, 8, 12, 16, 20, 28, 32])
			self.freq_index  = np.array([4, 4, 4, 4,   4,  8, 4,  8])
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

class UnitTensor:
	def __init__(self,D,ds):
		self.D = D
		self.ds= ds
		if self.D ==3:
			self.Nc4v = 2
			self.Ncs  = 7
			
		if self.D ==5:
			self.Nc4v = 10
			self.Ncs  = 38
		
	
	def add_TensorsC4v(self, TensorsC4v):
		self.TensorsC4v = TensorsC4v
	
	
	def add_TensorsCs(self, TensorsCs):
		self.TensorsCs = TensorsCs
		
		
		

class OneSite:
	def __init__(self,X, Tensors, Ry, C4v):
		self.Tensors = Tensors
		self.Ry = Ry#Bond
		#print(X,'*')
		
		if C4v:
			self.A   = ConstructC4vTensor(X,self.Tensors.TensorsC4v)
			self.XC4v   = X
			#self.c4v = True
		else:
			self.A   = ConstructCsTensor(X,self.Tensors.TensorsC4v)
			self.XCs   = X
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
		
def checkinlist(a,lists):
	for b in lists:
		if a ==b:
			return True
	return False

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
	
def returnoccupationC4v(lines):
	l0 = lines.split('[')
	l0 = l0[0]
	l0 = l0.split('A1_')
	l0 = l0[1]
	return int(l0)
	
def returnoccupationCs(lines):
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
	
	
	
	Tensors = UnitTensor(D,ds)
	
	form = (ds,D,D,D,D)#).shape 
	
	#Tensor_names = ['T_1_A1_13','T_2_A1_31']# for D = 3
	
	if D ==7:
		Nc4v = 30
		Ncs  = 117
		
		namec4v = 'D7.txt'
		namecs  = 'D7_Cs.txt'
		
		
		labels = []
	
	if D ==6:
		Nc4v = 11
		Ncs  = 41
		
		namec4v = 'D6.txt'
		namecs  = 'D6_Cs.txt'
		
		
		labels = ['310', '301', '103','130', '112', '112', '112', '121', '121', '121']
		
	if D ==5:
		Nc4v = 10
		Ncs  = 38
		
		namec4v = 'D5.dat'
		namecs  = 'D5_Cs.txt'
		
	
		labels = ['310', '301', '103','130', '112', '112', '112', '121', '121', '121']
	if D ==3:
		N = 2
		Nc4v = 2
		Ncs  = 7
		
		namec4v = 'D3.dat'
		namecs  = 'D3_Cs.txt'
		
		
		Tensor_names = ['T_1_A1_31', 'T_2_A1_13']
		labels = ['31', '13']
	
	
	
	TensorsC4v = zz((Nc4v,ds,D,D,D,D))
	TensorsCs = zz((Ncs,ds,D,D,D,D))
	
	
	#f = open('Tensors/D5.dat','r')
	f = open('Tensors/'+namecs,'r')
	f1 = f.readlines()
	
	n0 = 0
	for line in f1:
		#exec(line)
		if len(line) >1:
			
			index = returnindex(line)
			occup = returnoccupationCs(line)
			value = returnamp(line)
			#freeindex = freeind(occup, returnindex(line))
			
			#n0, freeindex = indexnumreturn(index)
			TensorsCs[n0, index[0], index[2], index[3], index[4], index[1]] = value
			 
			#TensorD5.add_values(returnindex(line), n0, freeindex, value)
			#print(TensorD5.index[n0])
			#indexes[i].append(returnidex(line))
		else:
			n0 = n0+1

	
	f = open('Tensors/'+namec4v,'r')
	f1 = f.readlines()
	
	n0 = 0
	for line in f1:
		#exec(line)
		if len(line) >1:
			
			index = returnindex(line)
			occup = returnoccupationC4v(line)
			value = returnamp(line)
			#freeindex = freeind(occup, returnindex(line))
			
			#n0, freeindex = indexnumreturn(index)
			TensorsC4v[n0, index[0], index[1], index[2], index[3], index[4]] = value
			 
			#TensorD5.add_values(returnindex(line), n0, freeindex, value)
			#print(TensorD5.index[n0])
			#indexes[i].append(returnidex(line))
		else:
			n0 = n0+1
		
	Bonds = zz((Nb, D,D))
	#sys.exit()

	for i0 in range(0):
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
	
	Tensors.add_TensorsCs(TensorsCs) 
	Tensors.add_TensorsC4v(TensorsC4v) 
	
	return Tensors, Bonds

		
def SaveParameters(X, *args):	
	
	f = open("Parameters",'a+')
	if len(args) !=0:
		for i in args:
			f.write(str(i)+'\t')
#	f.write(' paramaters'+ '\t')
	for i in range(len(X)):
		f.write(str(X[i])+ '\t')
	f.write('\n')
	f.close()

	
		
			
