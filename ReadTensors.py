
import numpy as np
from numpy import sqrt 
import scipy
import time
from matplotlib import pyplot as plt
from decimal import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import os.path
import torch
from  Basics import *
from ReadTens import *


def ReadTensors(ds,D,N=8, Nb=1):
	
	# ds = size of physical index
	# D  = sie of virtual index
	# N  = Number of Tensors files
	# Nb = Number of Bond operator files
	
	Tensors = zz((40, ds,D,D,D,D))
	for i0 in range(4):
		i = i0 +1
		string = 'Tensors/Tensor5'+str(i)+'.dat'
		f = open(string,'r')
		f1 = f.readlines()
		#details = f1[0].split()
		print('Tensor5' +str(i) +' is defined. file_len =', len(f1))
		f.close()
		for lines in f1:
			line = lines.split()
			#print(line)
			Di, di1, di2, di3, di4, eps = int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]), eval(line[5])
			#Tensors[i0, Di, di1, di2, di3, di4] = eps 
			if abs(eps)>10.**(-14):
				#eps = 1.
				if i0 ==0 or i0==1:
					if di1 !=0:			
						Tensors[4*i0+0, Di, di1, di2, di3, di4] = eps 
					if di2 !=0:			
						Tensors[4*i0+1, Di, di1, di2, di3, di4] = eps 
					if di3 !=0:			
						Tensors[4*i0+2, Di, di1, di2, di3, di4] = eps 
					if di4 !=0:			
						Tensors[4*i0+3, Di, di1, di2, di3, di4] = eps 
				if i0 >1:
					if di1 ==0:			
						Tensors[4*i0+0, Di, di1, di2, di3, di4] = eps 
						
					if di2 ==0:			
						Tensors[4*i0+1, Di, di1, di2, di3, di4] = eps 

					if di3 ==0:			
						Tensors[4*i0+2, Di, di1, di2, di3, di4] = eps 
					if di4 ==0:			
						Tensors[4*i0+3, Di, di1, di2, di3, di4] = eps 
			#print(Di, di1, di2, di3, di4, eps)
			
	
	if 1:
		string = 'Tensors/Tensor55.dat'
		f = open(string,'r')
		f1 = f.readlines()
		f.close()
		
		i0 = 4
		for lines in f1:
			line = lines.split()
	
			#print(line)		
			if len(line) ==0:
				i0 = i0+1
				print(i0,'$')
			else:
				Di, di1, di2, di3, di4, eps = int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]), eval(line[5])
				if abs(eps)>10.**(-14):
				#eps = 1.
					if di1 ==0:
						#print(line, i0)			
						Tensors[4*i0+0, Di, di1, di2, di3, di4] = eps 
						
					if di2 ==0:			
						Tensors[4*i0+1, Di, di1, di2, di3, di4] = eps 

					if di3 ==0:			
						Tensors[4*i0+2, Di, di1, di2, di3, di4] = eps 
					if di4 ==0:			
						Tensors[4*i0+3, Di, di1, di2, di3, di4] = eps 
	
	if 1:
		string = 'Tensors/Tensor55.dat'
		f = open(string,'r')
		f1 = f.readlines()
		f.close()
		
		i0 = 7
		for lines in f1:
			line = lines.split()
	
			#print(line)		
			if len(line) ==0:
				i0 = i0+1
				print(i0,'$')
			else:
				Di, di1, di2, di3, di4, eps = int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]), eval(line[5])
				if abs(eps)>10.**(-14):
				#eps = 1.
					if di1 ==0:
						#print(line, i0)			
						Tensors[4*i0+0, Di, di1, di2, di3, di4] = eps 
						
					if di2 ==0:			
						Tensors[4*i0+1, Di, di1, di2, di3, di4] = eps 

					if di3 ==0:			
						Tensors[4*i0+2, Di, di1, di2, di3, di4] = eps 
					if di4 ==0:			
						Tensors[4*i0+3, Di, di1, di2, di3, di4] = eps 
				
	########################
	
	Bonds = zz((Nb, D,D))


	for i0 in range(Nb):
		string = 'Tensors/BondOp'+str(i0+1)+'.dat'
		f = open(string,'r')
		f1 = f.readlines()
		#details = f1[0].split()
		print('Bond Operator -' +str(i0+1) +' is defined. file_len =', len(f1))
		f.close()
		for lines in f1:
			line = lines.split()
			Di, di1, di2,eps = int(line[0]), int(line[1]), int(line[2]), eval(line[3])
			Bonds[i0, di1, di2] = eps


	Rot = np.eye(D)#np.array([[0,1,0], [1,0,0], [0,0,1]])
	print(Tensors.shape, Rot.shape, Bonds.shape)
	for i in range(N//4):
		Tensors[i,:,:,:,:,:] = tsum('xijkl, ip, jq, kr, ls -> xpqrs', Tensors[i,:,:,:,:,:], Rot, Rot, Rot, Rot)
	
	print(Rot)
	Bonds[0,:,:] = tsum('ij, ip, jq->pq', Bonds[0,:,:], Rot, Rot)
	
	
	return Tensors, Bonds
	



##############################


