import numpy as np
import scipy
import time
from matplotlib import pyplot as plt
from decimal import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import os.path
import torch

from Basics import *
from Calculate import *
from Decs import *

def InitializeCTM(E, Chi0): # Initialie Corner and Edge Tensors
	
	D2 = E.shape[0]
	D  = int(np.sqrt(D2))
	
	# Initialie Tensors. C = Corner Matrix. T = Edge Tensor
	T = zz((D2, D2, D2))
	C = zz((D2, D2))
	
	T = tsum('ijkkl', E.reshape(D2, D2, D,D, D*D))	
	C = tsum('ijkk', T.reshape(D2,D2,D,D))
	
	Chi = D2
	
	while (Chi0)>Chi*D2:
		
		
		# Add Elementary Tensor E to C and T. Chi = Chi*D2
		T2 = tsum('ijkl, kmn -> ijmln ', E, T).reshape(D2, D2*Chi, D2*Chi)
		
		C2  = buildC2(C,T,T2, E)
		# Alternative:
		#C2 = tsum('ijkl, kmn, no, lop -> ipjm', E, T, C, T).reshape(D2*Chi, D2*Chi)
		
		T =T2
		C =C2
		Chi= Chi*D2
		
		
	return (C,T, Chi)

## Read Corner and Edge
def ReadCorner(D2,Chimax):
	D = int(np.sqrt(D2))
	if os.path.isfile('Environment/Corner'+str(D)+'_'+str(Chimax)):
		print('Reading Corner and Edge')
		Corner = zz((Chimax, Chimax))
		TEdge  = zz((D2,Chimax, Chimax))
		f = open('Environment/Corner'+str(D)+'_'+str(Chimax),'r')	
		f1 = f.readlines()
		for lines in f1:
			line = lines.split()
			index, realvalue, imagvalue = int(line[0]), float(line[1]), float(line[2])
			Corner[index, index] = complex(realvalue, imagvalue)
		f.close()
		f = open('Environment/TEdge'+str(D)+'_'+str(Chimax),'r')	
		f1 = f.readlines()
		for lines in f1:
			line = lines.split()		
			i1, i2, i3, realvalue, imagvalue= int(line[0]), int(line[1]), int(line[2]), float(line[3]), float(line[4])
			TEdge[i1,i2,i3] = complex(realvalue, imagvalue)
		print('****************** Read Environment from a file *****************')
		
		return (Corner, TEdge) 
	else:
		
		sys.exit('No file to read Corner from')
	
def ReadCorner_Manual(D2,Chimax, C,T,Corner_path, TEdge_path):
	D = int(np.sqrt(D2))
	if os.path.isfile(Corner_path):#('Environment/Corner'+str(D)+'_'+str(Chimax)):
		print('Reading Corner and Edge')
		Corner = zz((Chimax, Chimax))
		TEdge  = zz((D2,Chimax, Chimax))
		f = open(Corner_path, 'r')#('Environment/Corner'+str(D)+'_'+str(Chimax),'r')	
		f1 = f.readlines()
		for lines in f1:
			line = lines.split()
			index, realvalue, imagvalue = int(line[0]), float(line[1]), float(line[2])
			Corner[index, index] = complex(realvalue, imagvalue)
		f.close()
		f = open(TEdge_path, 'r')#('Environment/TEdge'+str(D)+'_'+str(Chimax),'r')	
		f1 = f.readlines()
		for lines in f1:
			line = lines.split()		
			i1, i2, i3, realvalue, imagvalue= int(line[0]), int(line[1]), int(line[2]), float(line[3]), float(line[4])
			TEdge[i1,i2,i3] = complex(realvalue, imagvalue)
		print('****************** Read Environment from a file *****************')
		
		return (Corner, TEdge) 
	else:
		return (C,T)
		#sys.exit('No file to read Corner from')
	


#### Check symmetries
def checkTSymmetry(T):
	return np.allclose(tsum('aij-> aji', T), T, atol=1e-13)

def checkhermiticity(T):
	D2 = T.shape[0]
	Chi= T.shape[1]
	D  = int(np.sqrt(D2))
	
	T1 = tsum('abij->baij', T.reshape(D,D,Chi,Chi)).reshape(D2,Chi,Chi)
	
	return np.allclose(abs(T), abs(T), atol=1e-14)

#####################
#####################
## Basic CTMRG. Involves 3 steps
## 1. Build Bigger Corner and Edge Tensor
## 2. Truncate Corner and Obtain Isometries
## 3. Truncate Edge Tensor
#####################
#####################

def Iterate(C,T,Z,E, Chimax):
	
	Chi = C.shape[0]
	D2  = T.shape[0]
	
	# Build T2
	T2 = tsum('ijkl, kmn -> ijmln ', E, T).reshape(D2, D2*Chi, D2*Chi)
	
	# Build C2
	C  = tsum('ij,jk -> ik', C, Z)
	C2 = buildC2(C,T,T2,E)
	
	# Uncomment the next to lines to Scale C2 and T2 down
	max1 = np.max([np.max(abs(C2)), np.max(abs(T2))])
	#if max1 < 1e-10:
	#print(np.round(C,12).shape)
	print('############', max1, np.max(abs(C2)), np.max(abs(T2)))
	T2 = T2/np.max(abs(T2))#max1
	C2 = C2/np.max(abs(C2))#max1
	
	# Alternatively: 
	#tsum('ijkl, kmn, no, lop -> ipjm', E, T2, C, T2).reshape(D2*Chi, D2*Chi)
	
	#Truncate C2 matrix, get isometry LM
	(Chi, C, Z, LM, RM) = TruncateC2(C2, Chimax)
	
	#Truncate T2 Tensor:
	T = TruncateT2(T2, LM, RM)
	
	return (C,T,Z,Chi)#(C/np.max(abs(C)),T/np.max(abs(C)),Z, Chi)
	

###### STEP-1
def buildT2(T,E): # Add E to T
	D2   = T.shape[0]
	Chi  = T.shape[1]
	return tsum('ijkl, kmn -> ijmln ', E, T).reshape(D2, D2*Chi, D2*Chi)

def buildC2(C,T,T2, E):
	Chi = C.shape[0]
	D2  = T.shape[0]
	
	# Add T to C
	CT = tsum('ijk, jl -> kil', T, C).reshape(Chi, D2*Chi)
	
	# Add T2 to T and C
	return tsum('ij, lmj -> lim', CT, T2).reshape(D2*Chi, D2*Chi)

######## STEP-3
def TruncateC2(C2, Chimax):
	
	Z   = np.eye(C2.shape[0])
	
	#C2  = (C2 + C2.T)/2.
	s,u = Takagipartial(C2, Chimax+2)#Takagipartial(C2)
	
	#s,u = EigenDecomposition(C2)
	
	print('~@~',np.linalg.norm(C2 - u @ np.diag(s) @ u.T), len(C2))
	
	#uin = scipy.linalg.inv(u)
	#Z   = scipy.linalg.inv(u.T @ u ) 
	
	# If Takagi decomposition failed:
	if len(s) ==2:
		Z = s[1]
		s = s[0]
	
	"""
	if np.allclose(C2, u @ np.diag(s) @ u.T):
		print('EIGEN OK')
	else:
		#print(np.allclose(Z, np.eye(Z.shape[0])))
		print(np.allclose(C2, C2.real, atol=1e-15))
		print('REAL')
		sys.exit()
	#"""
	trunc = False
	cut0 = Chimax # Dynamic cut variable
	
	# To Avoid cutting between multiplets: 
	scut = s[cut0]
	while (trunc ==False and cut0 <=Chimax):		
		if scut ==0.:
			trunc = True
		else:
			ratio = abs(s[cut0-1]/scut)
			if abs(ratio -1.) > 1e-1:
				trunc = True
			else:
				cut0 -= 1
	
	sort = np.arange(cut0)
	Sig = np.diag(s[sort])
	LM    = u[:,sort]
	RM    = LM#.conjugate()#uin[sort,:].T#vh[sort,:].T#LM.T
	Z = Z[0:len(sort), 0:len(sort)]
	
	return (cut0, Sig, Z, LM, RM)
	
def TruncateT2(T2, LM, RM): 
	return tsum('aij, ix, jy', T2, LM, RM)
		
##########################################################
########### SAVING CORNER AND EDGE TENSORS ###############
##########################################################
def SaveEnv(Corner, TEdge, Chimax, tol):
		D2 = TEdge.shape[0]
		D  = int(np.sqrt(D2))
		
		if not os.path.isdir('Environment/'):
			os.mkdir('Environment')
		f = open("Environment/Corner"+str(D)+'_'+str(Chimax),'w+')
		for i in range(Corner.shape[0]):
			if abs(Corner[i,i].imag) > tol:
				img = Corner[i,i].imag
			else:
				img =0.# Corner[i,i].imag#0.
				
			f.write(str(i)+'\t'+str(Corner[i,i].real/abs(Corner[0,0])) +'\t'+ str(img/abs(Corner[0,0]))+'\n')
		f.close()

		f = open("Environment/TEdge"+str(D)+'_'+str(Chimax),'w+')
		for i in range(TEdge.shape[0]):
			for j in range(TEdge.shape[1]):
				for k in range(TEdge.shape[2]):
					if abs(TEdge[i,j,k]) >10.**(-14):
						strr = str(i)+'\t'+str(j)+'\t'+str(k)+'\t'+str(TEdge[i,j,k].real)
						if abs(TEdge[i,j,k].imag) > tol:
							strr = strr + '\t'+str(TEdge[i,j,k].imag)+'\n'
						else:
							strr = strr + '\t'+str(TEdge[i,j,k].imag)+'\n'#str(0.)+'\n'
						f.write(strr)
		f.close()
		
def SaveEnv_Manual(Corner, TEdge, Chimax, tol, index, path = "Time_Env"):

		D2 = TEdge.shape[0]
		D  = int(np.sqrt(D2))
		
		if not os.path.isdir(path):
			os.mkdir(path)
		f = open(path + "/Corner"+str(D)+'_'+str(Chimax)+'_'+str(index),'w+')
		for i in range(Corner.shape[0]):
			if abs(Corner[i,i].imag) > tol:
				img = Corner[i,i].imag
			else:
				img =0.# Corner[i,i].imag#0.
				
			f.write(str(i)+'\t'+str(Corner[i,i].real/abs(Corner[0,0])) +'\t'+ str(img/abs(Corner[0,0]))+'\n')
		f.close()

		f = open(path + "/TEdge"+str(D)+'_'+str(Chimax)+'_'+str(index),'w+')
		for i in range(TEdge.shape[0]):
			for j in range(TEdge.shape[1]):
				for k in range(TEdge.shape[2]):
					if abs(TEdge[i,j,k]) >10.**(-14):
						strr = str(i)+'\t'+str(j)+'\t'+str(k)+'\t'+str(TEdge[i,j,k].real)
						if abs(TEdge[i,j,k].imag) > tol:
							strr = strr + '\t'+str(TEdge[i,j,k].imag)+'\n'
						else:
							strr = strr + '\t'+str(TEdge[i,j,k].imag)+'\n'#str(0.)+'\n'
						f.write(strr)
		f.close()
###########################################################
def checkconvergence(C, Cnew,m, tol):
	Chi1 = C.shape[0]
	Chi2 = Cnew.shape[0]
	stdev= 0.
	for i in range(min(Chi1,Chi2,m)):
		stdev += (abs(C[i,i]) - abs(Cnew[i,i]))**2
	stdev = np.sqrt(stdev)/min(Chi1,Chi2,m)
	
	if stdev < tol:
		return True, stdev
	else:
		return False, stdev

def checkrecursion(dev):
	l = len(dev)
	
	val = dev[0]
	for i in range(1,l-1):
		c = 1 - abs(dev[i]/val)
		#print(c, dev[i], val)
		if abs(c) < 1e-11:
			return True, i
	
	
	#cor = np.correlate(dev,dev)[0]
	#for irec in range(1,l-1):
	#	c = 1 - abs(np.correlate(dev, np.roll(dev, irec))[0]/cor)
	#	if abs(c) < 1e-9:
	#		return True
	#	print(c)
	return False, 0 
	

def iPEPS(E, C, T, Chimax, tol, Nsteps, Ea = None, Eb = None):
	# Start time
	inittime = time.time()
	D2= E.shape[0]
	
	#initialize
	Z = np.eye(C.shape[0])
	converge = False
	period   = 0.
	dev = []
	
	for isteps in range(1,Nsteps):
		if 0:#not checkTSymmetry(T):
			sys.exit('Edge not symmetric')
		
		norm, C1, C2 = Calculate2norm(C, T, E)
		SSxyz = Calculate2(C, T, Ea,Eb,C1, C2)/norm
		
			
		(Cnew, Tnew, Znew, Chi) = Iterate(C,T,Z,E, Chimax)
		spect = np.diag(Cnew)/np.max(abs(Cnew))
		#np.linalg.norm(np.round(Cnew @ Znew, 13)- np.round(C @ Z,13))
		if 0:#not checkTSymmetry(Tnew):
			print(checkTSymmetry(Tnew))
			sys.exit('Edgenew-1 not symmetric')
		converge, std = checkconvergence(C, Cnew, D2*2, tol)
		
		std = np.abs(SSxyz)
		dev.append(std)
		
		if not converge and isteps%52 ==0: 
			#print(dev[isteps-20:isteps])
			converge, period = checkrecursion(dev[isteps-50:isteps])
			
		print(isteps, std, checkhermiticity(T), (spect[np.arange(5)])/np.max(abs(spect)))
		if isteps%50 ==0:
			plt.clf()
			plt.yscale('log')
			plt.plot(dev)
			plt.savefig('convs.png')
			#plt.show()
			
		if converge:#dev <1e-10:
			converge = True
			plt.clf()
			plt.yscale('log')
			plt.plot(dev)
			plt.savefig('convs1.png')
			
			endtime = time.time()
			print('CTMRG converged, error ::', std)
			print('Time taken ::', int((endtime - inittime)//60), 'min', (endtime -inittime)%60, 'sec')
			return (C @ Z , T, Chi, isteps, period)
		
		C = Cnew
		T = Tnew
		Z = Znew
	
	return (C @ Z , T, Chi, isteps, period)
		
def PerformCTMRG(E, C,T, Chimax, tol, Saves, Ea = None, Eb = None):
	(C,T, Chi, steps, period) = iPEPS(E, C, T, Chimax, tol, Nsteps=2000, Ea = Ea, Eb = Eb)
	
	if Saves:
		SaveEnv(C, T, Chimax, tol)
	return C, T, period
	
	
	
