import numpy as np
from scipy import *
import scipy.optimize as opt
from scipy.linalg import sqrtm, expm
from matplotlib import pyplot as plt
import os.path
import time
import torch

from ctmrgt import *
from ReadTensors import *
from TimeEvolution import *
from Calculate import *
from ReadTens import *

#def main():
if 1:#__name__ == "__main__":
	start_time = time.time()
	Dlist = np.array([3,5,6])	
	
# C4v iPEPS on a square lattice S = 1/2
#for D in Dlist:
	D        = 3
	ds       = 2
	Chimax   = 16#16*4
	tol	 = 1e-12
	eps	 = [0.]#[1e-2, 1e-3, 1e-4]
	D2	 = D*D
	
	# Read short-range RVB wavefunction Tensors
	#Tensors, Bonds = ReadTensors(ds,D)
	
	Tensors, BondsD5 = ReadTensors_c4v(ds,D, N = 10)
	
	#sys.exit()	
	
	Bond = tsum('i,ijk    -> jk', [1.], BondsD5).T
	
	# Set parameters
	X = np.array([1., 0.])#, 0., 0., 0.,0.,0.,0., 0., 0.])
	if D==5:
		X = np.array([1., 0., 0., 0., 0.,0.,0.,0., 0., 0.])

	if D==0:
		X = np.array([1e-6,]*10)
		#for i in range(4):
		#	X[i] = np.random.random()
		
		X[0] = 1.
		X[1] = 5.75022302e-01
		X[2] = 2.52294532e-01
		X[3] = 9.35191942e-01
		#X = np.random.random(len(X))
		#X = [0.3654891516754517  ,0.18578496640735132  ,0.8343184247971781  ,0.3846359853734489  ,0.13141346507767182  ,0.15853213185621662  ,0.5916152891001248  ,0.05630635951559526  ,0.7480697365615984  ,0.9953933594133051]
		
	if D ==6:
		X = np.array([0.,]*11)
		X[10] = 1.
		X[7] = 0.25	
	
	if D ==7:
		N = Tensors.TensorsC4v.shape[0]
		X = np.array([1e-6,]*N)
		X[29] = 1.
		X[26] = 0.25	
	
	
	SaveParameters(X) 
	
#for noise in eps:
	#Construct the one-site tensor
	
	#X = np.random.random(4)
	#print(X)
	#A = AfromlambdaC4v(X, Tensors)
	
	Ry = np.array([[0,-1.], [1.,0.]])
	
	X0  = X#Xc4vtoXfew(X)#[1., 0.25]
	
	A0  = OneSite(X0,Tensors, Ry, True)
	
	A   = A0.A
	B   = A0.B
	E   = A0.E
	
	Xin = lambdafromA(A, Tensors.TensorsC4v)
	
	
	Sij, Sxx,Syy, Szz, Sijp, Spp, Smm, Sx, Sy, Sz = Hamiltonians(ds)
	H    = Sijp
	
	E1, E2   = HtoE(A,A, Sijp)
	
	
	B1 = B_singlet(A, Bond)
	
	print(Bond.real)
	print(np.allclose(B1, B))
	
	#A = Afromlambda(X0 + X1, Tensors)
	
	""
	#Initialize Corner and Edge Tensors
	C,T, Chi0 = InitializeCTM(E, Chimax)
	print(C.shape)
	print('X_start ::', X0)
	print('C4v symmetry ::',CheckC4vA(A))
	print('C4v symmetry ::',CheckC4vE(E))
	
	C,T = ReadCorner(D2,Chimax)
	
	print(np.allclose(tsum('ijkl -> ilkj',E), E, atol = 1e-15))
	
	C,T, period = PerformCTMRG(E, C,T, Chimax, tol, Saves =True, Ea = E1, Eb = E2)
	
	#print(np.diag(C))
	#sys.exit()
	#print('Periodicty ::', period)
	# Hamiltonians
	
	# Expectation value caclulation:
	if 1:
		############3
		norm, C1, C2 = Calculate2norm(C, T, E)
		
		Eax, Ebx = HtoE(A,A,Spp)
		Eay, Eby = HtoE(A,A,Smm)
		Eaz, Ebz = HtoE(A,B,Szz)
		E1, E2   = HtoE(A,A, Sijp)
		
		SSx = Calculate2(C, T, Eax,Ebx,C1, C2)/norm
		SSy = Calculate2(C, T, Eay,Eby,C1, C2)/norm
		SSz = Calculate2(C, T, Eaz,Ebz,C1, C2)/norm
		SSxyz0 = Calculate2(C, T, E1,E2,C1, C2)/norm
	
		SS  = -0.5*(SSx + SSy) + SSz
		
		print('Sx.Sx ::',SSx, 'Sy.Sy ::',SSy, 'Sz.Sz ::',SSz)
		print('S.S :: ', SSxyz0, SS)
		
		
		#############
		
		
	
	dtau = [0.001]#[0.1, 0.05, 0.01, 0.005, 0.001]#[0.05 for i in range(10)]#[0.1, 0.01, 0.01/2., 0.001]
	
	Ol  = []
	
	A1  = A0
	n=0
	for dt in dtau: 
	#for n in range(0):

		Gate = gateH(Sij,dt,Ry, False)
	
		#Xnew, fun, Erval, Ores = Hop(C,T, Tensors, A0,Gate)
		SaveParameters([dt])
		Xnew, fun, Erval, Ores, opt, echos, unopt = Hop(C,T, Tensors, A1,Gate)
		
		[o01, o02, o03, o04], [oin1, oin2, oin3, oin4], [in1, in2, in3, in4] = opt, echos, unopt
		
		#########
		Ores = 1 - ((1 - o01)*(1-o02)*(1-o03)*(1-o04))**(1./4.)
		
		Oin = 1 - ((1 - oin1)*(1-oin2)*(1-oin3)*(1-oin4))**(1./4.)
		
		init = 1 - ((1 - in1)*(1-in2)*(1-in3)*(1-in4))**(1./4.)
		########
	#	print(oin1, oin2, oin3, oin4)
	#	print(o01, o02, o03, o04)
		
		print(Ores, Oin, init)
		print(fun, Erval)
	
#		sys.exit()
		SaveParameters(X, (n+1)*dt)
		
		A1  = OneSite(Xnew, Tensors, Ry, True)

		A   = A1.A
		B   = A1.B
		E   = A1.E
		norm, C1, C2 = Calculate2norm(C, T, E)
		
		Eax, Ebx = HtoE(A,A,Spp)
		Eay, Eby = HtoE(A,A,Smm)
		Eaz, Ebz = HtoE(A,B,Szz)
		E1, E2   = HtoE(A,A, Sijp)
		SSx = Calculate2(C, T, Eax,Ebx,C1, C2)/norm
		SSy = Calculate2(C, T, Eay,Eby,C1, C2)/norm
		SSz = Calculate2(C, T, Eaz,Ebz,C1, C2)/norm
		SSxyz0 = Calculate2(C, T, E1,E2,C1, C2)/norm
		SS  = -0.5*(SSx + SSy) + SSz
		print('Sx.Sx ::',SSx, 'Sy.Sy ::',SSy, 'Sz.Sz ::',SSz)
		print('S.S :: ', SSxyz0, SS)
		
		Ax = tsum('aijkl, ax ->xijkl', A, Sx)
		Ay = tsum('aijkl, ax ->xijkl', A, Sy)
		Az = tsum('aijkl, ax ->xijkl', A, Sz)
		
		Ex = DoubleLayer(Ax,A)
		Ey = DoubleLayer(Ay,A)
		Ez = DoubleLayer(Az,A)
		
		norm0 = Onsite(C,T,E)
		
		print('<Sx>, <Sy>, <Sz> ::', (Onsite(C,T,Ex)/norm0).real, (Onsite(C,T,Ey)/norm0).real, (Onsite(C,T,Ez)/norm0).real)
				
		fun = SSxyz0.real
		
		Ol.append(fun)		

		E1, E2   = HtoE(A,A, Sijp)
	#	C,T, period = PerformCTMRG(E, C,T, Chimax, tol, Saves =False, Ea = E1, Eb = E2)
	
		
		f = open("Echo_En0",'a+')
		f.write(	str((1)*dt)+'\t'+ str(Ores) +'\t'+str(fun) +'\t'+ str(Erval)+'\t'+str(Oin)+'\t'+str(D) +'\n')
		f.close()
		
		f = open("Oin_substep",'a+')
		f.write(str((dt))+'\t'+ str(oin1) +'\t'+str(oin2) +'\t'+ str(oin3)+'\t'+str(oin4)+'\t'+str(D) +'\n')
		f.close()
		
		f = open("reference_substep",'a+')
		f.write(str((dt))+'\t'+ str(in1) +'\t'+str(in2) +'\t'+ str(in3)+'\t'+str(in4)+'\t'+str(D) +'\n')
		f.close()
		
		f = open("optimised_substep",'a+')
		f.write(str((dt))+'\t'+ str(o01) +'\t'+str(o02) +'\t'+ str(o03)+'\t'+str(o04)+'\t'+str(D) +'\n')
		f.close()
	
		plt.title((r'$\O$ vs $\tau$, $\chi = 16$ ' ) )	
		plt.ylabel(r'$O$')
		plt.xlabel(r'$\tau$')
	
#		plt.plot(dt, fun)
#		plt.show()
#		plt.savefig('Olap.png', bbox_inches='tight')
	
		print('total time ::', time.time() - start_time)
		
#for dt in dtau: 
if 0:
	for n in range(0):

		Gate = gateH(Sij,-dt,Ry, False)
	
		Xnew, fun, Erval, Ores, opt, echos, unopt = Hop(C,T, Tensors, A1,Gate)
		
		[o01, o02, o03, o04], [oin1, oin2, oin3, oin4], [in1, in2, in3, in4] = opt, echos, unopt
		
		
		SaveParameters(X, 10*dt - (n+1)*dt)
		
		A1  = OneSite(Xnew, Tensors, Ry, True)

		A   = A1.A
		B   = A1.B
		E   = A1.E
		
		E1, E2   = HtoE(A,A, Sijp)
		C,T, period = PerformCTMRG(E, C,T, Chimax, tol, Saves =False, Ea = E1, Eb = E2)
	
		norm, C1, C2 = Calculate2norm(C, T, E)
		
		Eax, Ebx = HtoE(A,A,Spp)
		Eay, Eby = HtoE(A,A,Smm)
		Eaz, Ebz = HtoE(A,B,Szz)
		E1, E2   = HtoE(A,A, Sijp)
		SSx = Calculate2(C, T, Eax,Ebx,C1, C2)/norm
		SSy = Calculate2(C, T, Eay,Eby,C1, C2)/norm
		SSz = Calculate2(C, T, Eaz,Ebz,C1, C2)/norm
		SSxyz0 = Calculate2(C, T, E1,E2,C1, C2)/norm
		SS  = -0.5*(SSx + SSy) + SSz
		print('Sx.Sx ::',SSx, 'Sy.Sy ::',SSy, 'Sz.Sz ::',SSz)
		print('S.S :: ', SSxyz0, SS)
		
		Ax = tsum('aijkl, ax ->xijkl', A, Sx)
		Ay = tsum('aijkl, ax ->xijkl', A, Sy)
		Az = tsum('aijkl, ax ->xijkl', A, Sz)
		
		Ex = DoubleLayer(Ax,A)
		Ey = DoubleLayer(Ay,A)
		Ez = DoubleLayer(Az,A)
		
		norm0 = Onsite(C,T,E)
		
		print('<Sx>, <Sy>, <Sz> ::', (Onsite(C,T,Ex)/norm0).real, (Onsite(C,T,Ey)/norm0).real, (Onsite(C,T,Ez)/norm0).real)
				
		

		Ol.append(fun)		

		fun = SSxyz0
		
		f = open("Echo_En",'a+')
		f.write(str(10*dt - (n+1)*dt)+'\t'+ str(Ores) +'\t'+str(fun) +'\t'+ str(Erval)+'\t'+str(D) +'\n')
		f.close()
	
		plt.title((r'$\O$ vs $\tau$, $\chi = 16$ ' ) )	
		plt.ylabel(r'$O$')
		plt.xlabel(r'$\tau$')
	
#		plt.plot(dt, fun)
#		plt.show()
#		plt.savefig('Olap.png', bbox_inches='tight')
	
		print('total time ::', time.time() - start_time)
