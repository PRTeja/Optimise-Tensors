import numpy as np
from scipy import *
import scipy.optimize as opt
from scipy.linalg import sqrtm, expm
from matplotlib import pyplot as plt
import os.path
import time
import torch
from Decs import *
#import tensorflow as tf

import numpy as np
from scipy import *
import scipy.optimize as opt
from scipy.linalg import sqrtm, expm
from matplotlib import pyplot as plt
import os.path
import time
import torch
from Decs import *
#import tensorflow as tf


def zz(*args,**kwargs):
     kwargs.update(dtype=np.complex128)
     return np.zeros(*args,**kwargs)  
			

Chimax = 16
with open("Echo_En0") as f2:
			data = [line.rstrip('\n') for line in f2]
	

t = [0.1,0.05, 0.01, 0.005,  0.001]
Dvals = [3,5]#[3,5,6]

En0 = -0.2933653240190329

time = []
echo = []

if 1:
	
		T = []
		O4= []
		En= []
		Err= []
		D =[]
		Oinit = []
		ratio = []
		new = []
			
for i in range(len(data)):
	lines = data[i].split()
	
	#print(data[i])
	
	ti = float(lines[0])
	O4i= eval(lines[1]).real
	Eni = abs((eval(lines[2]).real) - (-0.2933653240190329))
	Erri = eval(lines[3]).real
	Di  = int(lines[5])
	
	if len(lines)>=6:
		Oinit.append(eval(lines[4]).real)	
		ratio.append(O4i/Oinit[i])
	
	T.append(ti)
	O4.append(O4i)
	En.append(Eni)
	Err.append(Erri)
	D.append(Di)
	#new.append(float(lines[6]))
	
	#print(Oinit[i])


plx = []
ply = []
for d in Dvals:
	for i in range(len(D)):
		if D[i] == d:
			plx.append(T[i])
			ply.append(Oinit[i])
	
	if 1:
		#print(i, T, D, O4)
		
		plt.figure(2)
		plt.tight_layout()
		plt.title((r' $O(t=0)$ vs $D$, $\chi =$'+str(Chimax) )) 	
		plt.ylabel(r'|$O(t=0)|$')#(r'$\frac{S.S(r,t) - S.S(r,0)}{S.S(r,0)}$')
		plt.xlabel(r'$\tau$')
		plt.yscale('log')
		plt.plot(plx, ply, '--', label = r'$D = $'+str(d)+r' $O_{reference}$')
#		plt.show()	
	
	print(plx)
	print(ply)
	plx = []
	ply = []
		

plt.figure(2)
plt.legend(loc='best')
plt.savefig('Overlap_init_D356.png', bbox_inches='tight')
plt.show()
	

#sys.exit()

plx = []
ply = []
for d in Dvals:
	for i in range(len(D)):
		if D[i] == d:
			plx.append(T[i])
			ply.append(O4[i])
	
	if 1:
		#print(i, T, D, O4)
		
		plt.figure(3)
		plt.tight_layout()
		plt.title((r' $O(\tau)$ vs $D$, $\chi =$'+str(Chimax) )) 	
		plt.ylabel(r'$O(\tau)$')#(r'$\frac{S.S(r,t) - S.S(r,0)}{S.S(r,0)}$')
		plt.xlabel(r'$\tau$')
		plt.yscale('log')
		plt.plot(plx, ply, '-x', label = r'$D = $'+str(d)+r' O(t=$\tau$)')
#		plt.show()	
	
	plx = []
	ply = []
		

plt.figure(3)
plt.legend(loc='best')
plt.savefig('OptD356_compare.png', bbox_inches='tight')
plt.show()


########################
##########################
########################3
##########################
############################

plx = []
ply = []
for d in Dvals:
	for i in range(len(D)):
		if D[i] == d:
			plx.append(T[i])
			ply.append(O4[i])
	
	if 1:
		print(i, T, D, O4)
		
		plt.figure(4)
		plt.tight_layout()
		plt.title((r' $O(\tau)$ vs $D$, $\chi =$'+str(Chimax) )) 	
		plt.ylabel(r'$O(\tau)$')#(r'$\frac{S.S(r,t) - S.S(r,0)}{S.S(r,0)}$')
		plt.xlabel(r'$\tau$')
		plt.yscale('log')
		plt.plot(plx, ply, '-x', label = r'$D = $'+str(d)+r' O(t=$\tau$)')
#		plt.show()	
	
	plx = []
	ply = []
		

plt.figure(4)
plt.legend(loc='best')
#plt.savefig('Overlap_init_D356.png', bbox_inches='tight')
#plt.show()	




plx = []
ply = []
for d in [Dvals[0]]:
	for i in range(len(D)):
		if D[i] == d:
			plx.append(T[i])
			ply.append(Oinit[i])
	
	if 1:
		#print(i, T, D, O4)
		
		plt.figure(4)
		plt.tight_layout()
		plt.title((r' $O$ vs $\tau$, $\chi =$'+str(Chimax) )) 	
		#plt.ylabel(r'|$O(t=0)|$')#(r'$\frac{S.S(r,t) - S.S(r,0)}{S.S(r,0)}$')
		plt.xlabel(r'$\tau$')
		plt.yscale('log')
		plt.plot(plx, ply, '-.', label = r'$D = $'+str(d)+r' $O_{reference}$')
#		plt.show()	
	
	plx = []
	ply = []
		

plt.figure(4)
plt.legend(loc='best')
plt.savefig('ALL_D356_compare.png', bbox_inches='tight')




plx = []
ply = []
for d in Dvals:
	for i in range(len(D)):
		if D[i] == d:
			plx.append(T[i])
			ply.append(Err[i])
	
	if 1:
		#print(i, T, D, O4)
		
		plt.figure(4)
		plt.tight_layout()
		plt.title((r' $Deviation$ vs $D$, $\chi =$'+str(Chimax) )) 	
		plt.ylabel(r'$O(\tau)$')#(r'$\frac{S.S(r,t) - S.S(r,0)}{S.S(r,0)}$')
		plt.xlabel(r'$\tau$')
		plt.yscale('log')
		plt.plot(plx, ply, '-.', label = r'$D = $'+str(d)+r' Deviation')
#		plt.show()	
	
	plx = []
	ply = []
		

plt.figure(4)
plt.legend(loc='best')
plt.savefig('ErrD356_compare.png', bbox_inches='tight')
#plt.show()

plt.show()


plx = []
ply = []
for d in Dvals:
	for i in range(len(D)):
		if D[i] == d:
			plx.append(T[i])
			ply.append(Err[i])
	
			print(T[i], Err[i])
	if 1:
		#print(i, T, D, O4)
		
		plt.figure(4)
		plt.tight_layout()
		plt.title((r' $Deviation$ vs $\tau$, $\chi =$'+str(Chimax) )) 	
		#plt.ylabel(r'|$O(t=0)|$')#(r'$\frac{S.S(r,t) - S.S(r,0)}{S.S(r,0)}$')
		plt.xlabel(r'$\tau$')
		plt.yscale('log')
		plt.plot(plx, ply, '-v',  label = r'$D = $'+str(d)+r' Deviation')
#		plt.show()	
	
	plx = []
	ply = []
		
plx = []
ply = []
for d in Dvals:
	for i in range(len(D)):
		if D[i] == d:
			plx.append(T[i])
		#	ply.append(new[i]/4)
			print(T[i], new[i])
	if 1:
		#print(i, T, D, O4)
		
		plt.figure(4)
		plt.tight_layout()
		plt.title((r' $Deviation$ vs $\tau$, $\chi =$'+str(Chimax) )) 	
		#plt.ylabel(r'|$O(t=0)|$')#(r'$\frac{S.S(r,t) - S.S(r,0)}{S.S(r,0)}$')
		
		plt.plot(plx, ply, '-x',  label = r'$D = $'+str(d)+r'<Deviation_substep>')
#		plt.show()	
	
	plx = []
	ply = []
plt.figure(4)
plt.legend(loc='best')
plt.savefig('compare0.png', bbox_inches='tight')
plt.show()


sys.exit()


########################


