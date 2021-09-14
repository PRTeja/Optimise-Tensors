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
val = []
with open("CG_it") as f2:
			data = [line.rstrip('\n') for line in f2]
			

time = []
echo = []
for i in range(len(data)):
	lines = data[i].split()
	time.append(eval(lines[0]))
	echo.append(eval(lines[1]))
	
	
if 1:
	plt.figure(2)
	plt.tight_layout()
	plt.title((r' Func vs $CG-iterations$, $\chi =$'+str(Chimax) )) 	
	plt.ylabel('Fun value')#(r'$\frac{S.S(r,t) - S.S(r,0)}{S.S(r,0)}$')
	plt.xlabel(r'$iterations$')
	plt.yscale('log')
	plt.plot(time)
	#plt.savefig('Echotemp.png', bbox_inches='tight')
	plt.show()

		
