
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



def h31toindexD3(index):
	
	list1 = [np.array(['r','h','h','h']), np.array(['h','r','h','h']), np.array(['h','h','r','h']), np.array(['h','h','h','r'])] 
	totallist = list1
	indexnum = np.arange(len(totallist))
	for i in range(len(list1)):
		if (list1[i] == [numtorb0(index[0]), numtorb0(index[1]), numtorb0(index[2]), numtorb0(index[3]) ]).all():
			return indexnum[i]
	
	return None
	
def h13toindexD3(index):
	
	list1 = [np.array(['h','r','r','r']), np.array(['r','h','r','r']), np.array(['r','r','h','r']), np.array(['r','r','r','h'])] 
	totallist = list1
	indexnum = np.arange(len(totallist))
	for i in range(len(list1)):
		if (list1[i] == [numtorb0(index[0]), numtorb0(index[1]), numtorb0(index[2]), numtorb0(index[3]) ]).all():
			return indexnum[i]
	
	return None



def TakeInputD3(index):
	ind = h310toindex(index)
	if ind:
		return 1, ind
	
	ind = h130toindex(index)
	if ind:
		return 2, ind


