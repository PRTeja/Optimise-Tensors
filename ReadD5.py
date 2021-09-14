
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

def ntoc(a):
	if a == 0:
		return 'h'
	if a == 1 or a == 2:
		return 'r'
	if a ==3 or a ==4:
		return 'b'


def TakeInput(index):
	ind = h310toindex(index)
	if ind:
		return 1, ind
	
	ind = h130toindex(index)
	if ind:
		return 2, ind
	

def indexnumreturn(index):
	
	#########33
	ind = h310toindex(index)
	if ind is not None:
		return 0, ind
	
	ind = h130toindex(index)
	if ind is not None:
		return 1, ind
		
	ind = b301toindex(index)
	if ind is not None:
		return 2, ind
		
	ind = b103toindex(index)
	if ind is not None:
		return 3, ind
	
	###############
	ind = rb0totype_121_1(index)
	if ind is not None:
		return 4, ind
		
	ind = rb0totype_121_2(index)
	if ind is not None:
		return 5, ind
	
	ind = rb0totype_112_1(index)
	if ind is not None:
		return 6, ind
	
	ind = rb0totype_112_2(index)
	if ind is not None:
		return 7, ind
		
		
def h310toindex(index):
	
	list1 = [np.array(['r','h','h','h']), np.array(['h','r','h','h']), np.array(['h','h','r','h']), np.array(['h','h','h','r'])] 
	totallist = list1
	indexnum = np.arange(len(totallist))
	
	for i in range(len(list1)):
		if (totallist[i] == [ntoc(index[0]), ntoc(index[1]), ntoc(index[2]), ntoc(index[3]) ]).all():
			return indexnum[i]
	
	return None
	
def h130toindex(index):
	list1 = [np.array(['h','r','r','r']), np.array(['r','h','r','r']), np.array(['r','r','h','r']), np.array(['r','r','r','h'])] 
	totallist = list1
	indexnum = np.arange(len(totallist))
	for i in range(len(list1)):
		if (totallist[i] == [ntoc(index[0]), ntoc(index[1]), ntoc(index[2]), ntoc(index[3]) ]).all():
			return indexnum[i]
	
	return None
	
def b301toindex(index):
	list1 = [np.array(['b','h','h','h']), np.array(['h','b','h','h']), np.array(['h','h','b','h']), np.array(['h','h','h','b'])] 
	totallist = list1
	indexnum = np.arange(len(totallist))
	for i in range(len(list1)):
		if (totallist[i] == [ntoc(index[0]), ntoc(index[1]), ntoc(index[2]), ntoc(index[3]) ]).all():
			return indexnum[i]
	return None
	
def b103toindex(index):
	
	list1 = [np.array(['h','b','b','b']), np.array(['b','h','b','b']), np.array(['b','b','h','b']), np.array(['b','b','b','h'])] 
	totallist = list1
	indexnum = np.arange(len(totallist))
	for i in range(len(list1)):
		if (totallist[i] == [ntoc(index[0]), ntoc(index[1]), ntoc(index[2]), ntoc(index[3]) ]).all():
			return indexnum[i]
	
	return None
	
def rb0totype_121_1(index):
	list1 = [np.array(['h','r','b','r']), np.array(['r','h','r','b']), np.array(['b','r','h','r']), np.array(['r','b','r','h'])] 
	
	totallist = list1
	indexnum = np.arange(len(totallist))
	for i in range(len(list1)):
		if (totallist[i] == [ntoc(index[0]), ntoc(index[1]), ntoc(index[2]), ntoc(index[3]) ]).all():
			return indexnum[i]
	
	return None		

def rb0totype_121_2(index):
	# With reflection symmetry
	#Without reflection symmetry. one can move from list 2 to list 3 via reflection
	list2 = [np.array(['h','r','r','b']), np.array(['b','h','r','r']), np.array(['r','b','h','r']), np.array(['r','r','b','h'])]
	
	list3 = [np.array(['h','b','r','r']), np.array(['r','h','b','r']), np.array(['r','r','h','b']), np.array(['b','r','r','h']) ]
	
	totallist = list2 + list3
	indexnum = np.arange(len(totallist))
	for i in range(len(totallist)):
		if (totallist[i] == [ntoc(index[0]), ntoc(index[1]), ntoc(index[2]), ntoc(index[3]) ]).all():
			return indexnum[i]
	
	return None	

def rb0totype_112_1(index):
	# With reflection symmetry
	list1 = [np.array(['h','b','r','b']), np.array(['b','h','b','r']), np.array(['r','b','h','b']), np.array(['b','r','b','h'])] 
	
	totallist = list1
	indexnum = np.arange(len(totallist))
	for i in range(len(totallist)):
		if (totallist[i] == [ntoc(index[0]), ntoc(index[1]), ntoc(index[2]), ntoc(index[3]) ]).all():
			return indexnum[i]
	
	return None	
		
def rb0totype_112_2(index):
	#Without reflection symmetry. one can move from list 2 to list 3 via reflection
	list2 = [np.array(['h','b','b','r']), np.array(['r','h','b','b']), np.array(['b','r','h','b']), np.array(['b','b','r','h'])]
	
	list3 = [np.array(['h','r','b','b']), np.array(['b','h','r','b']), np.array(['b','b','h','r']), np.array(['r','b','b','h']) ]
	
	totallist =  list2 + list3
	indexnum = np.arange(len(totallist))
	for i in range(len(totallist)):
		if (totallist[i] == [ntoc(index[0]), ntoc(index[1]), ntoc(index[2]), ntoc(index[3]) ]).all():
			return indexnum[i]
	
	return None
	
def readindexD5(index):
	
	
	mainindex, freeindex = indexnumreturn(index[1:5])#[ ntoc(index[1]), ntoc(index[2]), ntoc(index[3]), ntoc(index[4])  ] )
	
	return mainindex, freeindex

