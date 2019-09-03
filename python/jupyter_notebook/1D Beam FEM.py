# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import sys
import time


def inpdata_1dfrm(fnameR,nod,nfree):
    f=open(fnameR,'r')
    text=f.readline()
    text=text.strip()
    text=text.split()
    npoin=int(text[0]) # Number of nodes
    nele =int(text[1]) # Number of elements
    nsec =int(text[2]) # Number of sections
    npfix=int(text[3]) # Number of restricted nodes
    nlod =int(text[4]) # Number of loaded nodes
    # array declaration
    ae    =np.zeros((5,nsec),dtype=np.float64)      # Section characteristics
    node  =np.zeros((nod+1,nele),dtype=np.int)       # node-element relationship
    x     =np.zeros((3,npoin),dtype=np.float64)      # Coordinates of nodes
    mpfix =np.zeros((nfree,npoin),dtype=np.int)      # Ristrict conditions
    rdis  =np.zeros((nfree,npoin),dtype=np.float64)  # Ristricted displacement
    fp    =np.zeros(nfree*npoin,dtype=np.float64)    # External force vector
    # section characteristics
    for i in range(0,nsec):
        text=f.readline()
        text=text.strip()
        text=text.split()
        ae[0,i] =float(text[0])  # E     : elastic modulus
        ae[1,i] =float(text[1])  # po    : Poisson's ratio
        ae[2,i] =float(text[2])  # a     : section area
        ae[3,i] =float(text[3])  # aix   : tortional constant
        ae[4,i] =float(text[4])  # gamma : unit weight of material
    # element-node
    for i in range(0,nele):
        text=f.readline()
        text=text.strip()
        text=text.split()
        node[0,i]=int(text[0]) #node_1
        node[1,i]=int(text[1]) #node_2
        node[2,i]=int(text[2]) #section characteristic number
    # node coordinates
    for i in range(0,npoin):
        text=f.readline()
        text=text.strip()
        text=text.split()
        x[0,i]=float(text[0])    # x-coordinate
        x[1,i]=float(text[1])    # y-coordinate
        x[2,i]=float(text[2])    # z-coordinate
    # boundary conditions (0:free, 1:restricted)
    for i in range(0,npfix):
        text=f.readline()
        text=text.strip()
        text=text.split()
        lp=int(text[0])              #fixed node
        mpfix[0,lp-1]=int(text[1])   #fixed in x-direction
        mpfix[1,lp-1]=int(text[2])   #fixed in y-direction
        mpfix[2,lp-1]=int(text[3])   #fixed in z-direction
        mpfix[3,lp-1]=int(text[4])   #fixed in rotation around x-axis
        mpfix[4,lp-1]=int(text[5])   #fixed in rotation around y-axis
        mpfix[5,lp-1]=int(text[6])   #fixed in rotation around z-axis
        rdis[0,lp-1]=float(text[7])  #fixed displacement in x-direction
        rdis[1,lp-1]=float(text[8])  #fixed displacement in y-direction
        rdis[2,lp-1]=float(text[9])  #fixed displacement in z-direction
        rdis[3,lp-1]=float(text[10]) #fixed rotation around x-axis
        rdis[4,lp-1]=float(text[11]) #fixed rotation around y-axis
        rdis[5,lp-1]=float(text[12]) #fixed rotation around z-axis
    # load
    if (nlod > 0):
        for i in range(0,nlod):
            text=f.readline()
            text=text.strip()
            text=text.split()
            lp=int(text[0])           #loaded node
            fp[6*lp-6]=float(text[1]) #load in x-direction
            fp[6*lp-5]=float(text[2]) #load in y-direction
            fp[6*lp-4]=float(text[3]) #load in z-direction
            fp[6*lp-3]=float(text[4]) #moment around x-axis
            fp[6*lp-2]=float(text[5]) #moment around y-axis
            fp[6*lp-1]=float(text[6]) #moment around z-axis
    f.close()
    return npoin,nele,nsec,npfix,nlod,ae,node,x,mpfix,rdis,fp


inpdata_1dfrm('test1.txt',nod = 3,nfree = 6)


