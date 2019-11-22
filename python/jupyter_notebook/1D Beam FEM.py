# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
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


def inputdata(fnameR,nod,nfree): #データパス, 要素節点数, 自由度
    f=open(fnameR,'r')
    text=f.readline()
    text=text.strip()
    text=text.split()
    npoin=int(text[0]) # ノード数
    nele =int(text[1]) # 要素数
    npfix=int(text[2]) # 拘束点数
    nlod =int(text[3]) # 荷重点数
    # 配列宣言
    ae    =np.zeros((2,nele),dtype=np.float64)       # 要素特性
    node  =np.zeros((nod+1,nele),dtype=np.int)       # 要素構成節点
    x     =np.zeros((2,npoin),dtype=np.float64)      # 座標
    mpfix =np.zeros((nfree,npoin),dtype=np.int)      # 拘束状態
    rdis  =np.zeros((nfree,npoin),dtype=np.float64)  # 既知変位
    fp    =np.zeros(nfree*npoin,dtype=np.float64)    # 外力
    # 要素特性
    for i in range(0,nele):
        text=f.readline()
        text=text.strip()
        text=text.split()
        ae[0,i] =float(text[0])  # E
        ae[1,i] =float(text[1])  # Ix

    # 要素構成節点
    for i in range(0,nele):
        text=f.readline()
        text=text.strip()
        text=text.split()
        node[0,i]=int(text[0]) #node_1
        node[1,i]=int(text[1]) #node_2
        node[2,i]=int(text[2]) #要素番号
    # 座標
    for i in range(0,npoin):
        text=f.readline()
        text=text.strip()
        text=text.split()
        x[0,i]=float(text[0])    # x-座標
        x[1,i]=float(text[1])    # z-座標
    # 境界条件（拘束状態） (0:free, 1:restricted)
    for i in range(0,npfix):
        text=f.readline()
        text=text.strip()
        text=text.split()
        lp=int(text[0])              #fixed node
        mpfix[0,lp-1]=int(text[1])   #fixed in direction
        mpfix[1,lp-1]=int(text[2])   #fixed in rotation
        rdis[0,lp-1]=float(text[3])  #fixed displacement
        rdis[1,lp-1]=float(text[4])  #fixed rotation

    # 荷重
    if (nlod > 0):
        for i in range(0,nlod):
            text=f.readline()
            text=text.strip()
            text=text.split()
            lp=int(text[0])           
            fp[2*lp-2]=float(text[1]) #荷重
            fp[2*lp-1]=float(text[2]) #モーメント
    f.close()
    return npoin,nele,npfix,nlod,ae,node,x,mpfix,rdis,fp


def els_mat(EI, L): 
    ek=np.zeros((4,4),dtype=np.float64) # local stiffness matrix
    C = EI / (L**3)
    ek[0, 0] = 12
    ek[0, 1] = 6 * L
    ek[0, 2] = -12
    ek[0, 3] = 6 * L
    ek[1, 0] = 6 * L
    ek[1, 1] = 4 * (L**2)
    ek[1, 2] = -6 * L
    ek[1, 3] = 2 * (L**2)
    ek[2, 0] = -12
    ek[2, 1] = -6 * L
    ek[2, 2] = 12
    ek[2, 3] = -6 * L
    ek[3, 0] = 6 * L
    ek[3, 1] = 2 * (L**2)
    ek[3, 2] = -6 * L
    ek[3, 3] = 4 * (L**2)
    return C * ek


def calelef(EI, L, dis):
    ek=eles_mat(EI, L) # Stiffness matrix in local coordinate
    secf=np.dot(ek, dis)
    return secf


def main():
    start=time.time()
    args = sys.argv
    fnameR=args[1] # input data file
    fnameW=args[2] # output data file
    nod=2              # 1要素での節点数
    nfree=2            # 自由度
    # データ読み込み
    npoin,nele,npfix,nlod,ae,node,x,mpfix,rdis,fp = inputdata('test1.txt',nod = 2,nfree = 2)
    # 配列宣言
    ir=np.zeros(nod*nfree,dtype=np.int)   
    gk=np.zeros((nfree*npoin,nfree*npoin),dtype=np.float64)   # Global stiffness matrix
    # assembly of stiffness matrix & load vectors
    for ne in range(0,nele):
        i=node[0,ne]-1 # 最初の方のノード番号 -1
        j=node[1,ne]-1 # 二番目のノード番号 -1
        m=node[2,ne]-1 # 要素の番号 -1
        x1=x[0,i]; z1=x[1,i];
        x2=x[0,j]; z2=x[1,j];
        ee   =ae[0,m]  
        aix  =ae[1,m]  # 断面二次モーメント
        EI = ee*aix
        ek   =eles_mat(EI, abs(x2 - x1))             # local stiffness matrix 
        ir[3] = 2*j+1; ir[2] = ir[3] - 1; 
        ir[1] = 2*i+1; ir[0] = ir[1] - 1;
        for i in range(0,nod*nfree):
            it=ir[i]
            for j in range(0,nod*nfree):
                jt=ir[j]
                gk[it,jt]+=ek[i,j]
    # treatment of boundary conditions
    for i in range(0,npoin):
        for j in range(0,nfree):
            if mpfix[j,i]==1:
                iz=i*nfree+j
                fp[iz]=0.0
    for i in range(0,npoin):
        for j in range(0,nfree):
            if mpfix[j,i]==1:
                iz=i*nfree+j
                fp=fp-rdis[j,i]*gk[:,iz]
                gk[:,iz]=0.0
                gk[iz,iz]=1.0
    # solution of simultaneous linear equations
    #disg = np.linalg.solve(gk, fp)
    gk = csr_matrix(gk)
    disg = spsolve(gk, fp, use_umfpack=True)
    # 拘束変位、回転を再代入
    for i in range(0,npoin):
        for j in range(0,nfree):
            if mpfix[j,i]==1:
                iz=i*nfree+j
                disg[iz]=rdis[j,i]
    # print out of result
    print(disg)
    # information
    dtime=time.time()-start
    print('n={0}  time={1:.3f}'.format(nfree*npoin,dtime)+' sec')
    fout=open(fnameW,'a')
    print('n={0}  time={1:.3f}'.format(nfree*npoin,dtime)+' sec',file=fout)
    fout.close()


main()


