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
