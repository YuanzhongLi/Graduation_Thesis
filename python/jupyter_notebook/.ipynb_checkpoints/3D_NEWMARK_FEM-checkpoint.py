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
    text=f.readline() # 1行ずつ読み込む
    text=text.strip() # 端の空白, 改行(\n)を削除
    text=text.split() # ','　で別れた要素ごとの配列にする
    npoin=int(text[0]) # ノード数
    nele =int(text[1]) # 要素数
    npfix=int(text[2]) # 拘束点数
    nlod =int(text[3]) # 荷重点数
    delta_t =float(text[4]) # 微小時間
    n_t =int(text[5]) # 微小時間数
    # 配列宣言
    ae    =np.zeros((10,nele),dtype=np.float64)       # 要素特性
    node  =np.zeros((nod+1,nele),dtype=np.int)       # 要素構成節点
    x     =np.zeros((3,npoin),dtype=np.float64)      # 座標
    mpfix =np.zeros((nfree,npoin),dtype=np.int)      # 拘束状態
    rdis  =np.zeros((nfree,npoin),dtype=np.float64)  # 既知変位
    fp    =np.zeros(nfree*npoin,dtype=np.float64)    # 外力
    # 要素特性
    for i in range(0,nele):
        text=f.readline()
        text=text.strip()
        text=text.split()
        ae[0,i] =float(text[0])  # E
        ae[1,i] =float(text[1])  # Po
        ae[2,i] =float(text[2])  # A 
        ae[3,i] =float(text[3])  # Ix
        ae[4,i] =float(text[4])  # Iy
        ae[5,i] =float(text[5])  # Iz
        ae[6,i] =float(text[6])  # m
        ae[7,i] =float(text[7])  # gkx
        ae[8,i] =float(text[8])  # gky
        ae[9,i] =float(text[9])  # gkz

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
        x[1,i]=float(text[1])    # y-座標
        x[2,i]=float(text[2])    # z-座標
        
    # 境界条件（拘束状態） (0:free, 1:restricted)
    for i in range(0,npfix):
        text=f.readline()
        text=text.strip()
        text=text.split()
        lp=int(text[0])              # 固定されたノード番号
        mpfix[0,lp-1]=int(text[1])   # x方向固定
        mpfix[1,lp-1]=int(text[2])   # y方向固定
        mpfix[2,lp-1]=int(text[3])   # z方向固定
        mpfix[3,lp-1]=int(text[4])   # x軸回転固定
        mpfix[4,lp-1]=int(text[5])   # y軸回転固定
        mpfix[5,lp-1]=int(text[6])   # z軸回転固定
        rdis[0,lp-1]=float(text[7])  # x方向既知変位
        rdis[1,lp-1]=float(text[8])  # y方向既知変位
        rdis[2,lp-1]=float(text[9])  # z方向既知変位
        rdis[3,lp-1]=float(text[10])  # x軸既知回転量
        rdis[4,lp-1]=float(text[11])  # y軸既知回転量
        rdis[5,lp-1]=float(text[12])  # z軸既知回転量                

    # 荷重
    if (nlod > 0):
        for i in range(0,nlod):
            text=f.readline()
            text=text.strip()
            text=text.split()
            lp=int(text[0])           
            fp[6*lp-6]=float(text[1]) # x方向荷重
            fp[6*lp-5]=float(text[2]) # y方向荷重
            fp[6*lp-4]=float(text[3]) # z方向荷重
            fp[6*lp-3]=float(text[4]) # x軸モーメント
            fp[6*lp-2]=float(text[5]) # y軸モーメント
            fp[6*lp-1]=float(text[6]) # z軸モーメント
    f.close()
    return npoin,nele,npfix,nlod,ae,node,x,mpfix,rdis,fp


# 要素剛性マトリックス作成
def sm_3dfrm(EA,GJ,EIy,EIz,x1,y1,z1,x2,y2,z2):
    ek=np.zeros((12,12),dtype=np.float64) # local stiffness matrix
    xx=x2-x1
    yy=y2-y1
    zz=z2-z1
    el=np.sqrt(xx**2+yy**2+zz**2)
    ek[ 0, 0]= EA/el
    ek[ 0, 6]=-EA/el
    ek[ 1, 1]= 12*EIz/el**3
    ek[ 1, 5]=  6*EIz/el**2
    ek[ 1, 7]=-12*EIz/el**3
    ek[ 1,11]=  6*EIz/el**2
    ek[ 2, 2]= 12*EIy/el**3
    ek[ 2, 4]= -6*EIy/el**2
    ek[ 2, 8]=-12*EIy/el**3
    ek[ 2,10]= -6*EIy/el**2
    ek[ 3, 3]= GJ/el
    ek[ 3, 9]=-GJ/el
    ek[ 4, 2]= -6*EIy/el**2
    ek[ 4, 4]=  4*EIy/el
    ek[ 4, 8]=  6*EIy/el**2
    ek[ 4,10]=  2*EIy/el
    ek[ 5, 1]=  6*EIz/el**2
    ek[ 5, 5]=  4*EIz/el
    ek[ 5, 7]= -6*EIz/el**2
    ek[ 5,11]=  2*EIz/el
    ek[ 6, 0]=-EA/el
    ek[ 6, 6]= EA/el
    ek[ 7, 1]=-12*EIz/el**3
    ek[ 7, 5]= -6*EIz/el**2
    ek[ 7, 7]= 12*EIz/el**3
    ek[ 7,11]= -6*EIz/el**2
    ek[ 8, 2]=-12*EIy/el**3
    ek[ 8, 4]=  6*EIy/el**2
    ek[ 8, 8]= 12*EIy/el**3
    ek[ 8,10]=  6*EIy/el**2
    ek[ 9, 3]=-GJ/el
    ek[ 9, 9]= GJ/el
    ek[10, 2]= -6*EIy/el**2
    ek[10, 4]=  2*EIy/el
    ek[10, 8]=  6*EIy/el**2
    ek[10,10]=  4*EIy/el
    ek[11, 1]=  6*EIz/el**2
    ek[11, 5]=  2*EIz/el
    ek[11, 7]= -6*EIz/el**2
    ek[11,11]=  4*EIz/el
    return ek


# 座標変換マトリックス作成
def tm_3dfrm(theta,x1,y1,z1,x2,y2,z2):
    tt=np.zeros((12,12),dtype=np.float64) # transformation matrix
    t1=np.zeros((3,3),dtype=np.float64)
    t2=np.zeros((3,3),dtype=np.float64)
    xx=x2-x1
    yy=y2-y1
    zz=z2-z1
    el=np.sqrt(xx**2+yy**2+zz**2)
    t1[0,0]=1
    t1[1,1]= np.cos(theta)
    t1[1,2]= np.sin(theta)
    t1[2,1]=-np.sin(theta)
    t1[2,2]= np.cos(theta)
    ll=(x2-x1)/el
    mm=(y2-y1)/el
    nn=(z2-z1)/el
    if x2-x1==0.0 and y2-y1==0.0:
        t2[0,2]=nn
        t2[1,0]=nn
        t2[2,1]=1.0
    else:
        qq=np.sqrt(ll**2+mm**2)
        t2[0,0]=ll
        t2[0,1]=mm
        t2[0,2]=nn
        t2[1,0]=-mm/qq
        t2[1,1]= ll/qq
        t2[2,0]=-ll*nn/qq
        t2[2,1]=-mm*nn/qq
        t2[2,2]=qq
    t3=np.dot(t1,t2)
    tt[0:3,0:3]  =t3[0:3,0:3]
    tt[3:6,3:6]  =t3[0:3,0:3]
    tt[6:9,6:9]  =t3[0:3,0:3]
    tt[9:12,9:12]=t3[0:3,0:3]
    return tt

