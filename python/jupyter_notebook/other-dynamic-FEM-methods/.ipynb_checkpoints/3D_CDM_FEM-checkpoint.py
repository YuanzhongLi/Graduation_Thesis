# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
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
    mpfix =np.zeros((n_t+1, nfree,npoin),dtype=np.int)      # 拘束状態
    rdis  =np.zeros((n_t+1, nfree,npoin),dtype=np.float64)  # 既知変位
    fp    =np.zeros((n_t+1, nfree*npoin),dtype=np.float64)    # 外力
    # 要素特性: ae
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
        ae[6,i] =float(text[6])  # mass
        ae[7,i] =float(text[7])  # gkx
        ae[8,i] =float(text[8])  # gky
        ae[9,i] =float(text[9])  # gkz

    # 要素構成節点: node
    for i in range(0,nele):
        text=f.readline()
        text=text.strip()
        text=text.split()
        node[0,i]=int(text[0]) #node_1
        node[1,i]=int(text[1]) #node_2
        node[2,i]=int(text[2]) #要素番号
        
    # 座標: x
    for i in range(0,npoin):
        text=f.readline()
        text=text.strip()
        text=text.split()
        x[0,i]=float(text[0])    # x-座標
        x[1,i]=float(text[1])    # y-座標
        x[2,i]=float(text[2])    # z-座標
    
    for j in range(0, n_t+1):
        # 境界条件（拘束状態） (0:free, 1:restricted)
        for i in range(0,npfix):
            text=f.readline()
            text=text.strip()
            text=text.split()
            lp=int(text[0])              # 固定されたノード番号
            mpfix[j,0,lp-1]=int(text[1])   # x方向固定
            mpfix[j,1,lp-1]=int(text[2])   # y方向固定
            mpfix[j,2,lp-1]=int(text[3])   # z方向固定
            mpfix[j,3,lp-1]=int(text[4])   # x軸回転固定
            mpfix[j,4,lp-1]=int(text[5])   # y軸回転固定
            mpfix[j,5,lp-1]=int(text[6])   # z軸回転固定
            rdis[j,0,lp-1]=float(text[7])  # x方向既知変位
            rdis[j,1,lp-1]=float(text[8])  # y方向既知変位
            rdis[j,2,lp-1]=float(text[9])  # z方向既知変位
            rdis[j,3,lp-1]=float(text[10])  # x軸既知回転量
            rdis[j,4,lp-1]=float(text[11])  # y軸既知回転量
            rdis[j,5,lp-1]=float(text[12])  # z軸既知回転量                

        # 荷重
        for i in range(0,nlod):
            text=f.readline()
            text=text.strip()
            text=text.split()
            lp=int(text[0])           
            fp[j,6*lp-6]=float(text[1]) # x方向荷重
            fp[j,6*lp-5]=float(text[2]) # y方向荷重
            fp[j,6*lp-4]=float(text[3]) # z方向荷重
            fp[j,6*lp-3]=float(text[4]) # x軸モーメント
            fp[j,6*lp-2]=float(text[5]) # y軸モーメント
            fp[j,6*lp-1]=float(text[6]) # z軸モーメント
    f.close()
    return npoin,nele,npfix,nlod,delta_t,n_t,ae,node,x,mpfix,rdis,fp


# 要素剛性マトリックス作成（local）
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


def mass_3dfrm(ae_mass, npoin, nfree):
    # 番兵追加
    mass = np.append(ae_mass, 0.0)
    mass = np.insert(mass, 0, 0.0)   
    ret = np.zeros((npoin*nfree, npoin*nfree), dtype=np.float64)
    node_mass_array = np.zeros(npoin, dtype=np.float64)
    for i in range(0, len(mass) - 1):
        node_mass = (mass[i] + mass[i+1]) / 2.0
        node_mass_array[i] = node_mass
        for j in range(3):
            idx = i*nfree + j
            ret[idx,idx] = node_mass
    return ret, node_mass_array


# gamma: mass, omega: k
def dumping_3dfrm(gamma, omega, mass_mat, k):
    m = gamma * mass_mat
    ret = omega * k    
    for i in range(0, len(ret)):
        ret[i, i] += m[i, i]        
    return ret


# CDM マトリックスを作成
def cdm_3dfrm(delta_t, mass_mat, c_mat):
    c_mat = (1.0/(2*delta_t))*c_mat
    for i in range(0, len(c_mat)):
        c_mat[i, i] += (1.0/(delta_t*delta_t))*mass_mat[i, i]        
    return c_mat


def main_3d_CDM_FEM(file_path):
        start=time.time()
        args = sys.argv
        fnameR=args[1]
        fnameW=args[2]
        nod=2
        nfree=6        
        gamma=0.0 # dumping param
        omega=0.0 # dumping param
        npoin,nele,npfix,nlod,delta_t,n_t,ae,node,x,mpfix,rdis,fp=inputdata(file_path, nod,nfree)
        mass_mat=mass_3dfrm(ae[6], npoin, nfree)        
        acc=np.zeros((n_t+1, nfree*npoin), dtype=np.float64)
        vec=np.zeros((n_t+1, nfree*npoin), dtype=np.float64)
        dis=np.zeros((n_t+1, nfree*npoin), dtype=np.float64)
        # U-1
        dis_negative1=dis[0] - delta_t*vec[0] + (delta_t*delta_t/2)*acc[0]        
        for step in range(1, n_t+1):
            ir=np.zeros(nod*nfree, dtype=np.int) 
            gk=np.zeros((nfree*npoin, nfree*npoin), dtype=np.float64) # Global stifness matrix
            
            # assembly stifness matrix & load vector
            for ne in range(0, nele):
                i=node[0,ne]-1
                j=node[1,ne]-1
                m=node[2,ne]-1
                x1=x[0,i]; y1=x[1,i]; z1=x[2,i]
                x2=x[0,j]; y2=x[1,j]; z2=x[2,j]
                ee   =ae[0,m]  # elastic modulus
                po   =ae[1,m]  # Poisson's ratio
                aa   =ae[2,m]  # section area
                aix  =ae[3,m] # tortional constant
                aiy  =ae[4,m] # moment of inertia around y-axis
                aiz  =ae[5,m] # moment of inertia around z-axis
                mass =ae[6,m] # unit weight of material
                gkX  =ae[7,m]   # seismic coefficient in X-direction
                gkY  =ae[8,m]  # seismic coefficient in Y-direction
                gkZ  =ae[9,m]  # seismic coefficient in Z-direction
                A=aa  # section area
                EA=ee*aa
                GJ=ee/2/(1+po)*aix
                EIy=ee*aiy
                EIz=ee*aiz
                ek   =sm_3dfrm(EA,GJ,EIy,EIz,x1,y1,z1,x2,y2,z2) # local Stiffness matrix                       
                ir[11]=6*j+5; ir[10]=ir[11]-1; ir[9]=ir[10]-1; ir[8]=ir[9]-1; ir[7]=ir[8]-1; ir[6]=ir[7]-1
                ir[5] =6*i+5; ir[4] =ir[5]-1 ; ir[3]=ir[4]-1 ; ir[2]=ir[3]-1; ir[1]=ir[2]-1; ir[0]=ir[1]-1                
                for i in range(0, nod*nfree):
                    it=ir[i]
                    for j in range(0, nod*nfree):
                        jt=ir[j]
                        gk[it, jt] = gk[it, jt] + ek[i,j]            
            c_mat=dumping_3dfrm(gamma, omega, gk, mass_mat)
            cdm_mat=cdm_3dfrm(delta_t, mass_mat, c_mat)
            # fpを整理
            tmp1 = gk
            for i in range(0, len(gk)):
                tmp1[i,i] -= (2.0/(delta_t**2))*mass_mat[i,i]
                
            tmp2 = (-1)*(1.0/(2*delta_t))*c_mat
            for i in range(0, len(c_mat)):
                tmp2[i,i] += (1.0/(delta_t**2))*mass_mat[i,i]
                
            if step==1:
                fp[step] = fp[step] - np.dot(tmp1, dis[step-1]) - np.dot(tmp2, dis_negative1)
            else:
                fp[step] = fp[step] - np.dot(tmp1, dis[step-1]) - np.dot(tmp2, dis[step-2])                
            
            # boudary conditions
            for i in range(0, npoin):
                for j in range(0, nfree):
                    if mpfix[step, j, i] == 1:
                        iz=i*nfree+j
                        fp[step,iz]=0.0

            for i in range (0, npoin):
                for j in range(0, nfree):
                    if mpfix[step, j, i] == 1:
                        iz=i*nfree+j
                        cdm_mat[:,iz]=0.0
                        cdm_mat[iz,iz]=1.0
                        
            # 疎行列圧縮格納
            sp_cdm_mat = csr_matrix(cdm_mat)
            dis[step] = spsolve(sp_cdm_mat, fp[step], use_umfpack=True)
                        
            # 拘束条件を再代入する
            for i in range(0, npoin):
                for j in range(0, nfree):
                    if mpfix[step, j, i] == 1:
                        iz=i*nfree+j
                        dis[step, iz] = rdis[step, j, i]   
        
        # 速度, 加速度計算    
        if (step==1):
            acc[step-1] = (1.0/(delta_t**2))*(dis_negative1 - 2*dis[step-1] - dis[step])
            vec[step-1] = (1.0/(2*delta_t))*(dis[step]-dis_negative1)
        else:
            acc[step-1] = (1.0/(delta_t**2))*(dis[step-2] - 2*dis[step-1] - dis[step])
            vec[step-1] = (1.0/(2*delta_t))*(dis[step]-dis[step-2])
                        
        dtime=time.time()-start
        print('time: {0:.3f}'.format(dtime)+'sec')
        return dis
