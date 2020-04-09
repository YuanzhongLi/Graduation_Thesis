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
import scipy
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix
import sys
import time
import pickle
from PLOT_FUNCTIONS import *


def inputdata(fnameR, nod, nfree):  #データパス, 要素節点数, 自由度
    f = open(fnameR, 'r')
    text = f.readline()  # コメント
    text = f.readline()  # 1行ずつ読み込む
    text = text.strip()  # 端の空白, 改行(\n)を削除
    text = text.split()  # ' '　で別れた要素ごとの配列にする
    npoin = int(text[0])  # ノード数
    nele = int(text[1])  # 要素数
    npfix = int(text[2])  # 拘束点数
    nlod = int(text[3])  # 荷重点数
    delta_t = float(text[4])  # 微小時間
    n_t = int(text[5])  # 微小時間数
    # 配列宣言
    ae = np.zeros((9, nele), dtype=np.float64)  # 要素特性
    node = np.zeros((nod + 1, nele), dtype=np.int)  # 要素構成節点
    x = np.zeros((3, npoin), dtype=np.float64)  # 座標
    mpfix = np.zeros((nfree, npoin), dtype=np.int)  # 拘束状態
    rdis = np.zeros((nfree, npoin), dtype=np.float64)  # 既知変位
    fp = np.zeros((nfree * npoin), dtype=np.float64)  # 外力
    
    # 要素特性: ae
    text = f.readline()  # コメント
    text = f.readline()
    text = text.strip().split()
    for i in range(0, nele):                
        ae[0, i] = float(text[0])  # A
        ae[1, i] = float(text[1])  # I11
        ae[2, i] = float(text[2])  # I12
        ae[3, i] = float(text[3])  # I22
        ae[4, i] = float(text[4])  # J
        ae[5, i] = float(text[5])  # E
        ae[6, i] = float(text[6])  # G
        ae[7, i] = float(text[7])  # density
    
    # 要素構成節点: node
    text = f.readline()  # コメント   
    for i in range(0, nele):
        text = f.readline()
        text = text.strip().split()
        node[0, i] = int(text[0])  #node_1
        node[1, i] = int(text[1])  #node_2
        node[2, i] = int(text[2])  #要素番号            

    # 座標: x
    text = f.readline()  # コメント    
    for i in range(0, npoin):
        text = f.readline()
        text = text.strip().split()
        x[0, i] = float(text[0])  # x-座標
        x[1, i] = float(text[1])  # y-座標
        x[2, i] = float(text[2])  # z-座標
    
    # 要素質量を計算
    for i in range(0, nele):
        node1_x = x[0, i]
        node1_y = x[1, i]
        node1_z = x[2, i]
        node2_x = x[0, i + 1]
        node2_y = x[1, i+1]
        node2_z = x[2, i+1]
        A = ae[0, i]
        density = ae[7, i]
        ae[8, i] = abs(node1_x - node2_x) * A * density        

    # 境界条件（拘束状態） (0:free, 1:restricted)
    text = f.readline()  # コメント    
    for i in range(0, npfix):
        text = f.readline()
        text = text.strip()
        text = text.split()
        lp = int(text[0])  # 固定されたノード番号
        mpfix[0, lp - 1] = int(text[1])  # x方向固定
        mpfix[1, lp - 1] = int(text[2])  # y方向固定
        mpfix[2, lp - 1] = int(text[3])  # z方向固定
        mpfix[3, lp - 1] = int(text[4])  # x軸回転固定
        mpfix[4, lp - 1] = int(text[5])  # y軸回転固定
        mpfix[5, lp - 1] = int(text[6])  # z軸回転固定
        rdis[0, lp - 1] = float(text[7])  # x方向既知変位
        rdis[1, lp - 1] = float(text[8])  # y方向既知変位
        rdis[2, lp - 1] = float(text[9])  # z方向既知変位
        rdis[3, lp - 1] = float(text[10])  # x軸既知回転量
        rdis[4, lp - 1] = float(text[11])  # y軸既知回転量
        rdis[5, lp - 1] = float(text[12])  # z軸既知回転量

    # 荷重
    text = f.readline()  # コメント    
    for i in range(0, nlod):
        text = f.readline()
        text = text.strip().split()
        lp = int(text[0])
        fp[6 * lp - 6] = float(text[1])  # x方向荷重
        fp[6 * lp - 5] = float(text[2])  # y方向荷重
        fp[6 * lp - 4] = float(text[3])  # z方向荷重
        fp[6 * lp - 3] = float(text[4])  # x軸モーメント
        fp[6 * lp - 2] = float(text[5])  # y軸モーメント
        fp[6 * lp - 1] = float(text[6])  # z軸モーメント
    f.close()
    return npoin, nele, npfix, nlod, delta_t, n_t, ae, node, x, mpfix, rdis, fp


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
    for i in range(0, len(mass) - 1):
        node_mass = (mass[i] + mass[i+1]) / 2.0
        for j in range(3):
            idx = i*nfree + j
            ret[idx,idx] = node_mass
    return ret


def dumping_3dfrm(gamma, omega, mass_mat, gk):
    m = gamma * mass_mat
    gk = omega * gk    
    for i in range(0, len(gk)):
        gk[i, i] += m[i, i]        
    return gk


# newmark マトリックスを作成
# newmarkのparam: alpha, beta
def newmark_3dfrm(delta_t, alpha, beta, gk, mass_mat, c_mat): 
    return (1.0/beta/delta_t**2) * mass_mat  \
         + (alpha/beta/delta_t) * c_mat \
         + gk


def main_3d_NEWMARK_FEM(file_path):
    start=time.time()
    args = sys.argv
    fnameR=args[1]
    fnameW=args[2]
    nod=2
    nfree=6
    alpha=0.5  # newmark param
    beta=0.25  # newmark param
    gamma=0.01 # dumping param
    omega=0.01 # dumping param
    npoin,nele,npfix,nlod,delta_t,n_t,ae,node,x,mpfix,rdis,fp=inputdata(file_path,nod,nfree)
    mass_mat=mass_3dfrm(ae[8], npoin, nfree)
    # メモリを32 * nfree * npointに抑える
    acc=np.zeros((32, nfree*npoin), dtype=np.float64)
    vec=np.zeros((32, nfree*npoin), dtype=np.float64)
    dis=np.zeros((32, nfree*npoin), dtype=np.float64)        
    z  =np.zeros(n_t+1, dtype=np.float64)

    ir=np.zeros(nod*nfree, dtype=np.int) 
    gk=np.zeros((nfree*npoin, nfree*npoin), dtype=np.float64) # Global stifness matrix

    # assembly stifness matrix & load vector
    for ne in range(0, nele):
        i=node[0,ne]-1
        j=node[1,ne]-1
        m=node[2,ne]-1
        x1=x[0,i]; y1=x[1,i]; z1=x[2,i]
        x2=x[0,j]; y2=x[1,j]; z2=x[2,j]
        A   =ae[0,m]  
        I11 =ae[1,m]  
        I12 =ae[2,m]
        I22 =ae[3,m]
        J   =ae[4,m]  
        E   =ae[5,m] 
        G   =ae[6,m] 
        EA=E*A
        GJ=G*J
        EIy=E*I11
        EIz=E*I22
        ek=sm_3dfrm(EA,GJ,EIy,EIz,x1,y1,z1,x2,y2,z2) # local stiffness matrix                                
        ir[11]=6*j+5; ir[10]=ir[11]-1; ir[9]=ir[10]-1; ir[8]=ir[9]-1; ir[7]=ir[8]-1; ir[6]=ir[7]-1
        ir[5] =6*i+5; ir[4] =ir[5]-1 ; ir[3]=ir[4]-1 ; ir[2]=ir[3]-1; ir[1]=ir[2]-1; ir[0]=ir[1]-1                
        # assemble
        for i in range(0, nod*nfree):
            it=ir[i]
            for j in range(0, nod*nfree):
                jt=ir[j]
                gk[it, jt] = gk[it, jt] + ek[i,j]               

    c_mat=dumping_3dfrm(gamma, omega, mass_mat, gk)
    newmark_mat=newmark_3dfrm(delta_t, alpha, beta, gk, mass_mat, c_mat)  
        
    for step in range(1, n_t+1):
        cur_32step = (step % 32)
        prev_32step = cur_32step-1
        if (prev_32step == -1):
            prev_32step = 31                                                            

        # fpを整理
        tmp_for_mass = (1.0/beta/delta_t**2)  * dis[prev_32step] \
                     + (1.0/beta/delta_t)     * vec[prev_32step] \
                     + ((1.0/2/beta)-1.0)     * acc[prev_32step]

        tmp_for_c = (alpha/beta/delta_t)            * dis[prev_32step] \
                  + ((alpha/beta)-1.0)              * vec[prev_32step] \
                  + delta_t * ((alpha/2/beta)-1.0)  * acc[prev_32step]                       

        Fp = fp + np.dot(mass_mat, tmp_for_mass) + np.dot(c_mat, tmp_for_c)            

        # boudary conditions
        for i in range(0, npoin):
            for j in range(0, nfree):
                if mpfix[j, i] == 1:                        
                    iz=i*nfree+j
                    Fp[iz]=0.0

        for i in range (0, npoin):
            for j in range(0, nfree):
                if mpfix[j, i] == 1:
                    iz=i*nfree+j
                    newmark_mat[:,iz]=0.0
                    newmark_mat[iz,iz]=1.0
        # 疎行列圧縮格納
        sp_newmark_mat = csr_matrix(newmark_mat)
        dis[cur_32step] = spsolve(sp_newmark_mat, Fp, use_umfpack=True)

        # 拘束条件を再代入する
        for i in range(0, npoin):
            for j in range(0, nfree):
                if mpfix[j, i] == 1:
                    iz=i*nfree+j
                    dis[cur_32step, iz] = rdis[j, i]

        # 速度, 加速度計算            
        acc[cur_32step] = (1.0/beta/delta_t**2)  * (dis[cur_32step] - dis[prev_32step]) \
                        - (1.0/beta/delta_t)     * vec[prev_32step] \
                        - ((1.0/2/beta)-1.0)     * acc[prev_32step]

        vec[cur_32step] = (alpha/beta/delta_t)         * (dis[cur_32step] - dis[prev_32step]) \
                        + (1.0-(alpha/beta))           * vec[prev_32step] \
                        + delta_t*(1.0-(alpha/2/beta)) * acc[prev_32step] 

        z[step] = dis[cur_32step, 22 * 6 + 2]        
        if (step % 100 == 0):                
            step_time = time.time()-start
            print('{0} step: {1:.3f}'.format(step, step_time))
    # print out result
    dtime=time.time()-start
    print('time: {0:.3f}'.format(dtime)+'sec')
    # 荷重方向のみ返している
    return z


y_dis = main_3d_NEWMARK_FEM('test1000step.txt')

plot_dis(y_dis)

# 基準となる100,000 step
y_dis2 = main_3d_NEWMARK_FEM('test100000step.txt')

y_dis2_formatted = format_y_dis(y_dis2, 100)

# 10,000 step
y_dis3 = main_3d_NEWMARK_FEM('test10000step.txt')

y_dis3_formatted = format_y_dis(y_dis3, 10)

# 2,000 step
y_dis4 = main_3d_NEWMARK_FEM('test2000step.txt')

y_dis4_formatted = format_y_dis(y_dis4, 2)

# 5,000 step
y_dis5 = main_3d_NEWMARK_FEM('test5000step.txt')

y_dis5_formatted = format_y_dis(y_dis5, 5)

data = {
    'step1000': y_dis,
    'step2000': y_dis4_formatted,
    'step5000': y_dis5_formatted,
    'step10000': y_dis3_formatted,
    'step100000': y_dis2_formatted,    
}

# dataを保存
with open ('newmark精度テストデータ001', 'wb') as f:
    pickle.dump(data, f)
