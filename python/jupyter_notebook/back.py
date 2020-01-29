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
import scipy
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import sys
import time
import matplotlib.pyplot as plt
import pickle


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
        #         node1_y = x[1, i]
        #         node1_z = x[2, i]
        node2_x = x[0, i + 1]
        #         node2_y = x[1, i+1]
        #         node2_z = x[2, i+1]
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


npoin,nele,npfix,nlod,delta_t,n_t,ae,node,x,mpfix,rdis,fp=inputdata('test_verification2.txt',2, 6)


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


def back_3dfrm(delta_t, gk, mass_mat, c_mat):
    return delta_t*gk + (1.0/delta_t)*mass_mat + c_mat


def calc_period_from_kL_mass_mat(kL, mass_mat):
    kL_eig_val, kL_eig_vec = np.linalg.eig(kL)
#     print(sorted(kL_eig_val)[:7])
    eig_val, eig_vec = scipy.linalg.eig(kL[6:, 6:], mass_mat[6:, 6:])
#     print(sorted(eig_val))
    w = sorted(eig_val)[0]
    T = 2 * np.pi / np.sqrt(w)
    return T


def main_3d_back(file_path):
    start=time.time()
    args = sys.argv
    fnameR=args[1]
    fnameW=args[2]
    nod=2
    nfree=6
    alpha=0.5 # newmark param
    beta=0.25 # newmark param
    gamma=0.01 # dumping param
    omega=0.01 # dumping param
    npoin,nele,npfix,nlod,delta_t,n_t,ae,node,x,mpfix,rdis,fp=inputdata(file_path,nod,nfree)
    mass_mat=mass_3dfrm(ae[8], npoin, nfree)
    vec=np.zeros((n_t+1, nfree*npoin), dtype=np.float64)
    dis=np.zeros((n_t+1, nfree*npoin), dtype=np.float64)
    
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
    back_mat = back_3dfrm(delta_t, gk, mass_mat, c_mat) 

    for step in range(1, n_t+1): 
        Fp = np.dot((1.0/delta_t)*mass_mat+c_mat, dis[step-1]) \
            + np.dot(mass_mat, vec[step-1]) \
            + delta_t*fp
#         if step == 1:
#             Fp = np.dot((1.0/delta_t)*mass_mat+c_mat, dis[step-1]) \
#                 + np.dot(mass_mat, vec[step-1]) \
#                 + delta_t*fp
#         else:
#             Fp = np.dot((1.0/delta_t)*mass_mat+c_mat, dis[step-1]) \
#                 + np.dot(mass_mat, vec[step-1])             
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
                    back_mat[:,iz]=0.0
                    back_mat[iz,iz]=1.0
        # 疎行列圧縮格納        
        sp_back_mat = csr_matrix(back_mat)
        dis[step] = spsolve(back_mat, Fp, use_umfpack=True)

        # 拘束条件を再代入する
        for i in range(0, npoin):
            for j in range(0, nfree):
                if mpfix[j, i] == 1:
                    iz=i*nfree+j
                    dis[step, iz] = rdis[j, i]
        vec[step] = (dis[step] - dis[step-1]) / delta_t
        if (step % 100 == 0):                
            step_time = time.time()-start
            print('{0} step: {1:.3f}'.format(step, step_time))
    # print out result
    dtime=time.time()-start
    print('time: {0:.3f}'.format(dtime)+'sec')
    return dis, vec, gk, mass_mat


dis, vec, gk, mass_mat = main_3d_back('test_verification2.txt')

calc_period_from_kL_mass_mat(gk, mass_mat)


def get_y_dis(dis):
    n_t = dis.shape[0]
    npoin = dis.shape[1] // 6
    ret = np.empty((n_t), dtype=np.float64)
    for i in range(n_t):
        ret[i] = dis[i, (npoin-1) * 6 + 2]
    return ret


def plot_dis(dis, save=False):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1,1, 1)
    ax.plot(dis, label='y elastic displacement', color='blue')
    plt.title('backward Euler method')
    plt.xlabel('step', fontsize=18)
    plt.ylabel('elastic displacement', fontsize=18)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=14)
    if save:        
        plt.savefig('back_verif.png')
    plt.show()


def calc_dis(F, l, E, I):
    return F*l**3/(3*E*I)


calc_dis(1, 0.9929200000000,1.000e+011, 1e-10)

y_dis = get_y_dis(dis)

dis.shape

plot_dis(y_dis2_formatted, save=True)

dis2, _, _, _ = main_3d_back('test_verification2.txt')

y_dis2 = get_y_dis(dis2)


def format_y_dis(y_dis, b):
    ret = np.zeros((1001), dtype=np.float64)
    for i in range(1, 1001):
        ret[i] = y_dis[i*b]
    return ret        


y_dis2_formatted = format_y_dis(y_dis2, 100)

dis3, _, _, _ = main_3d_back('test_verification2.txt')

y_dis3 = get_y_dis(dis3)

y_dis3_formatted = format_y_dis(y_dis3, 10)

dis4, _, _, _ = main_3d_back('test_verification2.txt')

y_dis4 = get_y_dis(dis4)

y_dis4_formatted = format_y_dis(y_dis4, 2)

dis5, _, _, _ = main_3d_back('test_verification2.txt')

y_dis5 = get_y_dis(dis5)

y_dis5_formatted = format_y_dis(y_dis5, 5)

data = {
    'step1000': y_dis,
    'step2000': y_dis4_formatted,
    'step5000': y_dis5_formatted,
    'step10000': y_dis3_formatted,
    'step100000': y_dis2_formatted
}

with open('後退オイラー法精度テストデータ1', 'wb') as f:
    pickle.dump(data, f)


