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

from scipy.spatial.transform import Rotation
from scipy.linalg import expm
import numpy as np
from Functions import inputdata, get_abaqus_data
from pyquaternion import Quaternion


# abaqusのデータからamplitudeを作成
# v_data = get_abaqus_data('test05V.txt', 265, 23)
def make_amp(v_data, delta_t=0.001, init_v=[-0.22906634372358425, 0.07038992291148816, -0.47632976594513821, 1.56221576610329290, 2.14566704525587680, -0.07210294407980200]):
    step = v_data.shape[0]
    npoin = v_data.shape[1] // 6
    amp_labels = ['VX', 'VY', 'VZ', 'RVX', 'RVY', 'RVZ']
    for index, label in enumerate(amp_labels):
        with open('amp_{0}_{1}step_{2}point'.format(label, step, npoin), 'w', encoding='ascii') as f:
            f.write('0 {0}\n'.format(init_v[index]))
            for s in range(1, step):
                f.write('{0:.3f} {1}\n'.format(delta_t*s, v_data[s, index]))                


# 作成したampを確認する
def show(ax, ay, az):
    for i, j, k in zip(ax, ay, az):
        print('{0:.3f} {1:.5f} {2:.5f} {3:.5f}'.format(i[0], i[1], j[1], k[1]))


def make_O(wx, wy, wz):
    return np.array([[0, wz, -wy, wx],
                           [-wz, 0, wx, wy],
                           [wy, -wx, 0, wz],
                           [-wx, -wy, -wz, 0]], dtype=np.float64) 


# +
def norm2(v):
    return np.sqrt(np.sum(v**2))

def get_q0(originCoordsG):
    npoin = originCoordsG.shape[1]
    a = np.array([originCoordsG[0, npoin-1], originCoordsG[1, npoin-1], originCoordsG[2, npoin-1]], 
                dtype=np.float64)
    b = a / norm2(a)
    x, y, z = b[0], b[1], b[2]
    
    theta1 = -np.arctan2(y, x)
    tmp_q1 = Quaternion(axis=np.array([0, 0, 1]), angle=theta1)
    
    c = tmp_q1.rotate(b)
    
    x1, y1, z1 = c[0], c[1], c[2]
    theta2 = np.arctan2(z1, x1)
    tmp_q2 = Quaternion(axis=np.array([0, 1, 0]), angle=theta2)
        
    q0 = tmp_q2 * tmp_q1    
    return q0


# -

def make_rig_RA_RU(VX, VY, VZ, RVX, RVY, RVZ, delta_t ,originCoordsG):    
    n_t = VX.shape[0] - 1
    npoin = originCoordsG.shape[1]
    # 原点情報    
    UX = np.zeros((n_t+1, 2), dtype=np.float64)
    UY = np.zeros((n_t+1, 2), dtype=np.float64)
    UZ = np.zeros((n_t+1, 2), dtype=np.float64)    
    AX = np.zeros((n_t+1, 2), dtype=np.float64)
    AY = np.zeros((n_t+1, 2), dtype=np.float64)    
    AZ = np.zeros((n_t+1, 2), dtype=np.float64)           
    
    RAX = np.zeros((n_t+1, 2), dtype=np.float64)
    RAY = np.zeros((n_t+1, 2), dtype=np.float64)    
    RAZ = np.zeros((n_t+1, 2), dtype=np.float64)

    for i in range(1, n_t+1):
        # 時間設定
        UX[i, 0] = VX[i, 0]
        AX[i, 0] = VX[i, 0]            
        UY[i, 0] = VY[i, 0]
        AY[i, 0] = VY[i, 0]
        UZ[i, 0] = VZ[i, 0]
        AZ[i, 0] = VZ[i, 0]
        
        RAX[i, 0] = RVX[i, 0]
        RAY[i, 0] = RVY[i, 0]
        RAZ[i, 0] = RVZ[i, 0]
        
        if i > 1:
            UX[i, 1] = (VX[i, 1] + VX[i-1, 1])/2 * delta_t + UX[i-1, 1]
            UY[i, 1] = (VY[i, 1] + VY[i-1, 1])/2 * delta_t + UY[i-1, 1] 
            UZ[i, 1] = (VZ[i, 1] + VZ[i-1, 1])/2 * delta_t + UZ[i-1, 1]
                        
            if i < n_t:
                AX[i, 1] = (VX[i+1, 1]-VX[i-1, 1])/2/delta_t
                AY[i, 1] = (VY[i+1, 1]-VY[i-1, 1])/2/delta_t  
                AZ[i, 1] = (VZ[i+1, 1]-VZ[i-1, 1])/2/delta_t                
                
                RAX[i, 1] = (RVX[i+1, 1]-RVX[i-1, 1])/2/delta_t
                RAY[i, 1] = (RVY[i+1, 1]-RVY[i-1, 1])/2/delta_t                 
                RAZ[i, 1] = (RVZ[i+1, 1]-RVZ[i-1, 1])/2/delta_t 
                
    # 初期設定
    AX[1, 1] = AX[2, 1]
    AY[1, 1] = AY[2, 1]
    AZ[1, 1] = AZ[2, 1]
    
    RAX[1, 1] = RAX[2, 1]
    RAY[1, 1] = RAY[2, 1]
    RAZ[1, 1] = RAZ[2, 1]
    
    tmp = np.array([originCoordsG[0, npoin-1], originCoordsG[1, npoin-1], originCoordsG[2, npoin-1]], dtype=np.float64)
    originE =  (1/np.sum(np.sqrt(tmp**2)))*tmp
    q0 = Quaternion(np.array([1, 0, 0, 0]))
    qs = [q0]
    total_qs = [q0]
    
    for step in range(1, n_t+1): # step-1からstepを作る
        wx = RVX[step, 1]
        wy = RVY[step ,1]
        wz = RVZ[step, 1]
        
        q_prev = total_qs[step-1]
        q_next = Quaternion()    
        q_next.integrate(np.array([wx, wy, wz], dtype=np.float64), delta_t)
    
        qs.append(q_next)
        total_qs.append(q_next*q_prev)
        
    return UX, UY, UZ, AX, AY, AZ, RAX, RAY, RAZ, qs, total_qs    


def make_expanded_amp(V_array, rate):    
    labels = ['VX', 'VY', 'VZ', 'RVX', 'RVY', 'RVZ']
    for V, label in zip(V_array, labels):
        delta_t = V[2, 0]
        size = len(V)-1
        expanded_amp = np.zeros((size*rate+1, 2), dtype=np.float64)
        for i in range(size):
            for idx, j in enumerate(range(i*rate+1, (i+1)*rate+1)):                
                expanded_amp[j-1, 0] = (j-1) * delta_t
                expanded_amp[j-1, 1] = (V[i,1] + (V[i+1, 1]-V[i,1])/rate*(idx+1))/rate

        with open('amp_{0}_expand{1}.txt'.format(label, rate), 'w', encoding='ascii') as f:
            for amp in expanded_amp[:-1]:
                if i < len(expanded_amp[:-1])-1:
                    f.write('{0:.5f} {1}\n'.format(amp[0], amp[1]))
                else:
                    f.write('{0:.5f} {1}'.format(amp[0], amp[1]))

# +
# make_expanded_amp([VX, VY, VZ, RVX, RVY, RVZ], 10)
# make_expanded_amp([VX, VY, VZ, RVX, RVY, RVZ], 5)
# make_expanded_amp([VX, VY, VZ, RVX, RVY, RVZ], 3)
# make_expanded_amp([VX, VY, VZ, RVX, RVY, RVZ], 2)
# +
# show(RVX, RVY, RVZ)
