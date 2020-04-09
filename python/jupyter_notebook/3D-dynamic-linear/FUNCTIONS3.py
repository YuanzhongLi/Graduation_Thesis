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
import matplotlib.pyplot as plt


# +
# 周期関連の関数
def plot_period(test_dis, save=False, title='Period'):
    step_num = test_dis.shape[0]
    nfree = 6
    node_num = test_dis.shape[1] // 6
    step = np.zeros(step_num, dtype=np.int)
    for i in range(step_num):
        step[i] = i
        
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(step, test_dis.T[(node_num-1)*6+1], label='y displacement')
    plt.title('Period', fontsize=24)
    plt.xlabel('time [1e-3 sec]', fontsize=24)
    plt.ylabel('elastic displacement [m]', fontsize=24)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=24)
    plt.tick_params(labelsize=18)
    if save:
        plt.savefig('{0}.png'.format(title))
    plt.show()
    
def calc_period_from_dis(test_dis, delta_t=0.001):
    step_num = test_dis.shape[0]
    nfree = 6
    node_num = test_dis.shape[1] // 6
    cnt = 0
    for index, y_dis in enumerate(test_dis.T[(node_num-1)*6+1]):
        if y_dis * test_dis.T[(node_num-1)*6+1][index+1] < 0:
            cnt += 1
            if (cnt == 2):
                return index*delta_t
            
def calc_period_from_kL_mass_mat(kL, mass_mat):
    kL_eig_val, kL_eig_vec = np.linalg.eig(kL)
    eig_val, eig_vec = scipy.linalg.eig(kL[6:, 6:], mass_mat[6:, 6:])
    w = sorted(eig_val)[0]
    T = 2 * np.pi / np.sqrt(w)
    return T


# -

# abaqusの解析結果を取得
def get_abaqus_data(file_path, step, node_num):
    ret = np.zeros((step, node_num*6), dtype=np.float64)
    with open(file_path, 'r') as f:
        for i in range(step):
            line = '-'
            while (len(line) < 20 or line[0] != '-'):
                line = f.readline()
            for j in range(node_num):
                l = f.readline().strip().split()
                for index,ele in enumerate(l):
                    if index > 0:
                        ret[i, j*6+index-1] = float(ele)
    return ret
# get_abaqus_data('3d-beam-rot-onlyU.txt', 1000, 11)


# +
# 計算結果を確認、図示するための関数
def print_dis(dis, ep=1):
    for i in range(len(dis)):
        if i % ep == 0:
            print(i)    
            print('ux uy uz rux ruy ruz')
            for j in range(11):                
                print(dis[i][j*6], dis[i][j*6+1], dis[i][j*6+2], dis[i][j*6+3], dis[i][j*6+4], dis[i][j*6+5])
            print('-----')
            
def plot_dis(e_disG, node_id=None, pos=None, save=False, title='displacement'):    
    step_num = e_disG.shape[0]
    nfree = 6
    node_num = e_disG.shape[1] // 6
    if node_id == None:
        node_id = node_num
    if pos==None:
        pos=1
    
    step = np.zeros(step_num, dtype=np.int)
    for i in range(step_num):
        step[i] = i
        
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    labels = ['x', 'y', 'z', 'rux', 'ruy', 'ruz']
    label = labels[pos]
    ax.plot(step, e_disG.T[(node_id-1)*6+pos], label='{0} node{1}'.format(label, node_id-1))
    plt.title('{0} displacement'.format(label), fontsize=24)
    plt.xlabel('time [1e-3 sec]', fontsize=24)
    plt.ylabel('displacement', fontsize=24)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=24)
    plt.tick_params(labelsize=18)
    if save:
        plt.savefig('{0}_{1}.png'.format(title, label))        
    plt.show() 
# plot_dis(e_disL, 11)             
            
def print_coords(cords, ep=1):
    for i in range(len(cords)):
        if i % ep == 0:
            print(i)
            print('x: ', cords[i][0])
            print('y: ', cords[i][1])
            print('z: ', cords[i][2])
            
def plot_coords(cords, node_ids=None, save=False, title='coord'):
    step_num = cords.shape[0] - 1
    node_num = cords.shape[2]
    if node_ids == None:
        node_ids = [i+1 for i in range(node_num)]
    step = np.zeros(step_num, dtype=np.int)
    for i in range(step_num):
        step[i] = i    
    
    axises = ['x', 'y', 'z']
    for i in range(3): # x, y, z
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        axis = axises[i]
        for node_id in node_ids:
            data = np.zeros(step_num, dtype=np.float64)
            for k in range(step_num):
                data[k] = cords[k, i, node_id-1]
            ax.plot(step, data, label='{0} node{1}'.format(axis, node_id-1))
        plt.title('{0} displacement'.format(axis), fontsize=24)
        plt.xlabel('time [1e-3 sec]', fontsize=24)
        plt.ylabel('coordinate', fontsize=24)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=24)
        plt.tick_params(labelsize=18)
        if save:
            plt.savefig('{0}_{1}.png'.format(title, label))        
        plt.show()         
# plot_coods(coordsG_plus_e_dis, [11])
