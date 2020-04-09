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
# 要素合成マトリックスと質量マトリックスから周期をみるための関数群
# test_NEWMARK_FEMの結果をプロットする
def plot_period(test_dis, save=False, title='Period'):
    step_num = test_dis.shape[0]
    nfree = 6
    node_num = test_dis.shape[1] // 6
    step = np.zeros(step_num, dtype=np.int)
    for i in range(step_num):
        step[i] = i
        
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(step, test_dis.T[(node_num-1)*6+1], label='y displacement')
    plt.title('Period')
    plt.xlabel('time [1e-3 sec]', fontsize=24)
    plt.ylabel('elastic displacement [m]', fontsize=24)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=1, fontsize=24)
    plt.tick_params(labelsize=18)
    if save:
        plt.savefig('{0}.png'.format(title))
    plt.show()

# test_NEWMARK_FEMの結果から周期を計算
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

# 要素合成マトリックスと質量マトリックスから周期を計算
def calc_period_from_kL_mass_mat(kL, mass_mat):
    eig_val, eig_vec = scipy.linalg.eig(kL[6:, 6:], mass_mat[6:, 6:])
    w = sorted(eig_val)[0]
    T = 2 * np.pi / np.sqrt(w)
    return T    


# +
# 解析結果の確認とプロットのための関数群
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
    node_num = disG.shape[1] // 6
    if node_id == None:
        node_id = node_num
    if pos==None:
        pos=1
    
    step = np.zeros(step_num, dtype=np.int)
    for i in range(step_num):
        step[i] = i
        
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(1, 1, 1)
    labels = ['x', 'y', 'z', 'rux', 'ruy', 'ruz']
    label = labels[pos]
    ax.plot(step, e_disG.T[(node_id-1)*6+pos], label='{0} node{1}'.format(label, node_id-1))
    plt.title('{0} displacement '.format(label))
    plt.xlabel('time [1e-3 sec]', fontsize=24)
    plt.ylabel('displacement [m]', fontsize=24)
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
        plt.title('{0} displacement'.format(axis))
        plt.xlabel('time [1e-3 sec]', fontsize=18)
        plt.ylabel('coord', fontsize=18)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=14)
        if save:
            plt.savefig('{0}_{1}.png'.format(title, label))        
        plt.show()         
# plot_coods(coordsG_plus_e_dis, [11])

def print_coords(cords, ep=1):
    for i in range(len(cords)):
        if i % ep == 0:
            print(i)
            print('x: ', cords[i][0])
            print('y: ', cords[i][1])
            print('z: ', cords[i][2])
            
def plot_xy_coords(cords, node_ids=None, save=False, title='xy_coord'):
    step_num = cords.shape[0] - 1
    node_num = cords.shape[2]
    if node_ids == None:
        node_ids = [i+1 for i in range(node_num)]     
    

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    axis = axises[i]
    for node_id in node_ids:
        x_data = np.zeros(step_num, dtype=np.float64)
        y_data = np.zeros(step_num, dtype=np.float64)
        for k in range(step_num):
            x_data[k] = cords[k, 0, node_id-1]
            y_data[k] = cords[k, 1, node_id-1]

        ax.scatter(x_data, y_data, label='{0} node{1}'.format(axis, node_id-1))
        plt.title(' x-y plane coordinate'.format(axis))
        plt.xlabel('x coordinate', fontsize=18)
        plt.ylabel('y coordinate', fontsize=18)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=14)
        if save:
            plt.savefig('{0}_{1}.png'.format(title, label))        
        plt.show()        


# -

# abaqusの解析結果データを取得
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
# get_abaqus_data('3d-beam-rot-onlyU.txt', 998, 11)


# +
def plot_xy_coords_compare_with_abaqus(cords, node_ids=None, save=False, title='xy_coord'):
    step_num = cords.shape[0] - 1
    node_num = cords.shape[2]
    if node_ids == None:
        node_ids = [i+1 for i in range(node_num)]         
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    axis = axises[i]
    for node_id in node_ids:
        x_data = np.zeros(step_num, dtype=np.float64)
        y_data = np.zeros(step_num, dtype=np.float64)
        for k in range(step_num):
            x_data[k] = cords[k, 0, node_id-1]
            y_data[k] = cords[k, 1, node_id-1]

        ax.scatter(x_data, y_data, label='{0} node{1}'.format(axis, node_id-1))
        plt.title(' x-y plane coordinate'.format(axis))
        plt.xlabel('x coordinate', fontsize=18)
        plt.ylabel('y coordinate', fontsize=18)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=14)
        if save:
            plt.savefig('{0}_{1}.png'.format(title, label))        
        plt.show()
        
def plot_compare_abaqus_dis(abaqus_dis, disG, node_ids=None, save=False, title='compare_displacement_from_origin_with_abaqus'):
    step_num = disG.shape[0]
    nfree = 6
    node_num = disG.shape[1] // 6
    if node_ids == None:
        node_ids = [i+1 for i in range(node_num)]
    step = np.zeros(step_num, dtype=np.int)
    for i in range(step_num):
        step[i] = i    
    
    labels = ['x', 'y', 'z', 'rux', 'ruy', 'ruz']
    for i in range(6): # ux, uy, uz, rux, ruy, ruz
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        label = labels[i]
        for node_id in node_ids:
            data = np.zeros(step_num, dtype=np.float64)
            abaqus_data = np.zeros(step_num, dtype=np.float64)
            for j in range(step_num):
                data[j] = disG[j, 6*(node_id-1)+i]
                abaqus_data[j] = abaqus_dis[j, 6*(node_id-1)+i]
            ax.plot(step, data, label='linear {0} node{1}'.format(label, node_id-1), color='blue')
            ax.plot(step, abaqus_data, label='non-linear {0} node{1}'.format(label, node_id-1), color='red')            
        plt.title('{0} displacement'.format(label), fontsize=24)
        plt.xlabel('time [1e-3 sec]', fontsize=24)
        plt.ylabel('displacement [m]', fontsize=24)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=24)
        plt.tick_params(labelsize=18)
        if save:
            plt.savefig('{0}_{1}.png'.format(title, label))        
        plt.show()
        
def calc_dis_diff(abaqus_dis, disG, node_id=None):    
    step_num = disG.shape[0]
    nfree = 6
    node_num = disG.shape[1] // 6
    ret = {}
    if node_id == None:
        node_id = [i+1 for i in range(node_num)]    
        
    labels = ['x', 'y', 'z', 'rux', 'ruy', 'ruz']
    for nid in node_id:
        ret[nid] = {}
        for index, label in enumerate(labels):
            diff = np.abs(abaqus_dis.T[(nid-1)*6+index] - disG.T[(nid-1)*6+index])
            max_diff = np.max(diff)
            total_diff = np.sum(diff)
            average_diff = np.average(diff)
            ret[nid][label] = {}
            ret[nid][label]['max_diff'] = max_diff
            ret[nid][label]['total_diff'] = total_diff
            ret[nid][label]['average_diff'] = average_diff
    return ret
# calc_dis_diff(abaqus_dis, disG)

def calc_coord_e_dis(abaqus_dis, disG, coordsG):
    origin_coords = coordsG[0]
    step = disG.shape[0]
    npoin = disG.shape[1] // 6
    coords_e_dis = np.empty_like(coordsG)
    abaqus_coords_e_dis = np.empty_like(coordsG)
    for i in range(step): # i: step
        for j in range(npoin): # j: node
            for k in range(3): # 0: x, 1: y, 2: z
                coords_e_dis[i, k, j] = disG[i, j*6+k] + origin_coords[k, j]
                abaqus_coords_e_dis[i, k, j] = abaqus_dis[i, j*6+k] + origin_coords[k, j]

    return abaqus_coords_e_dis - coordsG, coords_e_dis - coordsG

def plot_compare_abaqus_e_dis(abaqus_coordsG_e_dis, coordsG_e_dis, node_ids=None, save=False,
                            title='compare_elastic_displacement_with_abaqus'):
    step_num = abaqus_coordsG_e_dis.shape[0]    
    node_num = abaqus_coordsG_e_dis.shape[2]
    if node_ids == None:
        node_ids = [i+1 for i in range(node_num)]
        
    step = np.zeros(step_num, dtype=np.int)
    for i in range(step_num):
        step[i] = i
    
    labels = ['x', 'y', 'z']
    for i in range(len(labels)): # x, y, z
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        label = labels[i]
        for node_id in node_ids:
            data = np.zeros(step_num, dtype=np.float64)
            abaqus_data = np.zeros(step_num, dtype=np.float64)
            for j in range(step_num):
                data[j] = coordsG_e_dis[j, i, node_id-1]
                abaqus_data[j] = abaqus_coordsG_e_dis[j, i, node_id-1]                
            ax.plot(step, data, label='linear {0} node{1}'.format(label, node_id-1), color='blue')
            ax.plot(step, abaqus_data, label='non-linear {0} node{1}'.format(label, node_id-1), color='red')            
        plt.title('{0} elastic displacement'.format(label), fontsize=24)
        plt.xlabel('time [1e-3 sec]', fontsize=24)
        plt.ylabel('elastic displacement [m]', fontsize=24)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=24)
        plt.tick_params(labelsize=18)
        if save:
            plt.savefig('{0}_{1}.png'.format(title, label))        
        plt.show()    
        
def plot_compare_abaqus_dis(abaqus_dis, disG, node_ids=None, save=False, title='compare_displacement_from_origin_with_abaqus'):
    step_num = disG.shape[0]
    nfree = 6
    node_num = disG.shape[1] // 6
    if node_ids == None:
        node_ids = [i+1 for i in range(node_num)]
    step = np.zeros(step_num, dtype=np.int)
    for i in range(step_num):
        step[i] = i    
    
    labels = ['x', 'y', 'z', 'rux', 'ruy', 'ruz']
    for i in range(6): # ux, uy, uz, rux, ruy, ruz
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        label = labels[i]
        for node_id in node_ids:
            data = np.zeros(step_num, dtype=np.float64)
            abaqus_data = np.zeros(step_num, dtype=np.float64)
            for j in range(step_num):
                data[j] = disG[j, 6*(node_id-1)+i]
                abaqus_data[j] = abaqus_dis[j, 6*(node_id-1)+i]
            ax.plot(step, data, label='linear {0} node{1}'.format(label, node_id-1), color='blue')
            ax.plot(step, abaqus_data, label='non-linear {0} node{1}'.format(label, node_id-1), color='red')            
        plt.title('{0} displacement'.format(label), fontsize=24)
        plt.xlabel('time [1e-3 sec]', fontsize=24)
        plt.ylabel('displacement [m]', fontsize=24)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=24)
        plt.tick_params(labelsize=18)
        if save:
            plt.savefig('{0}_{1}.png'.format(title, label))        
        plt.show()
        
def calc_dis_diff(abaqus_dis, disG, node_id=None):    
    step_num = disG.shape[0]
    nfree = 6
    node_num = disG.shape[1] // 6
    ret = {}
    if node_id == None:
        node_id = [i+1 for i in range(node_num)]    
        
    labels = ['x', 'y', 'z', 'rux', 'ruy', 'ruz']
    for nid in node_id:
        ret[nid] = {}
        for index, label in enumerate(labels):
            diff = np.abs(abaqus_dis.T[(nid-1)*6+index] - disG.T[(nid-1)*6+index])
            max_diff = np.max(diff)
            total_diff = np.sum(diff)
            average_diff = np.average(diff)
            ret[nid][label] = {}
            ret[nid][label]['max_diff'] = max_diff
            ret[nid][label]['total_diff'] = total_diff
            ret[nid][label]['average_diff'] = average_diff
    return ret
# calc_dis_diff(abaqus_dis, disG)        
