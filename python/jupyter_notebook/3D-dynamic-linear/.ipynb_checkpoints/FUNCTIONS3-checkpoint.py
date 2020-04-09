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
#     print(sorted(kL_eig_val)[:7])
    eig_val, eig_vec = scipy.linalg.eig(kL[6:, 6:], mass_mat[6:, 6:])
#     print(sorted(eig_val))
    w = sorted(eig_val)[0]
    T = 2 * np.pi / np.sqrt(w)
    return T
