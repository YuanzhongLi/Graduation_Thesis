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
import pickle
import matplotlib.pyplot as plt


def calc_diff(data):
    ret = {}
    keys = ['step1000', 'step2000', 'step5000', 'step10000', 'step100000']
    for key in keys:
        if key == 'step100000':
            continue
        diff = 0.0
        max_diff = 0.0
        for (i, j) in zip(data[key], data['step100000']):
            diff += abs(i - j)
            max_diff = max(max_diff, abs(i - j))

        average_diff = diff / 1000   
        ret['max diff {0}'.format(key)] = max_diff
        ret['average diff {0}'.format(key)] = average_diff
    return ret


with open('newmark精度テストデータ001', 'rb') as f1:
    newmark_data = pickle.load(f1)

with open('後退オイラー法精度テストデータ001', 'rb') as f2:
    back_data = pickle.load(f2)

newmark_diff = calc_diff(newmark_data)

newmark_diff

back_diff = calc_diff(back_data)

back_diff


# +
def plot_log_max_diff(newmark_diff, back_diff, save=False):
    x = np.array([1e3, 2e3, 5e3, 1e4])
    y1 = np.array([newmark_diff['max diff step1000'], 
                   newmark_diff['max diff step2000'],
                   newmark_diff['max diff step5000'],
                   newmark_diff['max diff step10000']], dtype=np.float64)
    y2 = np.array([back_diff['max diff step1000'], 
                   back_diff['max diff step2000'],
                   back_diff['max diff step5000'],
                   back_diff['max diff step10000']], dtype=np.float64)
    
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x, y1, color='red', label='Newmark B difference', marker='o')
    plt.plot(x, y2, color='blue', label='back Euler difference', marker='o')

    ax = plt.gca()
    ax.spines['top'].set_color('none')

    ##
    ax.set_yscale('log')  # メイン: y軸をlogスケールで描く
    ax.set_xscale('log')
    ##
    plt.title('max difference logarithmic plot') 
    plt.xlabel('step',fontsize=18)
    plt.ylabel('diffence',fontsize=18)

    plt.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=14)

    if save:
        plt.savefig('max_diff.png')

    plt.show()
    
def plot_log_average_diff(newmark_diff, back_diff, save=False):
    x = np.array([1e3, 2e3, 5e3, 1e4])
    y1 = np.array([newmark_diff['average diff step1000'], 
                   newmark_diff['average diff step2000'],
                   newmark_diff['average diff step5000'],
                   newmark_diff['average diff step10000']], dtype=np.float64)
    y2 = np.array([back_diff['average diff step1000'], 
                   back_diff['average diff step2000'],
                   back_diff['average diff step5000'],
                   back_diff['average diff step10000']], dtype=np.float64)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x, y1, color='red', label='Newmark B difference', marker='o')
    plt.plot(x, y2, color='blue', label='back Euler difference', marker='o')

    ax = plt.gca()
    ax.spines['top'].set_color('none')

    ##
    ax.set_yscale('log')  # メイン: y軸をlogスケールで描く
    ax.set_xscale('log')
    ##
    plt.title('average difference logarithmic plot') 
    plt.xlabel('step',fontsize=18)
    plt.ylabel('diffence',fontsize=18)

    plt.grid(which="both") # グリッド表示。"both"はxy軸両方にグリッドを描く。
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=14)

    if save:
        plt.savefig('average_diff.png')

    plt.show()    


# -

plot_log_max_diff(newmark_diff, back_diff)

plot_log_average_diff(newmark_diff, back_diff)
