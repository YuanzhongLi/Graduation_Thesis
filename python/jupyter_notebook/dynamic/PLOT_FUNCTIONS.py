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


def plot_dis(dis, save=False, title='Newmark B method', fig_name='Newmark_verif.png'):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1,1, 1)
    ax.plot(dis, label='y elastic displacement', color='blue')
    plt.title(title, fontsize=18)
    plt.xlabel('time [1e-3 sec]', fontsize=18)
    plt.ylabel('elastic displacement [m]', fontsize=18)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=14)
    if save:        
        plt.savefig(fig_name)
    plt.show()


# b*1000 stepを1000stepに直す
def format_y_dis(y_dis, b):
    ret = np.zeros((1001), dtype=np.float64)
    for i in range(1, 1001):
        ret[i] = y_dis[i*b]
    return ret 
