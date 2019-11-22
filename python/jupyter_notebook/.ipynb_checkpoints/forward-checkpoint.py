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


def inputdata(fnameR, nod, nfree):  #データパス, 要素節点数, 自由度
    f = open(fnameR, 'r')
    text = f.readline()  # コメント
    text = f.readline()  # 1行ずつ読み込む
    text = text.strip()  # 端の空白, 改行(\n)を削除
    text = text.split()  # ','　で別れた要素ごとの配列にする
    npoin = int(text[0])  # ノード数
    nele = int(text[1])  # 要素数
    npfix = int(text[2])  # 拘束点数
    nlod = int(text[3])  # 荷重点数
    delta_t = float(text[4])  # 微小時間
    n_t = int(text[5])  # 微小時間数
    # 配列宣言
    ae = np.zeros((11, nele), dtype=np.float64)  # 要素特性
    node = np.zeros((nod + 1, nele), dtype=np.int)  # 要素構成節点
    x = np.zeros((3, npoin), dtype=np.float64)  # 座標
    mpfix = np.zeros((n_t + 1, nfree, npoin), dtype=np.int)  # 拘束状態
    rdis = np.zeros((n_t + 1, nfree, npoin), dtype=np.float64)  # 既知変位
    fp = np.zeros((n_t + 1, nfree * npoin), dtype=np.float64)  # 外力
    # 要素特性: ae
    text = f.readline()  # コメント
    for i in range(0, nele):
        text = f.readline()
        text = text.strip()
        text = text.split()
        ae[0, i] = float(text[0])  # E
        ae[1, i] = float(text[1])  # Po
        ae[2, i] = float(text[2])  # A
        ae[3, i] = float(text[3])  # Ix
        ae[4, i] = float(text[4])  # Iy
        ae[5, i] = float(text[5])  # Iz
        ae[6, i] = float(text[6])  # density
        ae[7, i] = float(text[7])  # gkx
        ae[8, i] = float(text[8])  # gky
        ae[9, i] = float(text[9])  # gkz

    text = f.readline()  # コメント
    # 要素構成節点: node
    for i in range(0, nele):
        text = f.readline()
        text = text.strip()
        text = text.split()
        node[0, i] = int(text[0])  #node_1
        node[1, i] = int(text[1])  #node_2
        node[2, i] = int(text[2])  #要素番号

    text = f.readline()  # コメント
    # 座標: x
    for i in range(0, npoin):
        text = f.readline()
        text = text.strip()
        text = text.split()
        x[0, i] = float(text[0])  # x-座標
        x[1, i] = float(text[1])  # y-座標
        x[2, i] = float(text[2])  # z-座標
    
    # 要素質量
    for i in range(0, nele):
        node1_x = x[0, i]
        #         node1_y = x[1, i]
        #         node1_z = x[2, i]
        node2_x = x[0, i + 1]
        #         node2_y = x[1, i+1]
        #         node2_z = x[2, i+1]
        A = ae[2, i]
        density = ae[6, i]
        ae[10, i] = abs(node1_x - node2_x) * A * density
        
#     for j in range(0, n_t+1):

    text = f.readline()  # コメント
    # 境界条件（拘束状態） (0:free, 1:restricted)
    for i in range(0, npfix):
        text = f.readline()
        text = text.strip()
        text = text.split()
        lp = int(text[0])  # 固定されたノード番号
        for j in range(0, n_t + 1):
            mpfix[j, 0, lp - 1] = int(text[1])  # x方向固定
            mpfix[j, 1, lp - 1] = int(text[2])  # y方向固定
            mpfix[j, 2, lp - 1] = int(text[3])  # z方向固定
            mpfix[j, 3, lp - 1] = int(text[4])  # x軸回転固定
            mpfix[j, 4, lp - 1] = int(text[5])  # y軸回転固定
            mpfix[j, 5, lp - 1] = int(text[6])  # z軸回転固定
            rdis[j, 0, lp - 1] = float(text[7])  # x方向既知変位
            rdis[j, 1, lp - 1] = float(text[8])  # y方向既知変位
            rdis[j, 2, lp - 1] = float(text[9])  # z方向既知変位
            rdis[j, 3, lp - 1] = float(text[10])  # x軸既知回転量
            rdis[j, 4, lp - 1] = float(text[11])  # y軸既知回転量
            rdis[j, 5, lp - 1] = float(text[12])  # z軸既知回転量

    text = f.readline()  # コメント
    # 荷重
    for i in range(0, nlod):
        text = f.readline()
        text = text.strip()
        text = text.split()
        lp = int(text[0])
        for j in range(0, n_t + 1):
            fp[j, 6 * lp - 6] = float(text[1])  # x方向荷重
            fp[j, 6 * lp - 5] = float(text[2])  # y方向荷重
            fp[j, 6 * lp - 4] = float(text[3])  # z方向荷重
            fp[j, 6 * lp - 3] = float(text[4])  # x軸モーメント
            fp[j, 6 * lp - 2] = float(text[5])  # y軸モーメント
            fp[j, 6 * lp - 1] = float(text[6])  # z軸モーメント
    f.close()
    return npoin, nele, npfix, nlod, delta_t, n_t, ae, node, x, mpfix, rdis, fp


