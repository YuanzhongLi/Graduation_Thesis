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

import pickle

with open('newmark精度テストデータ1', 'rb') as f1:
    newmark_data = pickle.load(f1)

with open('後退オイラー法精度テストデータ1', 'rb') as f2:
    back_data = pickle.load(f2)


