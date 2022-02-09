#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:36:25 2022

@author: ma2sevich
"""

folder='Data/'
data_file_name='VZ 15 Minutes (with indicators).txt'
patterns_file_name='buy_patterns.txt'
results_file_name = "pattern_model_test.csv"
batch=20 # размер бача
pattern=2 # паттерн который проверяем
extrema_window=40 # участок данных на который выбираем локальный минимум или максимум (-значение до точки и +значение после)
list_of_trashholds=[0.04,0.03,0.07]
list_of_patterns=[2,35,3]
data_shape=(-1,20,7) # размер данных