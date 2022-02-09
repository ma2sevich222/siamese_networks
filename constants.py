#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:36:25 2022

@author: ma2sevich
"""

SOURCE_ROOT = 'source_root'
DESTINATION_ROOT = 'outputs'


batch = 20  # размер бача
pattern=2  # паттерн который проверяем
extrema_window=40 # участок данных на который выбираем локальный минимум или максимум (-значение до точки и +значение после)
list_of_trashholds=[0.04,0.03,0.07]
list_of_patterns=[2,35,3]
data_shape=(-1,20,7) # размер данных


# list_of_tr = [0.03, 0.003, 0.005, 0.04, 0.005]
# list_of_patt = [48, 57, 4, 34, 77]
# list_of_tr = [0.1]
# list_of_patt = [75]