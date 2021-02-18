#!/usr/bin/env python
# coding: utf-8

# # __Scaling 함수__
# ---

# In[2]:


# DF의 모든 값들이 채워졌을 때 실행
# Z_score_normalization, Scaling 함수로 구성
# Scaling(DF)만 하면 됨


# In[ ]:


import pandas as pd
import numpy as np
import time


# In[ ]:


def z_score_normalize(lst):
    normalized_lst = []
    mean = sum(lst)/len(lst)
    std = np.std(lst)
    normalized_lst = [ (value - mean) / std for value in lst]
    return normalized_lst


# In[ ]:


def Scaling(DF):
    # 스케일링할 컬럼 이름
    column_lst = ['rainfall', 'rain_prob', 'temperature', 'dayofweek_median', 'same_menu']
    
    for column_name in column_lst:
        DF[column_name] = z_score_normalize(DF[column_name])


# In[ ]:


start = time.time()
Scaling(df)
print("time: ", time.time() - start)

