"""
author:wyh
data:2023.11.13
"""
import numpy as np
from sklearn.metrics import ndcg_score

import pandas as pd
import os
path = "./result"
dirs = os.listdir(path)
# 读取文件
df_A = pd.read_excel('./data/GB1.xlsx')
m = 0
mean_lsit = []
for d in dirs:
    try:
        df_B = pd.read_csv(path + "/" + d + "/PredictedFitness.csv")
    except:
        continue
    # 按照Fitness进行降序排列
    df_A = df_A.sort_values(by='Fitness', ascending=False)

    # 将B文件中的Variants值设为索引，以便快速查找
    df_B = df_B.set_index('AACombo')

    # 对A文件中的每一行，找出对应的PredictedFitness
    df_A['PredictedFitness'] = df_A['Variants'].map(df_B['PredictedFitness'])

    # 如果你想要删除没有找到匹配的行
    df_A = df_A.dropna(subset=['PredictedFitness'])

    # 得到两个列表
    fitness_list = np.array(df_A['Fitness'])
    predicted_fitness_list = np.array(df_A['PredictedFitness'])
    ndcg = ndcg_score(np.expand_dims(fitness_list,axis=0),np.expand_dims(predicted_fitness_list,axis=0))
    m = max(m,ndcg)
    mean_lsit.append(ndcg)
    print(d + ":" + str(ndcg))

print("max:"+str(m))
mean_lsit = np.array(mean_lsit)
print("mean:"+str(np.mean(mean_lsit)))
