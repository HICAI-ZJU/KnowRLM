import numpy as np
import pandas as pd
import os

groundtruth = pd.read_excel(r"./data/GB1.xlsx")
path = "./result"
dirs = os.listdir(path)

max = []
mean = []
mean_ori = []
count = 0
for d in dirs:
    try:
        result = pd.read_csv(path + "/" + d + "/PredictedFitness.csv")
    except:
        continue

    res = result.sort_values(by="PredictedFitness", ascending=False)
    res_384 = res[res["InTrainingData?"]=="YES"]
    res_96 = res[res["InTrainingData?"]=="NO"].head(96)
    res_con = pd.concat([res_384, res_96])
    ground_fitness_list = []
    ground_fitness_list_96 = []
    ground_fitness_list_96_ori = []
    for j in range(len(res_con)):
        combo = res_con.iloc[j]["AACombo"]
        ground_fitness_list.append(groundtruth[groundtruth["Variants"] == combo]["Fitness"].values[0])
        if res_con.iloc[j]["InTrainingData?"]=="NO":
            ground_fitness_list_96_ori.append(groundtruth[groundtruth["Variants"] == combo]["Fitness"].values[0])

    ground_fitness_list = np.array(ground_fitness_list)
    ground_fitness_list_96 = sorted(ground_fitness_list,reverse = True)[:96]

    print("max:{}".format(np.max(ground_fitness_list)))
    print("mean:{}".format(np.mean(ground_fitness_list_96)))
    print("mean_ori:{}".format(np.mean(ground_fitness_list_96_ori)))
    max.append(np.max(ground_fitness_list))
    mean.append(np.mean(ground_fitness_list_96))
    mean_ori.append(np.mean(ground_fitness_list_96_ori))
    # if np.max(ground_fitness_list) > 133:
    # if np.max(ground_fitness_list) > 8.7:
    #     count += 1

print("max:{}".format(np.mean(max)))
print("mean:{}".format(np.mean(mean)))
print("mean_ori:{}".format(np.mean(mean_ori)))