# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import csv

#定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


if __name__ == '__main__':
    # #所有节点集合
    AllDrug = pd.read_csv(r'ALLDrug.csv').values.tolist()
    AllMi = pd.read_csv(r'ALLMi.csv').values.tolist()
    AllNode = []
    AllNode.extend(AllDrug)
    AllNode.extend(AllMi)
    print(len(AllDrug),len(AllMi))
    print(len(AllNode))
    feature_drug = pd.read_csv(r'feature_miRNA64.csv', header=None).values.tolist()
    feature_miRNA = pd.read_csv(r'feature_drug64.csv', header=None).values.tolist()
    NodeEmbedding = []
    NodeEmbedding.extend(feature_drug)
    NodeEmbedding.extend(feature_miRNA)
    print(len(feature_drug),len(feature_miRNA))
    print(len(NodeEmbedding))

    # # Behavior
    AllNodeBehavior = []
    counter = 0
    while counter < len(AllNode):
        pair = []
        pair.append(AllNode[counter][0])
        counter1 = 0
        while counter1 < len(NodeEmbedding[0])-2:
            pair.append(0)
            counter1 = counter1 + 1
        AllNodeBehavior.append(pair)
        counter = counter + 1

    print(np.array(NodeEmbedding).shape)

    for a in range(len(AllNode)):
        for b in range(len(NodeEmbedding)):
            if AllNode[a][1] == NodeEmbedding[b][0]:
                #print(a,b)
                AllNodeBehavior[a][1:] = NodeEmbedding[b][1:-1]
                break


    print(np.array(NodeEmbedding).shape)

    pd.DataFrame(AllNodeBehavior).to_csv(r'SM2miR2 BiNE 64.csv',header=None,index=None)

