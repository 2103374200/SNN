# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:40:49 2020

@author: Administer
"""
import pandas as pd


def read_nucleotide_sequences(file):
    import re, os, sys
    df = pd.read_csv(file)

    # 假设CSV文件包含 'seq' 和 'label' 列
    seqs = []
    for _, row in df.iterrows():
        header = row['label']  # 使用标签作为标识符
        sequence = re.sub('[^ACGTU-]', '-', row['seq'].upper())  # 过滤无效字符并大写
        sequence = re.sub('U', 'T', sequence)  # 将U替换为T
        seqs.append([header, sequence])
    return seqs
 

def save_to_csv(encodings, file):
    with open(file, 'w') as f:
        for line in encodings[1:]:
            f.write(str(line[0]))
            for i in range(1,len(line)):
                f.write(',%s' % line[i])
            f.write('\n')

def file_remove():
    import os
    dir_list1=os.listdir("./features/")
    for x in dir_list1:
        if x.split(".")[-1]=="csv":
            os.remove("./features/"+str(x))
    for i in range(1,11,1):
        dir_list2=os.listdir("./features/mm/"+str(i))
        for x in dir_list2:
            if x.split(".")[-1]=="csv":
                os.remove("./features/mm/"+str(i)+"/"+str(x)) 
        dir_list3=os.listdir("./features/mm/"+str(i)+"/f_b/")
        for x in dir_list3:
            if x.split(".")[-1]=="csv":
                os.remove("./features/mm/"+str(i)+"/f_b/"+str(x))
                
    dir_list4=os.listdir("./features/combined_features/")
    for x in dir_list4:
        if x.split(".")[-1]=="csv":
            os.remove("./features/combined_features/"+str(x))
        
    # dir_list5=os.listdir("./results/")
    # for x in dir_list5:
        # os.remove("./results/"+str(x))