import re
import pandas as pd


# 从CSV文件读取DNA序列
def read_sequences(file):
    # 读取CSV文件，包含标题行
    df = pd.read_csv(file)

    # 假设CSV文件包含 'seq' 和 'label' 列
    seqs = []
    for _, row in df.iterrows():
        header = row['label']  # 使用标签作为标识符
        sequence = re.sub('[^ACGTU-]', '-', row['seq'].upper())  # 过滤无效字符并大写
        sequence = re.sub('U', 'T', sequence)  # 将U替换为T
        if len(sequence) < 201:
            sequence = sequence.ljust(201, '-')  # 使用 '-' 填充至 201 长度
        seqs.append([header, sequence])
    return seqs


# 互补配对函数
def complementary_pairing(dna_sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement_dict.get(base, '-') for base in dna_sequence)
