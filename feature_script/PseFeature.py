import re
import itertools

from feature_script import sequence_read_save_csv

myDiIndex = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
    'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15
}
myTriIndex = {
    'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAT': 3,
    'ACA': 4, 'ACC': 5, 'ACG': 6, 'ACT': 7,
    'AGA': 8, 'AGC': 9, 'AGG': 10, 'AGT': 11,
    'ATA': 12, 'ATC': 13, 'ATG': 14, 'ATT': 15,
    'CAA': 16, 'CAC': 17, 'CAG': 18, 'CAT': 19,
    'CCA': 20, 'CCC': 21, 'CCG': 22, 'CCT': 23,
    'CGA': 24, 'CGC': 25, 'CGG': 26, 'CGT': 27,
    'CTA': 28, 'CTC': 29, 'CTG': 30, 'CTT': 31,
    'GAA': 32, 'GAC': 33, 'GAG': 34, 'GAT': 35,
    'GCA': 36, 'GCC': 37, 'GCG': 38, 'GCT': 39,
    'GGA': 40, 'GGC': 41, 'GGG': 42, 'GGT': 43,
    'GTA': 44, 'GTC': 45, 'GTG': 46, 'GTT': 47,
    'TAA': 48, 'TAC': 49, 'TAG': 50, 'TAT': 51,
    'TCA': 52, 'TCC': 53, 'TCG': 54, 'TCT': 55,
    'TGA': 56, 'TGC': 57, 'TGG': 58, 'TGT': 59,
    'TTA': 60, 'TTC': 61, 'TTG': 62, 'TTT': 63
}

baseSymbol = 'ACGT'


def get_kmer_frequency(sequence, kmer):
    myFrequency = {}
    for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
        myFrequency[pep] = 0
    for i in range(len(sequence) - kmer + 1):
        myFrequency[sequence[i: i + kmer]] = myFrequency[sequence[i: i + kmer]] + 1
    for key in myFrequency:
        myFrequency[key] = myFrequency[key] / (len(sequence) - kmer + 1)
    return myFrequency


def correlationFunction(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + (float(myPropertyValue[p][myIndex[pepA]]) - float(myPropertyValue[p][myIndex[pepB]])) ** 2
    return CC / len(myPropertyName)


def correlationFunction_type2(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + float(myPropertyValue[p][myIndex[pepA]]) * float(myPropertyValue[p][myIndex[pepB]])
    return CC


def get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        theta = 0
        for i in range(len(sequence) - tmpLamada - kmer):
            theta = theta + correlationFunction(sequence[i:i + kmer],
                                                sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer], myIndex,
                                                myPropertyName, myPropertyValue)
        thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
    return thetaArray


def get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        for p in myPropertyName:
            theta = 0
            for i in range(len(sequence) - tmpLamada - kmer):
                theta = theta + correlationFunction_type2(sequence[i:i + kmer],
                                                          sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer],
                                                          myIndex,
                                                          [p], myPropertyValue)
            thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
    return thetaArray





def make_PseKNC_vector(fastas, myPropertyName, myPropertyValue, lamadaValue, weight, kmer):
    encodings = []
    myIndex = myDiIndex
    header = ['#']
    header = header + sorted([''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))])
    for k in range(1, lamadaValue + 1):
        header.append('lamada_' + str(k))
    encodings.append(header)
    for i in fastas:
        name, sequence, = i[0], re.sub('-', '', i[1])
        code = [name]
        kmerFreauency = get_kmer_frequency(sequence, kmer)
        thetaArray = get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
        for pep in sorted([''.join(j) for j in list(itertools.product(baseSymbol, repeat=kmer))]):
            code.append(kmerFreauency[pep] / (1 + weight * sum(thetaArray)))
        for k in range(len(baseSymbol) ** kmer + 1, len(baseSymbol) ** kmer + lamadaValue + 1):
            code.append((weight * thetaArray[k - (len(baseSymbol) ** kmer + 1)]) / (1 + weight * sum(thetaArray)))
        encodings.append(code)
    return encodings



def make_SCPseDNC_vector(fastas, myPropertyName, myPropertyValue, lamadaValue, weight):
    encodings = []
    myIndex = myDiIndex
    header = ['#']
    for pair in sorted(myIndex):
        header.append(pair)
    for k in range(1, lamadaValue * len(myPropertyName) + 1):
        header.append('lamada_' + str(k))

    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        dipeptideFrequency = get_kmer_frequency(sequence, 2)
        thetaArray = get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
        for pair in sorted(myIndex.keys()):
            code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
        for k in range(17, 16 + lamadaValue * len(myPropertyName) + 1):
            code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
        encodings.append(code)
    return encodings


def make_SCPseTNC_vector(fastas, myPropertyName, myPropertyValue, lamadaValue, weight):
    encodings = []
    myIndex = myTriIndex
    header = ['#']
    for pep in sorted(myIndex):
        header.append(pep)
    for k in range(1, lamadaValue * len(myPropertyName) + 1):
        header.append('lamada_' + str(k))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        tripeptideFrequency = get_kmer_frequency(sequence, 3)
        thetaArray = get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 3)
        for pep in sorted(myIndex.keys()):
            code.append(tripeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
        for k in range(65, 64 + lamadaValue * len(myPropertyName) + 1):
            code.append((weight * thetaArray[k - 65]) / (1 + weight * sum(thetaArray)))
        encodings.append(code)
    return encodings

import sys, os, platform

import pickle

myDictDefault = {
    'PseKNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
               'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'SCPseDNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                 'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'SCPseTNC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
}

myDataFile = {
    'PseKNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'SCPseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'SCPseTNC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
}


def check_Pse_arguments(fastas,method,nctype,weight,kmer,lamadaValue):
    #os.chdir('C:/Users/Administer/Desktop/DeepAc4C-master/')
    #if not os.path.exists("input.fasta"):
     #   print('Error: the input file does not exist.')
    #    sys.exit(1)
    if not 0 < weight < 1:
        print('Error: the weight factor ranged from 0 ~ 1.')
        sys.exit(1)
    if not 0 < kmer < 10:
        print('Error: the kmer value ranged from 1 - 10')
        sys.exit(1)

    fastaMinLength = 100000000
    for i in fastas:
        if len(i[1]) < fastaMinLength:
            fastaMinLength = len(i[1])
    if not 0 <= lamadaValue <= (fastaMinLength - 2):
        print('Error: lamada value error, please see the manual for details.')
        sys.exit(1)


    myIndex = myDictDefault[method][nctype]
    dataFile = myDataFile[method][nctype]
    if dataFile != '':
        with open('./' + dataFile, 'rb') as f:
            myProperty = pickle.load(f)

    if len(myIndex) == 0 or len(myProperty) == 0:
        print('Error: arguments is incorrect.')
        sys.exit(1)

    return myIndex, myProperty, lamadaValue, weight, kmer

def Pse_feature(fastas):
    import sequence_read_save
    import check_parameters
    
    # # PseKNC
    # my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas,"PseKNC","DNA",0.1,3,10)
    # encodings = make_PseKNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight, kmer)
    # sequence_read_save.save_to_csv(encodings,"./features/PseKNC_test.csv")
    
    # SCPseDNC 
    my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas,"SCPseDNC","DNA",0.1,2,5)
    # print('SCPSEDNC')
    # print(lamada_value, weight, kmer)
    encodings = make_SCPseDNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight) 
    sequence_read_save.save_to_csv(encodings,f'./features/SCPseDNC_tran.csv')
    
    # # SCPseTNC
    # my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas,"SCPseTNC","DNA",0.1,3,4)
    # encodings = make_SCPseTNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    # sequence_read_save.save_to_csv(encodings,"./features/SCPseTNC_test.csv")

if __name__ == '__main__':
    import os
    # os.chdir('C:/Users/Administer/Desktop/DeepAc4C-master/')
    import sequence_read_save
    # for index in range(1, 11):  # range(1, 11) 生成从 1 到 10 的数字
    #     train_fastas = sequence_read_save.read_nucleotide_sequences(
    #         f'../data/balanced_testing_datasets/cd_ac4c_testing_{index}.fasta')
    #     Pse_feature(train_fastas,index)
    train_fastas = sequence_read_save.read_nucleotide_sequences(
                    f'../data/traintran.txt')
    Pse_feature(train_fastas)

