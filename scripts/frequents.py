import numpy as np
import pandas as pd


def data_gen(dataset):
    for row in dataset.iloc[:,0]:
        yield row.replace(" ", "").split(",")


def row_gen(dataset):
    for row in dataset:
        yield row.replace(" ", "").split(",")


def threshold(data):
    return {item:count for item, count in data.items() if count >= min_sup}


if __name__ == '__main__':

    dataset = pd.read_csv('data/apriori_ex.csv', index_col=0, header=0)
    min_sup = 0.5
    min_sup *= len(dataset)
    print(dataset)

    freq1_items = set()
    for data in data_gen(dataset):
        for item in data:
            freq1_items.add(item)
    print(freq1_items)


    scan1 = dict.fromkeys(freq1_items, 0)
    for data in data_gen(dataset):
        for item in data:
            scan1[item] += 1
    scan1 = threshold(scan1)
    print('scan1', scan1)

    unionset1 = set([','.join((i, j)) for i in scan1 for j in scan1 if i != j if i < j])
    scan2 = dict.fromkeys(unionset1, 0)
    # print(scan2)

    for data in data_gen(dataset):
        for row in scan2:
            if set(row.split(',')).issubset(set(data)):
                scan2[row] += 1
    scan2 = threshold(scan2)
    print('scan2', scan2)

    unionset2 = set([','.join((i, j)) for i in scan2 for j in scan2 if i != j if i < j])
    # print('unionset2', unionset2)
    newunion2 = set()
    for row in unionset2:
        newset = set(sorted(row.split(',')))
        for col in row_gen(scan2):
            if set(col).issubset(newset) and scan2[','.join(col)] >= 3 and len(newset) == 3:
                newunion2.add(','.join(newset))
    # print(newunion2)
    scan3 = dict.fromkeys(newunion2, 0)
    # print(scan3)

    for data in data_gen(dataset):
        for row in scan3:
            if set(row.split(',')).issubset(set(data)):
                scan3[row] += 1
    scan3 = threshold(scan3)
    print('scan3', scan3)