import numpy as np
import pandas as pd
from pandas.core.reshape.tile import qcut


def dissimilarity_nominal(dataset=None, p=None, m=None, weights=None):
    """computes the dissimilarity b/t two objects (for nominal
    attributes). Can input either a column dataset or directly input
    p and m values.
    :param dataset: pandas dataset to perform dissimiarity analysis
    :param p: total number of attributes describing the objects
    :param m: the number of matches
    :param weights: optional weights array to increase the effect
    of m or to assign greater weight to the matches in attributes
    having a larger number of states
    return: dissimilarity matrix
    only testes for single column
    """
    if dataset is not None:
        dis_mat = np.zeros((len(dataset), (len(dataset))))
        p = len(dataset.columns)
        m = 0
        for i in range(0, len(dis_mat)):
            for j in range(0, len(dis_mat)):
                for col in dataset.columns:
                    if dataset[col].iloc[i] == dataset[col].iloc[j]:
                        m += 1
                dis_mat[i, j] = (p-m)/p
                m = 0
        return dis_mat    
    elif p != None and m != None:
        return round((p-m)/p, 2)


def similarity_nominal(dataset=None, p=None, m=None, weights=None):
    """computes the similarity b/t two objects (for nominal
    attributes). Can input either a column dataset or directly input
    p and m values.
    :param dataset: pandas dataframe to perform similarity analysis
    :param p: total number of attributes describing the objects
    :param m: the number of matches
    :param weights: optional weights array to increase the effect
    of m or to assign greater weight to the matches in attributes
    having a larger number of states
    return: similarity matrix
    only testes for single column
    """
    if dataset is not None:
        dis_mat = dissimilarity_nominal(dataset=dataset, p=p, m=m, 
                                        weights=weights)
        sim_mat = np.subtract(1, dis_mat)
        return sim_mat
    elif p != None and m != None:
        return round(m/p, 2)


def dissimilarity_binary(dataset=None, q=None, r=None, s=None, t=None, 
                         symmetric=True):
    """computes the dissimilarity b/t two objects (for binary
    attributes). Can input either a column dataset or directly input
    q, r, s, t values.
    :param dataset: pandas dataframe to perform dissimilarity analysis
    :param q: number of attributes that equal 1 for both objects i and j
    :param r: nuber of attributes that equal 1 for object i but 0 for j
    :param s: number of attributes that equal 0 for i but 1 for j
    :param t: number of attributes that equal 0 for both i and j
    :param symmetric: True=binary attribute are symmetric; each state is 
    equally valuable. False=asymmetric binary attribute; states are not 
    equally important
    :return: binary dissimilarity
    """
    if dataset is not None:
        dis_mat = np.zeros((len(dataset), (len(dataset))))
        q = 0
        r = 0
        s = 0
        t = 0
        for i in range(0, len(dis_mat)):
            for j in range(0, len(dis_mat)):
                for col in dataset.columns:
                    a = int(dataset[col].iloc[i])
                    b = int(dataset[col].iloc[j])
                    if a == 1 and b == 1:
                        q += 1
                    elif a == 1 and b == 0:
                        r += 1
                    elif a == 0 and b == 1:
                        s += 1
                    elif a == 0 and b == 0:
                        t += 1
                    if symmetric:
                        dis_mat[i, j] = round((r+s)/(q+r+s+t), 2)
                    elif not symmetric:
                        dis_mat[i, j] = round((r+s)/(q+r+s), 2)
                q = 0
                r = 0
                s = 0
                t = 0
        return dis_mat    
    elif q != None and r != None and s != None and t != None:
        if symmetric:
            return round((r+s)/(q+r+s+t), 2)
        elif not symmetric:
            return round((r+s)/(q+r+s), 2)


def similarity_binary(dataset=None, q=None, r=None, s=None, t=None, symmetric=True):
    """measure the difference b/t two binary attributes based on similarity; 
    also known as the Jaccard coefficient.
    :param q: number of attributes that equal 1 for both objects i and j
    :param r: nuber of attributes that equal 1 for object i but 0 for j
    :param s: number of attributes that equal 0 for i but 1 for j
    :param t: number of attributes that equal 0 for both i and j
    :param symmetric: True=binary attribute are symmetric; each state is 
    equally valuable. False=asymmetric binary attribute; states are not 
    equally important
    :return: binary similarity matrix
    """
    if dataset is not None:
        dis_mat = dissimilarity_binary(dataset=dataset, q=q, r=r, s=s, t=t, 
                                   symmetric=symmetric)
        sim_mat = np.subtract(1, dis_mat)
        return sim_mat
    elif q != None and r != None and s != None and t != None:
        if symmetric:
            return round(q/(q+r+s+t), 2)
        else:
            return round(q/(q+r+s), 2)


def manhattan_distance(dataset=None, x=None, y=None):
    """
    :param dataset: two object dataset with numeric attributes
    :param x: list of first object's numeric attributes
    :param y: list of second object's numeric attributes
    :return: manhattan distance
    """
    if dataset is not None:
        x = dataset.iloc[0, :].tolist()
        y = dataset.iloc[1, :].tolist()
    return round(sum(np.abs(a-b) for a, b in zip(x, y)), 4)


def euclidean_distance(dataset=None, x=None, y=None):
    """
    :param dataset: two object dataset with numeric attributes
    :param x: list of first object's numeric attributes
    :param y: list of second object's numeric attributes
    :return: euclidean distance
    """
    if dataset is not None:
        x = dataset.iloc[0, :].tolist()
        y = dataset.iloc[1, :].tolist()
    return round(np.sqrt(sum((a-b)**2 for a, b in zip(x ,y))), 4)


def minkowski_distance(dataset=None, x=None, y=None, p_value=None):
    """generalization of the euclidean and manhattan distances.
    :param dataset: two object dataset with numeric attributes
    :param x: list of first object's numeric attributes
    :param y: list of second object's numeric attributes
    :return: minkowski distance
    """
    if dataset is not None:
        x = dataset.iloc[0, :].tolist()
        y = dataset.iloc[1, :].tolist()
    sum_val = sum(np.abs(a-b)**p_value for a, b in zip(x, y))

    return np.round(sum_val**(1 / p_value), 4)


def hamming_distance(s1, s2):
    """return the Hamming distance b/t equal-length sequences
    """
    if len(s1) != len(s2):
        raise ValueError("undefined for sequences of unequal length")
    result = sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

    return (len(s1) - result) / len(s1)


def cosine_similarity(x,y):
    numerator = sum(a*b for a, b in zip(x,y))
    sqrtx = round(np.sqrt(sum([a*a for a in x])), 3)
    sqrty = round(np.sqrt(sum([a*a for a in y])), 3)
    denom = sqrtx*sqrty
    result = round(numerator/denom, 4)

    return result


if __name__ == '__main__':


    ## nominal dissimilarity/similarity
    # df_mixed = pd.DataFrame()
    # df_mixed['test1_nom'] = ['code A', 'code B', 'code C', 'code A']
    # df_mixed['test2_ord'] = ['excellent', 'fair', 'good', 'excellent']
    # df_mixed['test3_num'] = [45, 22, 64, 28]
    # df_mixed.to_csv('data/mixed_sample.csv')
    df_mixed = pd.read_csv('data/mixed_sample.csv', index_col=0)
    # print(df_mixed)
    df_nominal = df_mixed[['test1_nom']]
    # print(df_nominal)
    dis_mat = dissimilarity_nominal(dataset=df_nominal, p=None, m=None, weights=None)
    # print(dis_mat)
    # [[0. 1. 1. 0.]
    # [1. 0. 1. 1.]
    # [1. 1. 0. 1.]
    # [0. 1. 1. 0.]]
    sim_mat = similarity_nominal(dataset=df_nominal, p=None, m=None, weights=None)
    # print(sim_mat)
    # [[1. 0. 0. 1.]
    # [0. 1. 0. 0.]
    # [0. 0. 1. 0.]
    # [1. 0. 0. 1.]]

    ## binary dissimilarity/similarity
    # df_binary = pd.DataFrame()
    # df_binary['name'] = ['Jack', 'Jim', 'Mary']
    # df_binary['gender'] = ['M', 'M', 'F']
    # df_binary['fever'] = ['Y', 'Y', 'Y']
    # df_binary['cough'] = ['N', 'Y', 'N']
    # df_binary['test1'] = ['P', 'N', 'P']
    # df_binary['test2'] = ['N', 'N', 'N']
    # df_binary['test3'] = ['N', 'N', 'P']
    # df_binary['test4'] = ['N', 'N', 'N']
    # df_binary.to_csv('data/binary_sample.csv')
    df_binary = pd.read_csv('data/binary_sample.csv', index_col=0)
    for i in range(0, len(df_binary)):
        for j in range(0, len(df_binary.columns)):
            if df_binary.iloc[i, j] in ['Y', 'P']:
                df_binary.iloc[i, j] = 1
            elif df_binary.iloc[i, j] == 'N':
                df_binary.iloc[i, j] = 0
    # print(df_binary)
    df_binary_asym = df_binary[['fever', 'cough', 'test1', 'test2', 
                                'test3', 'test4']]
    dis_mat = dissimilarity_binary(dataset=df_binary_asym, q=None, r=None, 
                                   s=None, t=None, symmetric=False)
    # print(dis_mat)
    # [[0.   0.67 0.33]
    # [0.67 0.   0.75]
    # [0.33 0.75 0.  ]]
    sim_mat = similarity_binary(dataset=df_binary_asym, q=None, r=None, 
                                s=None, t=None, symmetric=False)
    # print(sim_mat)
    # [[1.   0.33 0.67]
    # [0.33 1.   0.25]
    # [0.67 0.25 1.  ]]
    dis_val = dissimilarity_binary(dataset=None, q=1, r=1, 
                                s=1, t=1, symmetric=False)
    # print(dis_val) # 0.67
    dis_val = dissimilarity_binary(dataset=None, q=1, r=1, 
                                s=2, t=0, symmetric=False)
    # print(dis_val) # 0.75


    # numeric data dissimilarity

    # manhattan distance
    result = manhattan_distance(x=[10, 20, 10], y=[10, 20, 20])
    # print(result) # 10
    dataset = pd.DataFrame(data=np.array([[10, 20, 10], [10, 20, 20]]), 
                           columns=['a', 'b', 'c'])
    result = manhattan_distance(dataset=dataset, x=None, y=None)
    # print(result) # 10

    # euclidean distance
    result = euclidean_distance(x=[0, 3, 4, 5], y=[7, 6, 3, -1])
    # print(result) # 9.7468
    dataset = pd.DataFrame(data=np.array([[0, 3, 4, 5], [7, 6, 3, -1]]))
    result = euclidean_distance(dataset=dataset, x=None, y=None)
    # print(result) # 9.7468

    # minkowski distance
    result = minkowski_distance(x=[0, 3, 4, 5], y=[7, 6, 3, -1], p_value=3)
    print(result) # 8.373
    dataset = pd.DataFrame(data=np.array([[0, 3, 4, 5], [7, 6, 3, -1]]))
    result = minkowski_distance(dataset=dataset, x=None, y=None, p_value=3)
    print(result) # 8.373
    exit()

    # hamming distance
    result = hamming_distance('CATCATCATCATCATCATCTTTTT',
                              'CATCATCTTCATCATCATCTTTTT')
    print(result)

    # hamming distance 2
    result = hamming_distance('ATGCATCATCATCATCATCTTTTT',
                              'CATCATCTTCATCATCATCTTTTT')
    print(result)





    # minkowski distance
    result = cosine_similarity([5, 0, 3, 0, 2, 0, 0, 2, 0, 0], [3, 0, 2, 0, 1, 1, 0, 1, 0, 1])
    print(result)
