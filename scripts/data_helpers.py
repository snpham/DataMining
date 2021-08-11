import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import plotly.express as px



def min_max_normalization(fileName=None, attribute=None, old_min=None, old_max=None, 
                          new_min=0, new_max=1, single_val=None):
    '''
    :param fileName: filename of csv for normalization
    :param attribute: name of attribute to normalize within file
    :param old_min: minimum value of original data list 
    :param old_max: maximum value of original data list
    :param new_min: new minimum to normalize to
    :param new_max: new maximum to normalize to
    :param single_val: compute only for a single case
    '''

    if single_val is not None:
        return round((single_val-old_min)/(old_max-old_min)*(new_max-new_min)+new_min, 6)

    csv_file = pd.read_csv(fileName)
    
    if attribute == 'close':
        attr = 'Close/Last'
    else:
        attr = attribute.capitalize()
        
    pd_attr = pd.DataFrame()
    pd_attr[attr] = csv_file[attr]
    
    if attr != 'Volume':
        pd_attr[attr] = pd_attr[attr].str[1:].astype(float)
    else:
        pd_attr[attr] = pd_attr[attr].astype(float)

    new_max = 1
    new_min = 0
    pd_attr[attr + '_norm'] = \
        (pd_attr[attr]-pd_attr[attr].min()) \
            / (pd_attr[attr].max() - pd_attr[attr].min()) \
                * (new_max-new_min) + new_min
    
    return pd_attr[attr + '_norm'].tolist()


def zscore_normalization(fileName=None, attribute=None, single_val=None, 
                         mean_val=None, std_val=None):
    '''
    :param fileName: filename of csv to be normalized
    :param attribute: attribute name
    :param single_val: compute using a single value (instead of a file)
    :param mean_val: mean value of the attribute (for single_val compute)
    :param std_val: standard deviation of the attribute (for single val)
    :return: list of z-score normalized values
    not tested using mean absolute deviation
    '''
    if single_val is not None:
        return round((single_val-mean_val)/std_val, 6)

    csv_file = pd.read_csv(fileName)

    if attribute == 'close':
        attr = 'Close/Last'
    else:
        attr = attribute.capitalize()
        
    pd_attr = pd.DataFrame()
    pd_attr[attr] = csv_file[attr]
    
    if attr != 'Volume':
        pd_attr[attr] = pd_attr[attr].str[1:].astype(float)
    else:
        pd_attr[attr] = pd_attr[attr].astype(float)

    attr_list = pd_attr[attr].to_list()
    attr_mean =  mean(attr_list)
    mean_abs_dev = 1/len(attr_list)*sum(np.abs(x-attr_mean) for x in attr_list)
    pd_attr[attr + '_znorm'] = \
        (pd_attr[attr]-attr_mean)/mean_abs_dev

    return pd_attr[attr + '_znorm'].tolist()


def mean(datalist):
    return sum(datalist)/len(datalist)


def median(datalist):
    datalist.sort()
    mid = len(datalist) // 2
    return (datalist[mid] + datalist[~mid]) / 2


def mode(datalist, type='numeric'):
    from collections import Counter
    c = Counter(datalist)
    return [k for k, v in c.items() if v == c.most_common(1)[0][1]]



if __name__ == '__main__':

    # # norm for a single value - min/max
    # val = min_max_normalization(single_val=73600, old_min=12000, old_max=98000)
    # assert np.allclose(val, 0.716279)
    
    # # norm for a single value - z-score
    # val = zscore_normalization(single_val=73600, mean_val=54000, std_val=16000)
    # assert np.allclose(val, 1.225)

    # fileName = 'Data/AMAT_HistoricalData_1623003561951.csv'
    # attribute = 'low'
    # vals = zscore_normalization(fileName, attribute)
    # # print(vals)

    # datalist = [30, 47, 50, 52, 52, 56, 110, 36, 60, 63, 70, 70]
    # ret = mean(datalist)
    # print(ret)

    # datalist = [30, 47, 50, 52, 52, 56, 110, 36, 60, 63, 70, 70]
    # ret = median(datalist)
    # print(ret)

    # datalist = [30, 47, 50, 52, 52, 56, 110, 36, 60, 63, 70, 70]
    # ret = mode(datalist)
    # print(ret)

    dataset = pd.read_csv('data/integrated_data_v3.csv', index_col=0, header=0)
    numcols = ['Num1','Num2a','Num2b','Num3']
    dataset = dataset[np.hstack(['Residency', numcols])]

    for num in numcols:
        old_min = dataset[num].min()
        old_max = dataset[num].max()
        if num == 'Num3':
            old_max = 100
        if num == 'Num1':
            new_min = 0
            new_max = 1
        if num == 'Num2a':
            new_min = 0
            new_max = 1
        if num == 'Num2b':
            new_min = 0
            new_max = 1
        if num == 'Num3':
            new_min = 0
            new_max = 1
        for row in range(0, len(dataset[num])):
            val = dataset[num].iloc[row]
            if val is not None:
                newval = min_max_normalization(old_min=old_min, old_max=old_max, 
                                    new_min=new_min, new_max=new_max, single_val=val)
                if num == 'Num1':
                    num1_correct = min_max_normalization(old_min=old_min, old_max=old_max, 
                                    new_min=new_min, new_max=new_max, single_val=25)
                if num == 'Num2a':
                    num2a_correct = min_max_normalization(old_min=old_min, old_max=old_max, 
                                    new_min=new_min, new_max=new_max, single_val=30)
                if num == 'Num2b':
                    num2b_correct = min_max_normalization(old_min=old_min, old_max=old_max, 
                                    new_min=new_min, new_max=new_max, single_val=20)
                if num == 'Num3':
                    num3_correct = min_max_normalization(old_min=old_min, old_max=old_max, 
                                    new_min=new_min, new_max=new_max, single_val=50)
            dataset[num].iloc[row] = newval 
    print(num1_correct, num2a_correct, num2b_correct, num3_correct)
    dataset.to_csv('data/numerical_dataset.csv')

    fig = px.box(dataset, x="Residency", y="Num1", notched=True, points='all')
    fig.add_shape(type='line', x0='AU',y0=num1_correct,x1='US',y1=num1_correct,
                    line=dict(color='Red',), xref='x', yref='y')
    fig.write_image("outputs/plots/num1_stats.pdf")
    # fig.show()
    fig = px.box(dataset, x="Residency", y="Num2a", notched=True, points='all')
    fig.add_shape(type='line', x0='AU',y0=num2a_correct,x1='US',y1=num2a_correct,
                    line=dict(color='Red',), xref='x', yref='y')
    fig.write_image("outputs/plots/num2a_stats.pdf")
    # fig.show()
    fig = px.box(dataset, x="Residency", y="Num2b", notched=True, points='all')
    fig.add_shape(type='line', x0='AU',y0=num2b_correct,x1='US',y1=num2b_correct,
                    line=dict(color='Red',), xref='x', yref='y')
    fig.write_image("outputs/plots/num2b_stats.pdf")
    # fig.show()
    fig = px.box(dataset[dataset['Num3'] <= 1], x="Residency", y="Num3", notched=True, points='all')
    fig.add_shape(type='line', x0='AU',y0=num3_correct,x1='US',y1=num3_correct,
                    line=dict(color='Red',), xref='x', yref='y')
    fig.write_image("outputs/plots/num3_stats.pdf")
    # fig.show()
