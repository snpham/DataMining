import numpy as np
import pandas as pd


def correlation(fn1, fn2, att1, att2):

    data1 = pd.read_csv(fn1)
    data2 = pd.read_csv(fn2)

    a = data1[att1].tolist()
    b = data2[att2].tolist()
    A = data1[att1].mean()
    B = data2[att2].mean()
    N = len(data1[att1])
    stddeva = data1[att1].std()
    stddevb = data2[att2].std()

    corr = sum((x-A)*(y-B) for x, y in zip(a, b))/(N*stddeva*stddevb)
#     corr = (sum((x*y) for x, y in zip(a, b))-N*A*B)/(N*stddeva*stddevb)    
#     corr = data1[att1].corr(data2[att2])

    return round(corr, 6)


def min_max_normalization(fileName=None, attribute=None, old_min=None, old_max=None, 
                          new_min=0, new_max=1, single_val=None):
    '''
    Input Parameters:
        fileName: The comma seperated file that must be considered for 
        the normalization with the different format first line removed
        attribute: The attribute for which you are performing the 
        normalization
        
    Output:
        Return an object of list type of normalized values
        Please do not return a list of strings
        Please do not print anything to stdout
        Use only Python3
        Points will be deducted if you do not follow exact instructions
    '''
    if single_val:
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
    Input Parameters:
        fileName: The comma seperated file that must be considered for the 
        normalization with the different format first line removed
        attribute: The attribute for which you are performing the normalization
        
    Output:
        Return an object of list type of normalized values
        Please do not return a list of strings
        Please do not print anything to stdout
        Use only Python3
        Points will be deducted if you do not follow exact instructions
    '''
    if single_val:
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

    pd_attr[attr + '_znorm'] = \
        (pd_attr[attr]-pd_attr[attr].mean())/pd_attr[attr].std()
    
    return pd_attr[attr + '_znorm'].tolist()


if __name__ == '__main__':

    # norm for a single value
    val = min_max_normalization(single_val=73600, old_min=12000, old_max=98000)
    print(val)
    
    val = zscore_normalization(single_val=73600, mean_val=54000, std_val=16000)
    print(val)