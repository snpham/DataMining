import numpy as np
import pandas as pd
from data_helpers import mean


def chi_squared(df):
    """computes the correlation relationship b/t two nominal attributes
    :param df: pandas dataframe with the observed frequency (actual count)
    contingency table
    :return chi2: chi squared value
    :return dof: degrees of freedom for the given table (for chi2 distribution lookup)
    """

    df['sum_row'] = df.sum(axis=1)
    df.loc["sum_col"] = df.sum()
    nrow = len(df)-1
    ncol = len(df.columns)-1

    e = np.zeros(shape=(nrow, ncol))
    chi2 = 0
    for ii in range(0, nrow):
        for jj in range(0, ncol):
            # i,j expectancy value
            e[ii, jj] = (df['sum_row'][ii]*df.loc['sum_col'][jj]) / df.loc['sum_col', 'sum_row']
            # chi squared value summation
            chi2 += (df.iloc[ii, jj]-e[ii, jj])**2 / e[ii, jj]
    # print(e)
    # print(chi2)
    dof = (nrow-1)*(ncol-1)

    return chi2, dof


def correlation(df=None, fn1=None, fn2=None, attr1=None, attr2=None):
    """correlation coefficient for numeric evaluation of the correlation 
    between two attributes
    :param df: pandas dataframe containing the two numerical attributes and
    objects
    :param fn1: optional csv file input to be used as first attribute dataset
    :param fn2: optional csv file input to be used as second attr. dataset
    :param attr1: name of attribute to analyze for first file
    :param attr2: name of attribute to analyze for second file
    :return corr: correlation coefficient
    """
    if df is None:
        data1 = pd.read_csv(fn1)
        data2 = pd.read_csv(fn2)
        a = data1[attr1].tolist()
        b = data2[attr2].tolist()
        stddeva = data1[attr1].std()
        stddevb = data2[attr2].std()
    else:
        a = df.iloc[:, 0].tolist()
        b = df.iloc[:, 1].tolist()
        stddeva = df.iloc[:, 0].std()
        stddevb = df.iloc[:, 1].std()

    A = mean(a)
    B = mean(b)
    N = len(a)
    corr = sum((x-A)*(y-B) for x, y in zip(a, b))/(N*stddeva*stddevb)
#     corr = (sum((x*y) for x, y in zip(a, b))-N*A*B)/(N*stddeva*stddevb)    
#     corr = data1[attr1].corr(data2[attr2])

    return round(corr, 6)


def covariance(df=None, fn1=None, fn2=None, attr1=None, attr2=None):
    """covariance for numeric evaluation of the correlation between two 
    attributes
    :param df: pandas dataframe containing the two numerical attributes and
    objects
    :param fn1: optional csv file input to be used as dataset for first attr. 
    :param fn2: optional csv file input to be used as dataset for second attr.
    :param attr1: name of attribute to analyze for first file
    :param attr2: name of attribute to analyze for second file
    :return corr: correlation coefficient
    """
    if df is None:
        data1 = pd.read_csv(fn1)
        data2 = pd.read_csv(fn2)
        a = data1[attr1].tolist()
        b = data2[attr2].tolist()
    else:
        a = df.iloc[:, 0].tolist()
        b = df.iloc[:, 1].tolist()

    A = mean(a)
    B = mean(b)
    N = len(a)

    return sum((x-A)*(y-B) for x, y in zip(a, b))/(N)


def support(df, arg1, arg2):
    """
    """
    df['sum_row'] = df.sum(axis=1)
    df.loc["sum_col"] = df.sum()
    nrow = len(df)-1
    ncol = len(df.columns)-1
    try:
        return df.loc[arg1, arg2] / df.loc['sum_col', 'sum_row']
    except KeyError:
        return df.loc[arg2, arg1] / df.loc['sum_col', 'sum_row']


def confidence(df, arg1, arg2):
    """
    """
    df['sum_row'] = df.sum(axis=1)
    df.loc["sum_col"] = df.sum()
    nrow = len(df)-1
    ncol = len(df.columns)-1

    try:
        return df.loc[arg1, arg2] / df.loc[arg1, 'sum_row']
    except KeyError:
        return df.loc[arg2, arg1] / df.loc['sum_col', arg1]


def lift(df, arg1, arg2):
    """
    """
    df['sum_row'] = df.sum(axis=1)
    df.loc["sum_col"] = df.sum()
    nrow = len(df)-1
    ncol = len(df.columns)-1
    total = df.loc['sum_col', 'sum_row']
    # lift = P(AandB)/(P(A)*P(B)) = support(AandB)/(P(A)*P(B))
    try:
        return df.loc[arg1, arg2] / (df.loc[arg1, 'sum_row']*df.loc['sum_col', arg2])*total
    except KeyError:
        return df.loc[arg2, arg1] / (df.loc['sum_col', arg1]*df.loc[arg2, 'sum_row'])*total


if __name__ == '__main__':


    # example 1 - chi2 test
    df_chi2 = pd.DataFrame(index=['LikeSciFi', 'NotLikeSciFi'], 
                        columns=['PlayChess', 'NotPlayChess'])
    df_chi2.loc['LikeSciFi', 'PlayChess'] = 250
    df_chi2.loc['NotLikeSciFi', 'PlayChess'] = 50
    df_chi2.loc['LikeSciFi', 'NotPlayChess'] = 200
    df_chi2.loc['NotLikeSciFi', 'NotPlayChess'] = 1000
    # print(df_chi2)
    chi2, dof = chi_squared(df_chi2)
    assert np.allclose(chi2, 507.93650)
    # DOF = (r-1)(c-1) = 1 -> chi2 to reject at 0.001 = 10.828

    # example 2 - correlation test
    df = pd.read_csv('data/correlation_sample.csv', index_col=0)
    cov = covariance(df)
    assert np.allclose(cov, 7.0)

    # example 3 - support and confidence
    df_sup = pd.DataFrame(index=['LikeSki', 'NotLikeSki'], 
                        columns=['PlayFootball', 'NotPlayFootball'])
    df_sup.loc['LikeSki', 'PlayFootball'] = 1700
    df_sup.loc['NotLikeSki', 'PlayFootball'] = 400
    df_sup.loc['LikeSki', 'NotPlayFootball'] = 800
    df_sup.loc['NotLikeSki', 'NotPlayFootball'] = 1000
    # LikeSki -> PlayFootball
    sup = support(df_sup.copy(), 'LikeSki', 'PlayFootball')
    assert np.allclose(sup, 0.435897435)
    conf = confidence(df_sup.copy(), 'LikeSki', 'PlayFootball')
    assert np.allclose(conf, 0.68)
    lft = lift(df_sup.copy(), 'LikeSki', 'PlayFootball')
    assert np.allclose(lft, 1.2628571428571)

    # example 6d - lift
    df_sup = pd.DataFrame(index=['C', 'NotC'], 
                        columns=['B', 'NotB'])
    df_sup.loc['C', 'B'] = 400
    df_sup.loc['NotC', 'B'] = 200
    df_sup.loc['C', 'NotB'] = 350
    df_sup.loc['NotC', 'NotB'] = 50
    # B -> C
    lft = lift(df_sup.copy(), 'B', 'C')
    assert np.allclose(lft, 0.888, rtol=1e-2)
    lft = lift(df_sup.copy(), 'B', 'NotC')
    assert np.allclose(lft, 1.33, rtol=1e-2)

    # example 6d - chi2 test 2
    chi2, dof = chi_squared(df_sup)
    assert np.allclose(chi2, 55.56, rtol=1e-2)
    # DOF = (r-1)(c-1) = 1 -> chi2 to reject at 0.001 = 10.828

    # example 6d - support and confidence and lift
    df_sup = pd.DataFrame(index=['Coffee', 'NotCoffee'], 
                        columns=['Tea', 'NotTea'])
    df_sup.loc['Coffee', 'Tea'] = 150
    df_sup.loc['NotCoffee', 'Tea'] = 50
    df_sup.loc['Coffee', 'NotTea'] = 750
    df_sup.loc['NotCoffee', 'NotTea'] = 50
    # Tea -> Coffee
    sup = support(df_sup.copy(), 'Tea', 'Coffee')
    assert np.allclose(sup, 0.15)
    conf = confidence(df_sup.copy(), 'Tea', 'Coffee')
    assert np.allclose(conf, 0.75, rtol=1e-2)
    lft = lift(df_sup.copy(), 'Tea', 'Coffee')
    assert np.allclose(lft, 0.833, rtol=1e-2)

    # example 6d - lift vs chi2
    df_sup = pd.DataFrame(index=['cereal', 'Notcereal'], 
                        columns=['basketball', 'Notbasketball'])
    df_sup.loc['cereal', 'basketball'] = 2000
    df_sup.loc['Notcereal', 'basketball'] = 1000
    df_sup.loc['cereal', 'Notbasketball'] = 1750
    df_sup.loc['Notcereal', 'Notbasketball'] = 250
    # cereal -> basketball
    lft = lift(df_sup.copy(), 'cereal', 'basketball')
    assert np.allclose(lft, 0.89, rtol=1e-2) # negatively corr.
    lft = lift(df_sup.copy(), 'basketball', 'Notcereal')
    assert np.allclose(lft, 1.33, rtol=1e-2) # positively corr.
    chi2, dof = chi_squared(df_sup)
    assert np.allclose(chi2, 277.78, rtol=1e-2) # corr. since > 10.828
    

    ## LIFT AND CHI2
    df_sup = pd.DataFrame(index=['bread', 'Notbread'], 
                        columns=['orange', 'Notorange'])
    df_sup.loc['bread', 'orange'] = 900
    df_sup.loc['Notbread', 'orange'] = 450
    df_sup.loc['bread', 'Notorange'] = 750
    df_sup.loc['Notbread', 'Notorange'] = 900
    # bread -> orange
    lft = lift(df_sup.copy(), 'bread', 'orange')
    # lft = lift(df_sup.copy(), 'orange', 'Notbread')
    chi2, dof = chi_squared(df_sup)
    print(lft, chi2)
    
    