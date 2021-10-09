import numpy as np
import pandas as pd
import plotly.graph_objs as go


def entropy(p_list):
    """info(D)
    gets the entropy from the list of classes
    """
    p_list = [i for i in p_list if i != 0]

    return sum(-p_i*np.log(p_i)/np.log(2) for p_i in p_list)


def info_D(df_data, allclasslabels):

    key = list(allclasslabels.keys())[0]
    labels = list(allclasslabels.values())[0]

    if len(labels) == 2:
        # print('class label is binary')
        D1 = len(df_data[df_data[key] == labels[0]])
        D2 = len(df_data[df_data[key] == labels[1]])
        Dtot = sum((D1, D2))
    elif len(labels) > 2:
        print('continuous trees not implemented')

    return entropy(p_list=[D1/Dtot, D2/Dtot])


def info_subset(lens1, lens2):
    """get info if class C_i in a data partition; same as info_Ci_D
    but uses lengths of tuples in class D and C_i_D
    :param lens1: list of tuple counts in different classes in D
    :param lens2: list of tuple counts matching the desired 
    class label
    in work
    """
    Dtot = sum((lens1))

    info_i = 0
    for set1, set1y in zip(lens1,lens2):
        leny = set1y
        lenn = set1 - leny
        info_i += (leny+lenn)/Dtot*\
            entropy(p_list=[leny/(leny+lenn), lenn/(leny+lenn)])

    return info_i


def info_gain(info_D, info_Ci_D):
    """information gain is defined as the difference between the
    original information requirement (based on just the proportion
    of classes) and the new requirement (obtained after partitioning
    of i)
    """
    return info_D - info_Ci_D


def info_Ci_D(df_data, setlabels, classlabel):
    """find the information gain of a partition i from a dictionary of
    class labels
    """
    df_lists = []
    df_classes = []
    df_count = []
    df_classcount = []
    for v in list(setlabels.values())[0]:
        df = df_data[df_data[list(setlabels.keys())[0]] == v]
        df_class = df[df[list(classlabel.keys())[0]] == list(classlabel.values())[0]]
        df_lists.append(df)
        df_classes.append(df_class)
        df_count.append(len(df))
        df_classcount.append(len(df_class))
        # print(data[data[list(setlabels.keys())[0]] == v])
    info_age = info_subset(lens1=df_count, lens2=df_classcount)

    return info_age


def get_info_gain(df, setlabels, classlabel, allclasslabels):
    """
    """
    info_d = info_D(df_data=df, allclasslabels=allclasslabels)
    info_i = info_Ci_D(df, setlabels, classlabel)
    gain_i = info_gain(info_d, info_i)
    return gain_i


def split_info(p_list, Dtot):
    """
    """
    p_list = [i for i in p_list if i != 0]

    return sum(-p_i/Dtot*np.log(p_i/Dtot)/np.log(2) for p_i in p_list)


def gain_ratio(df_data, setlabels, gain):
    """find the information gain of a partition i from a dictionary of
    class labels
    """
    df_count = []
    for v in list(setlabels.values())[0]:
        df = df_data[df_data[list(setlabels.keys())[0]] == v]
        df_count.append(len(df))
    split = split_info(p_list=df_count, Dtot=sum(df_count))

    return gain/split


def naive_bayesian(df=None, label_dict=None, class_label=None, single=False):
    """
    """
    items = next(iter(class_label.items()))
    df_class = df[df[items[0]] == items[1]]
    if single:
        return len(df_class)/len(df)

    len_i = 1
    for item in label_dict.items():
        len_i *= len(df_class[df_class[item[0]] == item[1]])/ \
            len(df[df[items[0]] == items[1]])

    return len_i


def accuracy(df):
    return (df.iloc[0,0]+df.iloc[1,1]) / (df.iloc[2,2])


def precision(df):
    return df.iloc[0,0] / (df.iloc[0,0] + df.iloc[1,0])


def recall(df):
    return df.iloc[0,0] / (df.iloc[0,0] + df.iloc[0,1])


def gini_D(len1, len2):
    """math function for computing gini index
    """
    Dtot = sum((len1, len2))
    p_list=[len1/Dtot, len2/Dtot]
    p_list = [i for i in p_list if i != 0]

    return 1 - sum((p_i)**2 for p_i in p_list)


def gini_index(df_data, allclasslabels):
    """finds the gini index for a class within a dataset
    """
    key = list(allclasslabels.keys())[0]
    labels = list(allclasslabels.values())[0]

    if len(labels) == 2:
        # print('class label is binary')
        D1 = len(df_data[df_data[key] == labels[0]])
        D2 = len(df_data[df_data[key] == labels[1]])
        Dtot = sum((D1, D2))
    elif len(labels) > 2:
        print('continuous trees not implemented')
    gini_d = gini_D(len1=D1, len2=D2)

    return gini_d


def gini_subset(df_data, setlabels, classlabel):
    """return the gini index for the subset 
    """
    Dtot = len(df_data)
    df_subset1 = df_data[df_data[list(setlabels.keys())[0]].\
        isin(list(setlabels.values())[0])]
    df_subset2 = df_data[np.logical_not(df_data[list(setlabels.keys())[0]].\
        isin(list(setlabels.values())[0]))]

    len1_subset1 = len(df_subset1[df_subset1[list(classlabel.keys())[0]] == \
        list(classlabel.values())[0]])
    len2_subset1 = len(df_subset1) - len1_subset1

    len1_subset2 = len(df_subset2[df_subset2[list(classlabel.keys())[0]] == \
        list(classlabel.values())[0]])
    len2_subset2 = len(df_subset2) - len1_subset2

    len_class_subset1 = [len1_subset1, len2_subset1]
    len_class_subset2 = [len1_subset2, len2_subset2]

    gini_i = 0
    for sets in [len_class_subset1, len_class_subset2]:
        gini_i += sum(sets)/Dtot*gini_D(len1=sets[0], len2=sets[1])

    return gini_i





def project_classification():
    ## project
    get_gain_table = False
    # finding gains for split
    if get_info_gain:
        data = pd.read_csv('data/integrated_data_v3.csv', skipinitialspace=True, 
                            header=0, dtype = str)
        data = data.dropna( how='any', subset=['Vaccine_1', 'Vaccine_2', 'CanadaQ_1'])
        classify_meta = pd.read_csv('data/classify_meta.csv', skipinitialspace=True, 
                            header=0)
        classify_meta = classify_meta.drop(classify_meta.columns[0], axis=1)
        unique = dict()
        uniques_list = []
        uniques_dict = {}
        classify_meta = classify_meta.astype('object')
        for ii, att in enumerate(classify_meta['index']):
            unique[att] = [x for x in list(data[att].unique()) if str(x) != 'nan']
            uniques_list.append(unique[att])
            uniques_dict[att] = unique[att]
        classify_meta["possible_values"] = uniques_list
        classify_meta.to_csv('data/classify_meta_mod.csv')
        # infomation gain table
        classlabels1 = {'Vaccine_1': ('1.0', '2.0')}
        classlabel1 = {'Vaccine_1': '1.0'}
        attribute1 = classify_meta['index']
        attribute1_vals = classify_meta['possible_values']
        df_gain = pd.DataFrame(columns=['attribute', 'gain'])
        for att, values in zip(attribute1, attribute1_vals):
            att_dict = {}
            att_dict[att] = values
            gain1 = get_info_gain(df=data, setlabels=att_dict, classlabel=classlabel1, 
                                allclasslabels=classlabels1)
            # print(att, round(gain1, 4))
            df_gain = df_gain.append({'attribute': att, 'gain': gain1}, True)
        df_gain = df_gain.sort_values(by='gain', ascending=False)
        df_gain.to_csv('data/gain_summary.csv')




    df = pd.read_csv('data/gain_summary.csv')
    df = df_gain.head(6)
    df = df.append(df_gain.tail(6))
    # print(df)
    gain_table = go.Figure(data=[go.Table(
                    header=dict(values=list(df.columns),
                                align='left'),
                    cells=dict(values=[df.attribute, df.gain.round(4)],
                            fill_color='lavender',
                            align='left'))])    
    # gain_table.write_image('outputs/plots/gain_table.pdf')
    # gain_table.show()

    # # splitting on PosterstrustQ1
    # count = 0
    # countries = data['Residency'].unique()
    # df_attr = classify_meta[classify_meta['index'] == 'PosterstrustQ1']
    # class_label_yes = {'Vaccine_1': '1.0'}
    # class_label_no = {'Vaccine_1': '2.0'}

    # df_PosterstrustQ1 = data[['PosterstrustQ1', 'Vaccine_1']]
    # data_PosterstrustQ1 = pd.DataFrame()
    # for idx in sorted(df_attr.possible_values.tolist()[0]):
    #     P_yes = len(df_PosterstrustQ1[df_PosterstrustQ1['Vaccine_1'] == '1.0'])/\
    #         len(df_PosterstrustQ1['Vaccine_1'])
    #     P_no = len(df_PosterstrustQ1[df_PosterstrustQ1['Vaccine_1'] == '2.0'])/\
    #         len(df_PosterstrustQ1['Vaccine_1'])
    #     person_X = {'PosterstrustQ1': idx}
    #     PX_yes = naive_bayesian(df=df_PosterstrustQ1, label_dict=person_X, 
    #                             class_label=class_label_yes)
    #     PX_no = naive_bayesian(df=df_PosterstrustQ1, label_dict=person_X, 
    #                             class_label=class_label_no)
    #     PX_yes_P_yes = PX_yes * P_yes
    #     PX_no_P_no = PX_no * P_no
    #     data_PosterstrustQ1.loc[count, 'PosterstrustQ1'] = idx
    #     data_PosterstrustQ1.loc[count, 'Vaccine-Yes, by PostersTrustQ1'] = \
    #         f'{PX_yes_P_yes:.3f}'
    #     data_PosterstrustQ1.loc[count, 'Vaccine-No, by PostersTrustQ1'] = \
    #         f'{PX_no_P_no:.3f}'
    #     count += 1

    # print(data_PosterstrustQ1)
    # exit()





    columns = ['Residency', 'PosterstrustQ1', 'quota_age', 'Vaccine_1', 'Vaccine_2', 'CanadaQ_1']
    data = pd.read_csv('data/integrated_data_v3.csv', skipinitialspace=True, 
                        header=0, dtype = str, usecols=columns)
    data = data.dropna( how='any', subset=['Vaccine_1', 'Vaccine_2', 'CanadaQ_1'])
    quota_ages = pd.read_csv('data/metadata.csv', index_col=0, 
                             header=0, encoding='cp1252')
    quota_ages = dict([x.replace(" ", "").split('=') \
                        for x in quota_ages.iloc[2,2].split(',')])

    count = 0
    data_vaccine_poster = pd.DataFrame()
    data_vaccine_global = pd.DataFrame()
    countries = data['Residency'].unique()
    for ii, country in enumerate(countries):
        for idx, age in zip([1, 2, 3, 4, 5, 6, 7], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]):

            data_country = data[data['Residency'] == country]
            idx = str(float(idx))
            person_X = {'PosterstrustQ1': idx, 'Residency': country}
            data_vaccine_poster.loc[count, 'Residency'] = country
            class_label_yes = {'Vaccine_1': '1.0'}
            class_label_no = {'Vaccine_1': '2.0'}
            PX_yes = naive_bayesian(df=data_country, label_dict=person_X, 
                                    class_label=class_label_yes)
            PX_no = naive_bayesian(df=data_country, label_dict=person_X, 
                                   class_label=class_label_no)
            P_yes = len(data_country[data_country['Vaccine_1'] == '1.0'])/\
                len(data_country['Vaccine_1'])
            P_no = len(data_country[data_country['Vaccine_1'] == '2.0'])/\
                len(data_country['Vaccine_1'])
            if idx == '1.0':
                data_vaccine_global.loc[ii, 'Residency'] = country
                data_vaccine_global.loc[ii, 'Vaccine-Yes'] = \
                f'{P_yes:.3f}'
                data_vaccine_global.loc[ii, 'Vaccine-No'] = \
                f'{P_no:.3f}'
            # print(P_yes, P_no)
            PX_yes_P_yes = PX_yes * P_yes
            PX_no_P_no = PX_no * P_no
            data_vaccine_poster.loc[count, 'PosterstrustQ1_val'] = age

            data_vaccine_poster.loc[count, 'Vaccine-Yes, by Country+PosterstrustQ1'] = \
                f'{PX_yes_P_yes:.3f}'
            data_vaccine_poster.loc[count, 'Vaccine-No, by Country+PosterstrustQ1'] = \
                f'{PX_no_P_no:.3f}'
            # recommend
            class_label_yes = {'Vaccine_2': '1.0'}
            class_label_no = {'Vaccine_2': '2.0'}
            PX_yes = naive_bayesian(df=data_country, label_dict=person_X, 
                                    class_label=class_label_yes)
            PX_no = naive_bayesian(df=data_country, label_dict=person_X, 
                                    class_label=class_label_no)
            P_yes = len(data_country[data_country['Vaccine_2'] == '1.0'])/ \
                len(data_country['Vaccine_2'])
            P_no = len(data_country[data_country['Vaccine_2'] == '2.0'])/ \
                len(data_country['Vaccine_2'])
            if idx == '1.0':
                data_vaccine_global.loc[ii, 'Recommend-Yes'] = \
                f'{P_yes:.3f}'
                data_vaccine_global.loc[ii, 'Recommend-No'] = \
                f'{P_no:.3f}'
            # print(P_yes, P_no)
            PX_yes_P_yes = PX_yes * P_yes
            PX_no_P_no = PX_no * P_no
            data_vaccine_poster.loc[count, 'PosterstrustQ1_val'] = age
            data_vaccine_poster.loc[count, 'Recommend-Yes, by Country+PosterstrustQ1'] = \
                f'{PX_yes_P_yes:.3f}'
            data_vaccine_poster.loc[count, 'Recommend-No, by Country+PosterstrustQ1'] = \
                f'{PX_no_P_no:.3f}'
            count += 1

    data_vaccine_poster.to_csv('data/naive_bayesian_vaccine_PosterstrustQ1_per_country.csv')
    print(data_vaccine_poster)
    data_vaccine_global.to_csv('data/naive_bayesian_vaccine_PosterstrustQ1_global.csv')
    print(data_vaccine_global)

    data_vaccine_poster_global = pd.DataFrame()
    for idx, age in zip([1, 2, 3, 4, 5, 6, 7], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]):
        idx = str(float(idx))
        person_X = {'PosterstrustQ1': idx}
        class_label_yes = {'Vaccine_1': '1.0'}
        class_label_no = {'Vaccine_1': '2.0'}
        PX_yes = naive_bayesian(df=data, label_dict=person_X, class_label=class_label_yes)
        PX_no = naive_bayesian(df=data, label_dict=person_X, class_label=class_label_no)
        P_yes = len(data[data['Vaccine_1'] == '1.0'])/len(data['Vaccine_1'])
        P_no = len(data[data['Vaccine_1'] == '2.0'])/len(data['Vaccine_1'])
        # print(P_yes, P_no)
        PX_yes_P_yes = PX_yes * P_yes
        PX_no_P_no = PX_no * P_no
        data_vaccine_poster_global.loc[count, 'PosterstrustQ1_val'] = age
        data_vaccine_poster_global.loc[count, 'Vaccine-Yes, by PosterstrustQ1_val'] = f'{PX_yes_P_yes:.3f}'
        data_vaccine_poster_global.loc[count, 'Vaccine-No, by PosterstrustQ1_val'] = f'{PX_no_P_no:.3f}'
        # recommend
        class_label_yes = {'Vaccine_2': '1.0'}
        class_label_no = {'Vaccine_2': '2.0'}
        PX_yes = naive_bayesian(df=data, label_dict=person_X, class_label=class_label_yes)
        PX_no = naive_bayesian(df=data, label_dict=person_X, class_label=class_label_no)
        P_yes = len(data[data['Vaccine_2'] == '1.0'])/len(data['Vaccine_2'])
        P_no = len(data[data['Vaccine_2'] == '2.0'])/len(data['Vaccine_2'])
        # print(P_yes, P_no)
        PX_yes_P_yes = PX_yes * P_yes
        PX_no_P_no = PX_no * P_no
        data_vaccine_poster_global.loc[count, 'PosterstrustQ1_val'] = age
        data_vaccine_poster_global.loc[count, 'Recommend-Yes, by PosterstrustQ1_val'] = f'{PX_yes_P_yes:.3f}'
        data_vaccine_poster_global.loc[count, 'Recommend-No, by PosterstrustQ1_val'] = f'{PX_no_P_no:.3f}'
        count += 1

    data_vaccine_poster_global.to_csv('data/naive_bayesian_vaccine_PosterstrustQ1_global.csv')
    print(data_vaccine_poster_global)

    data.loc[(data['CanadaQ_1'] == "1.0"), "CanadaQ_1"] = "1.0"
    data.loc[(data['CanadaQ_1'] == "2.0"), "CanadaQ_1"] = "1.0"
    data.loc[(data['CanadaQ_1'] == "3.0"), "CanadaQ_1"] = "1.0"
    data.loc[(data['CanadaQ_1'] == "4.0"), "CanadaQ_1"] = "2.0"
    data.loc[(data['CanadaQ_1'] == "5.0"), "CanadaQ_1"] = "2.0"

    data_serious = pd.DataFrame()
    for idx, age in zip([1, 2, 3, 4, 5, 6, 7], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]):
        idx = str(float(idx))
        person_X = {'PosterstrustQ1': idx}
        class_label_yes = {'CanadaQ_1': '2.0'}
        class_label_no = {'CanadaQ_1': '1.0'}
        PX_yes = naive_bayesian(df=data, label_dict=person_X, class_label=class_label_yes)
        PX_no = naive_bayesian(df=data, label_dict=person_X, class_label=class_label_no)
        P_yes = len(data[data['CanadaQ_1'] == '2.0'])/len(data['CanadaQ_1'])
        P_no = len(data[data['CanadaQ_1'] == '1.0'])/len(data['CanadaQ_1'])
        # print(P_yes, P_no)
        PX_yes_P_yes = PX_yes * P_yes
        PX_no_P_no = PX_no * P_no
        data_serious.loc[idx, 'PosterstrustQ1_val'] = age
        data_serious.loc[idx, 'Serious-Yes Probability'] = f'{PX_yes_P_yes:.3f}'
        data_serious.loc[idx, 'Serious-No Probability'] = f'{PX_no_P_no:.3f}'
    data_serious.to_csv('data/naive_bayesian_canada1_global.csv')
    print(data_serious)


def tests_datamining():

    # example 8b - using entropy
    D_total_8b = 30
    p_list_8b = [16/D_total_8b, 14/D_total_8b]
    infoD_8b = entropy(p_list=p_list_8b)
    assert np.allclose(infoD_8b, 0.99, rtol=1e-2)

    # example 8b - using entropy and finding info gains manually
    class_yes_8b = 9
    class_no_8b = 5
    D_total_8b = sum((class_yes_8b, class_no_8b))
    p_list_8b = [class_yes_8b/D_total_8b, class_no_8b/D_total_8b]
    infoD_8b2 = entropy(p_list=p_list_8b)
    info_age_8b2 = 5/14*entropy([2/5,3/5]) + 4/14*entropy([4/4,0/4]) \
             + 5/14*entropy([3/5,2/5])
    gain_age_8b2 = infoD_8b2 - info_age_8b2
    assert np.allclose(infoD_8b2, 0.940, rtol=1e-2)
    assert np.allclose(info_age_8b2, 0.694, rtol=1e-2)
    assert np.allclose(gain_age_8b2, 0.246, rtol=1e-2)

    # example 8b - using entropy and finding info gains automatically
    info_age_8b3 = info_subset(lens1=[5,4,5], lens2=[2,4,3])
    gain_age_8b3 = info_gain(infoD_8b2, info_age_8b3)
    assert np.allclose([info_age_8b2, gain_age_8b2], [info_age_8b3, gain_age_8b3])


    ## another decision tree example
    data_ex = pd.read_csv('data/decisiontree_ex.csv', skipinitialspace=True)

    # ex-1: info D for decisiontree_ex.csv example by entropy
    D_len1_ex1 = len(data_ex[data_ex['buys_computer'] == 'yes'])
    D_len2_ex1 = len(data_ex[data_ex['buys_computer'] == 'no'])
    D_total_ex1 = sum((D_len1_ex1, D_len2_ex1))
    infoD_ex1a =  entropy(p_list=[D_len1_ex1/D_total_ex1, D_len2_ex1/D_total_ex1])
    assert np.allclose(infoD_ex1a, 0.940, rtol=1e-2)
    # info D by dictionary function
    classlabels_ex = {'buys_computer': ('yes', 'no')}
    infoD_ex1b = info_D(df_data=data_ex, allclasslabels=classlabels_ex)
    assert np.allclose(infoD_ex1a, infoD_ex1b, rtol=1e-2)

    # ex-2: age info and gain using functions
    classlabels_ex2 = classlabels_ex
    ages_ex2 = {'age': ('<=30', '31-40', '>40')}
    ages_class_ex2 = {'buys_computer': 'yes'}
    infoD_ex2 = infoD_ex1b
    info_ages_ex2 = info_Ci_D(df_data=data_ex, setlabels=ages_ex2, classlabel=ages_class_ex2)
    gain_ages_ex2 = info_gain(infoD_ex2, info_ages_ex2)
    assert np.allclose(gain_ages_ex2, 0.246, rtol=1e-2)
    # testing all-in-one function
    gain_ages_ex2b = get_info_gain(df=data_ex, setlabels=ages_ex2, classlabel=ages_class_ex2, 
                             allclasslabels=classlabels_ex2)
    assert np.allclose(gain_ages_ex2b, 0.246, rtol=1e-2)

    # ex-3: age info and gain manually using infoD from above
    infoD_ex3 = infoD_ex1b
    ageset1_ex3 = data_ex[data_ex['age'] == '<=30']
    ageset1_ex3yes = ageset1_ex3[ageset1_ex3['buys_computer'] == 'yes' ]
    ageset2_ex3 = data_ex[data_ex['age'] =='31-40']
    ageset2_ex3yes = ageset2_ex3[ageset2_ex3['buys_computer'] == 'yes' ]
    ageset3_ex3 = data_ex[data_ex['age'] =='>40']
    ageset3_ex3yes = ageset3_ex3[ageset3_ex3['buys_computer'] == 'yes' ]
    lens_age_ex3 = [len(ageset1_ex3), len(ageset2_ex3), len(ageset3_ex3)]
    lens_age_ex3y = [len(ageset1_ex3yes), len(ageset2_ex3yes), len(ageset3_ex3yes)]
    info_age_ex3 = info_subset(lens1=lens_age_ex3, lens2=lens_age_ex3y)
    gain_age_ex3 = infoD_ex3 - info_age_ex3
    assert np.allclose(gain_age_ex3, gain_ages_ex2b, rtol=1e-2)

    # ex-4: age info and gain manually using infoD from above
    infoD_ex4 = infoD_ex1b
    income1_ex4 = data_ex[data_ex['income'] == 'high']
    income1_ex4yes = income1_ex4[income1_ex4['buys_computer'] == 'yes' ]
    income2_ex4 = data_ex[data_ex['income'] =='medium']
    income2_ex4yes = income2_ex4[income2_ex4['buys_computer'] == 'yes' ]
    income3_ex4 = data_ex[data_ex['income'] =='low']
    income3_ex4yes = income3_ex4[income3_ex4['buys_computer'] == 'yes' ]
    lens_income_ex4 = [len(income1_ex4), len(income2_ex4), len(income3_ex4)]
    lens_income_ex4y = [len(income1_ex4yes), len(income2_ex4yes), len(income3_ex4yes)]
    info_income_ex4 = info_subset(lens1=lens_income_ex4, lens2=lens_income_ex4y)
    gain_income_ex4 = infoD_ex4 - info_income_ex4
    assert np.allclose(gain_income_ex4, 0.029, rtol=1e-2)

    # ex-5: income info and gain using functions
    classlabels_ex5 = classlabels_ex
    income_ex5 = {'income': ('high', 'medium', 'low')}
    income_class_ex5 = {'buys_computer': 'yes'}
    infoD_ex5 = infoD_ex1b
    info_income_ex5 = info_Ci_D(df_data=data_ex, setlabels=income_ex5, classlabel=income_class_ex5)
    gain_income_ex5 = info_gain(infoD_ex5, info_income_ex5)
    assert np.allclose(gain_income_ex5, gain_income_ex4, rtol=1e-2)
    # testing all-in-one function
    gain_income_ex5b = get_info_gain(df=data_ex, setlabels=income_ex5, classlabel=income_class_ex5, 
                             allclasslabels=classlabels_ex5)
    assert np.allclose(gain_income_ex5b, gain_income_ex4, rtol=1e-2)

    # ex-6: student info and gain manually using infoD from above
    student1_ex6 = data_ex[data_ex['student'] == 'yes']
    student1_ex6yes = student1_ex6[student1_ex6['buys_computer'] == 'yes' ]
    student2_ex6 = data_ex[data_ex['student'] =='no']
    student2_ex6yes = student2_ex6[student2_ex6['buys_computer'] == 'yes' ]
    lens_student_ex6 = [len(student1_ex6), len(student2_ex6)]
    lens_student_ex6y = [len(student1_ex6yes), len(student2_ex6yes)]
    info_student_ex6 = info_subset(lens1=lens_student_ex6, lens2=lens_student_ex6y)
    gain_student_ex6 = infoD_ex1b - info_student_ex6
    assert np.allclose(gain_student_ex6, 0.151, rtol=1e-2)

    # ex-7: student info and gain using functions
    classlabels_ex7 = classlabels_ex
    student_ex7 = {'student': ('yes', 'no')}
    student_class_ex7 = {'buys_computer': 'yes'}
    infoD_ex7 = infoD_ex1b
    info_student_ex7 = info_Ci_D(df_data=data_ex, setlabels=student_ex7, classlabel=student_class_ex7)
    gain_student_ex7 = info_gain(infoD_ex7, info_student_ex7)
    assert np.allclose(gain_student_ex7, gain_student_ex6, rtol=1e-2)
    # testing all-in-one function
    gain_student_ex7b = get_info_gain(df=data_ex, setlabels=student_ex7, classlabel=student_class_ex7, 
                             allclasslabels=classlabels_ex7)
    assert np.allclose(gain_student_ex7b, gain_student_ex7, rtol=1e-2)

    # ex-8: credit rating info and gain manually using infoD from above
    credit1_ex8 = data_ex[data_ex['credit_rating'] == 'excellent']
    credit1_ex8yes = credit1_ex8[credit1_ex8['buys_computer'] == 'yes' ]
    credit2_ex8 = data_ex[data_ex['credit_rating'] =='fair']
    credit2_ex8yes = credit2_ex8[credit2_ex8['buys_computer'] == 'yes' ]
    lens_credit_ex8 = [len(credit1_ex8), len(credit2_ex8)]
    lens_credit_ex8y = [len(credit1_ex8yes), len(credit2_ex8yes)]
    info_credit_ex8 = info_subset(lens1=lens_credit_ex8, lens2=lens_credit_ex8y)
    gain_credit_ex8 = infoD_ex1b - info_credit_ex8
    assert np.allclose(gain_credit_ex8, 0.048, rtol=1e-2)

    # ex-9: credit rating info and gain using functions
    classlabels_ex9 = classlabels_ex
    credit_ex9 = {'credit_rating': ('excellent', 'fair')}
    credit_class_ex9 = {'buys_computer': 'yes'}
    infoD_ex9 = infoD_ex1b
    info_credit_ex9 = info_Ci_D(df_data=data_ex, setlabels=credit_ex9, classlabel=credit_class_ex9)
    gain_credit_ex9 = info_gain(infoD_ex9, info_credit_ex9)
    assert np.allclose(gain_credit_ex9, gain_credit_ex8, rtol=1e-2)
    # testing all-in-one function
    gain_credit_ex9b = get_info_gain(df=data_ex, setlabels=credit_ex9, classlabel=credit_class_ex9, 
                             allclasslabels=classlabels_ex9)
    assert np.allclose(gain_credit_ex9b, gain_credit_ex9, rtol=1e-2)


    # ex-10: gain ratio for attribute selection
    lens_incomes_ex10 = lens_income_ex4
    split_income_ex10 = split_info(p_list=lens_incomes_ex10, Dtot=D_total_ex1)
    assert np.allclose(split_income_ex10, 1.556, rtol=1e-2)
    gain_ratio_income_ex10 = gain_income_ex5b/split_income_ex10
    assert np.allclose(gain_ratio_income_ex10, 0.0187, rtol=1e-2)

    # ex-11: gain ratio for attribute selection
    income_ex11 = {'income': ('high', 'medium', 'low')}
    gain_ratio_income11 = gain_ratio(df_data=data_ex, setlabels=income_ex11, gain=gain_income_ex5b)
    assert np.allclose(gain_ratio_income11, gain_ratio_income_ex10, rtol=1e-2)


    # ex-12 - decision tree worksheet
    # print('>> Decision Tree Worksheet')
    data_ex12 = pd.read_csv('data/decisiontree_ex.csv', skipinitialspace=True)
    allclasslabels_ex12 = {'buys_computer': ('yes', 'no')}
    infoD_WS = info_D(df_data=data_ex12, allclasslabels=allclasslabels_ex12)
    # print(f'info_D = {infoD_WS:.3f}')
    for age in data_ex12['age'].unique():
        total = data_ex12[data_ex12['age'] == age]
        total_yes = total[total['buys_computer'] == 'yes']
        entrop = entropy(p_list= \
            [len(total_yes)/len(total), (len(total)-len(total_yes))/len(total)])
        # print(f'group: {age}: , entropy: {entrop:.3f}')
    classlabel_ex12 = {'buys_computer': 'yes'}
    setlabels = {'age': data_ex12['age'].unique()}
    info_age_ex12 = info_Ci_D(data_ex12, setlabels, classlabel_ex12)
    # print(f'info_age: {info_age_ex12:.3f}')
    gain_age_ex12 = info_gain(infoD_WS, info_age_ex12)
    # print(f'Gain(age): {gain_age_ex12:.3f}')
    gain_income_ex12 = get_info_gain(df=data_ex12, 
                                     setlabels={'income': data_ex12['income'].unique()}, 
                                     classlabel={'buys_computer': 'yes'}, 
                                     allclasslabels={'buys_computer': \
                                         data_ex12['buys_computer'].unique()})
    # print(f'Gain(income): {gain_income_ex12:.3f}')
    gain_student_ex12 = get_info_gain(df=data_ex12, 
                                     setlabels={'student': data_ex12['student'].unique()}, 
                                     classlabel={'buys_computer': 'yes'}, 
                                     allclasslabels={'buys_computer': \
                                         data_ex12['buys_computer'].unique()})
    # print(f'Gain(student): {gain_student_ex12:.3f}')
    gain_credit_ex12 = get_info_gain(df=data_ex12, 
                                     setlabels={'credit_rating': \
                                         data_ex12['credit_rating'].unique()}, 
                                     classlabel={'buys_computer': 'yes'}, 
                                     allclasslabels={'buys_computer': \
                                         data_ex12['buys_computer'].unique()})
    # print(f'Gain(credit): {gain_credit_ex12:.3f}')
    assert np.allclose([infoD_WS, gain_age_ex12, gain_income_ex12, 
                            gain_student_ex12, gain_credit_ex12],
                        [0.940, 0.247, 0.029, 0.152, 0.048], rtol=1e-2)

    # ex-13 gini index
    gini_d_ex13 = gini_index(df_data=data_ex12, 
        allclasslabels={'buys_computer': data_ex12['buys_computer'].unique()})
    assert np.allclose(gini_d_ex13, 0.459, rtol=1e-2) 
    
    gini_inc_ex13 = gini_subset(df_data=data_ex12, 
                                setlabels={'income': ['low', 'medium']}, 
                                classlabel={'buys_computer': 'yes'})
    assert np.allclose(gini_inc_ex13, 0.4428, rtol=1e-3) 
    gini_inc2_ex13 = gini_subset(df_data=data_ex12, 
                                setlabels={'income': ['low', 'high']}, 
                                classlabel={'buys_computer': 'yes'})
    assert np.allclose(gini_inc2_ex13, 0.4583, rtol=1e-3) 
    gini_inc3_ex13 = gini_subset(df_data=data_ex12, 
                                setlabels={'income': ['medium', 'high']}, 
                                classlabel={'buys_computer': 'yes'})
    assert np.allclose(gini_inc3_ex13, 0.450, rtol=1e-3) 

    del_gini_inc1_ex13 = round(gini_d_ex13 - gini_inc_ex13, 4)
    del_gini_inc2_ex13 = round(gini_d_ex13 - gini_inc2_ex13, 4)
    del_gini_inc3_ex13 = round(gini_d_ex13 - gini_inc3_ex13, 4)
    # print(del_gini_inc1_ex13, del_gini_inc2_ex13, del_gini_inc3_ex13)

    gini_age_ex13 = gini_subset(df_data=data_ex12, 
                                setlabels={'age': [' <=30', '31-40']}, 
                                classlabel={'buys_computer': 'yes'})
    assert np.allclose(gini_age_ex13, 0.3571, rtol=1e-3) 
    # assert np.allclose(gini_age_ex13, 0.4428, rtol=1e-3) 
    gini_age2_ex13 = gini_subset(df_data=data_ex12, 
                                setlabels={'age': [' <=30', '>40']}, 
                                classlabel={'buys_computer': 'yes'})
    # print(gini_age2_ex13)
    # assert np.allclose(gini_age2_ex13, 0.4583, rtol=1e-3) 
    gini_age3_ex13 = gini_subset(df_data=data_ex12, 
                                setlabels={'age': ['31-40', '>40']}, 
                                classlabel={'buys_computer': 'yes'})
    # print(gini_age3_ex13)
    del_gini_age1_ex13 = round(gini_d_ex13 - gini_age_ex13, 4)
    del_gini_age2_ex13 = round(gini_d_ex13 - gini_age2_ex13, 4)
    del_gini_age3_ex13 = round(gini_d_ex13 - gini_age3_ex13, 4)
    # print(del_gini_age1_ex13, del_gini_age2_ex13, del_gini_age3_ex13)

    gini_student_ex13 = gini_subset(df_data=data_ex12, 
                                setlabels={'student': ['yes']}, 
                                classlabel={'buys_computer': 'yes'})
    assert np.allclose(gini_student_ex13, 0.3673, rtol=1e-3) 

    gini_credit_ex13 = gini_subset(df_data=data_ex12, 
                                setlabels={'credit_rating': ['fair']}, 
                                classlabel={'buys_computer': 'yes'})
    assert np.allclose(gini_credit_ex13, 0.4285, rtol=1e-3) 




    # example 8c - naive bayesian
    data = pd.read_csv('data/decisiontree_ex.csv', skipinitialspace=True)
    
    P_yes = len(data[data['buys_computer'] == 'yes'])/len(data['buys_computer'])
    P_no = len(data[data['buys_computer'] == 'no'])/len(data['buys_computer'])
    # print(P_yes, P_no)
    class_label_yes = {'buys_computer': 'yes'}
    label_dict_age = {'age': '<=30'}
    label_dict_income = {'income': 'medium'}
    label_dict_student = {'student': 'yes'}
    label_dict_credit = {'credit_rating': 'fair'}
    bayesian_age_yes = naive_bayesian(df=data, label_dict=label_dict_age, 
                                      class_label=class_label_yes)
    bayesian_income_yes = naive_bayesian(df=data, label_dict=label_dict_income, 
                                         class_label=class_label_yes)
    bayesian_student_yes = naive_bayesian(df=data, label_dict=label_dict_student, 
                                          class_label=class_label_yes)
    bayesian_credit_yes = naive_bayesian(df=data, label_dict=label_dict_credit, 
                                         class_label=class_label_yes)
    assert np.allclose([bayesian_age_yes, bayesian_income_yes, 
                        bayesian_student_yes, bayesian_credit_yes],
                        [0.222, 0.444, 0.667, 0.667], rtol=1e-2)
    class_label_no = {'buys_computer': 'no'}
    bayesian_age_no = naive_bayesian(df=data, label_dict=label_dict_age, 
                                     class_label=class_label_no)
    bayesian_income_no = naive_bayesian(df=data, label_dict=label_dict_income, 
                                        class_label=class_label_no)
    bayesian_student_no = naive_bayesian(df=data, label_dict=label_dict_student, 
                                         class_label=class_label_no)
    bayesian_credit_no = naive_bayesian(df=data, label_dict=label_dict_credit, 
                                        class_label=class_label_no)
    assert np.allclose([bayesian_age_no, bayesian_income_no, 
                        bayesian_student_no, bayesian_credit_no],
                        [0.6, 0.4, 0.2, 0.4], rtol=1e-2)

    person_X = {'age': '<=30', 'income': 'medium', 'student': 'yes', 'credit_rating': 'fair'}
    class_label_yes = {'buys_computer': 'yes'}
    class_label_no = {'buys_computer': 'no'}
    PX_yes = naive_bayesian(df=data, label_dict=person_X, class_label=class_label_yes)
    PX_no = naive_bayesian(df=data, label_dict=person_X, class_label=class_label_no)
    assert np.allclose([PX_yes, PX_no], [0.0438, 0.0192], rtol=1e-2)

    # yes probability for class X 
    PX_yes_P_yes = PX_yes * P_yes
    PX_no_P_no = PX_no * P_no
    assert np.allclose([PX_yes_P_yes, PX_no_P_no], [0.02822, 0.00686], rtol=1e-2)
    # print(f'PX_yes * P_yes: {PX_yes_P_yes:.4f}')
    # print(f'PX_no * P_no: {PX_no_P_no:.4f}')

    data = pd.read_csv('data/classifier_ex.csv', skipinitialspace=True, index_col=0)
    acc = accuracy(data)
    prec = precision(data)
    rec = recall(data)
    assert np.allclose([acc, prec, rec], [0.965, 0.3913, 0.3], rtol=1e-2)

    ## PRECISION, ACCURACY, RECALL
    data2 = pd.read_csv('data/midterm_classifier.csv', skipinitialspace=True, index_col=0)
    acc = accuracy(data2)
    prec = precision(data2)
    rec = recall(data2)
    # print(acc*100, prec*100, rec*100)


    ## INFO GAIN, BAYESIAN
    data = pd.read_csv('data/midterm_info.csv', skipinitialspace=True)

    # info D
    Dy = len(data[data['bigtip'] == 'yes'])
    Dn = len(data[data['bigtip'] == 'no'])
    Dtot = Dy + Dn
    infoD =  entropy(p_list=[Dy/Dtot, Dn/Dtot])

    # info food
    foodset1 = data[data['food'] == 'good']
    foodset1yes = foodset1[foodset1['bigtip'] == 'yes' ]
    foodset2 = data[data['food'] =='mediocre']
    foodset2yes = foodset2[foodset2['bigtip'] == 'yes' ]
    foodset3 = data[data['food'] =='yikes']
    foodset3yes = foodset3[foodset3['bigtip'] == 'yes' ]
    lens_food = [len(foodset1), len(foodset2), len(foodset3)]
    lens_foody = [len(foodset1yes), len(foodset2yes), len(foodset3yes)]
    info_food = info_subset(lens1=lens_food, lens2=lens_foody)
    gain_food = infoD - info_food
    # print(gain_food)

    X_yes = {'food': 'good', 'speedy': 'yes', 'price': 'high'}
    class_label = {'bigtip': 'yes'}
    PX_yes = naive_bayesian(df=data, label_dict=X_yes, class_label=class_label)
    # print(PX_yes)


if __name__ == '__main__':


    tests_datamining()
    
    # csci 5622 problem set 2
    data_ex = pd.read_csv('data/decisiontree_5622.csv', skipinitialspace=True)
    classlabels =  {'college_degree': ('Yes', 'No')}

    # info D by dictionary function
    infoD = info_D(df_data=data_ex, allclasslabels=classlabels)

    res = {'residency': ('Yes', 'No')}
    res_class = {'college_degree': 'yes'}
    info_res = info_Ci_D(df_data=data_ex, setlabels=res, classlabel=res_class)
    gain_res = info_gain(infoD, info_res)
    print(gain_res)
    # testing all-in-one function
    gain_resb = get_info_gain(df=data_ex, setlabels=res, classlabel=res_class, 
                             allclasslabels=classlabels)
    print(gain_resb)
