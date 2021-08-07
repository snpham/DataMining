import numpy as np
import pandas as pd


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
        print('class label is binary')
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


def naive_bayesian(df=None, label_dict=None, class_label=None, single=False):
    """
    """
    items = next(iter(class_label.items()))
    df_class = df[df[items[0]] == items[1]]
    if single:
        return len(df_class)/len(df)

    len_i = 1
    for item in label_dict.items():
        len_i *= len(df_class[df_class[item[0]] == item[1]])/len(df[df[items[0]] == items[1]])

    return len_i


def accuracy(df):
    return (df.iloc[0,0]+df.iloc[1,1]) / (df.iloc[2,2])


def precision(df):
    return df.iloc[0,0] / (df.iloc[0,0] + df.iloc[1,0])


def recall(df):
    return df.iloc[0,0] / (df.iloc[0,0] + df.iloc[0,1])


if __name__ == '__main__':


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

    exit()


    split_income = split_info(p_list=[len(incomeset1), len(incomeset2), len(incomeset3)], 
                              Dtot=Dtot)
    assert np.allclose(split_income, 1.556, rtol=1e-2)
    gain_ratio_income = gain_income/split_income
    assert np.allclose(gain_ratio_income, 0.0187, rtol=1e-2)

    # example 8c - naive bayesian
    data = pd.read_csv('data/decisiontree_ex.csv', skipinitialspace=True)

    label_dict = {'age': '<=30'}
    class_label = {'buys_computer': 'yes'}
    len1 = naive_bayesian(df=data, label_dict=label_dict, class_label=class_label)
    assert np.allclose(len1, 0.222, rtol=1e-2)
    label_dict = {'age': '<=30'}
    class_label = {'buys_computer': 'no'}
    len2 = naive_bayesian(df=data, label_dict=label_dict, class_label=class_label)
    assert np.allclose(len2, 0.6, rtol=1e-2)
    label_dict = {'income': 'medium'}
    class_label = {'buys_computer': 'yes'}
    len3 = naive_bayesian(df=data, label_dict=label_dict, class_label=class_label)
    assert np.allclose(len3, 0.444, rtol=1e-2)
    label_dict = {'income': 'medium'}
    class_label = {'buys_computer': 'no'}
    len4 = naive_bayesian(df=data, label_dict=label_dict, class_label=class_label)
    assert np.allclose(len4, 0.4, rtol=1e-2)
    label_dict = {'student': 'yes'}
    class_label = {'buys_computer': 'yes'}
    len5 = naive_bayesian(df=data, label_dict=label_dict, class_label=class_label)
    assert np.allclose(len5, 0.667, rtol=1e-2)
    label_dict = {'student': 'yes'}
    class_label = {'buys_computer': 'no'}
    len6 = naive_bayesian(df=data, label_dict=label_dict, class_label=class_label)
    assert np.allclose(len6, 0.2, rtol=1e-2)
    label_dict = {'credit_rating': 'fair'}
    class_label = {'buys_computer': 'yes'}
    len7 = naive_bayesian(df=data, label_dict=label_dict, class_label=class_label)
    assert np.allclose(len7, 0.667, rtol=1e-2)
    label_dict = {'credit_rating': 'fair'}
    class_label = {'buys_computer': 'no'}
    len8 = naive_bayesian(df=data, label_dict=label_dict, class_label=class_label)
    assert np.allclose(len8, 0.4, rtol=1e-2)

    X_no = {'age': '<=30', 'income': 'medium', 'student': 'yes', 'credit_rating': 'fair'}
    class_label = {'buys_computer': 'yes'}
    PX_yes = naive_bayesian(df=data, label_dict=X_no, class_label=class_label)
    assert np.allclose(PX_yes, 0.0438, rtol=1e-2)
    X_yes = {'age': '<=30', 'income': 'medium', 'student': 'yes', 'credit_rating': 'fair'}
    class_label = {'buys_computer': 'no'}
    PX_no = naive_bayesian(df=data, label_dict=X_yes, class_label=class_label)
    assert np.allclose(PX_no, 0.0192, rtol=1e-2)



    data = pd.read_csv('data/classifier_ex.csv', skipinitialspace=True, index_col=0)
    acc = accuracy(data)
    prec = precision(data)
    rec = recall(data)
    # print(acc, prec, rec)


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
    gain_food = info_D - info_food
    # print(gain_food)



    X_yes = {'food': 'good', 'speedy': 'yes', 'price': 'high'}
    class_label = {'bigtip': 'yes'}
    PX_yes = naive_bayesian(df=data, label_dict=X_yes, class_label=class_label)
    # print(PX_yes)