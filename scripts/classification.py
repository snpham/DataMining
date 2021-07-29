import numpy as np
import pandas as pd


def entropy(p_list):
    """info(D)
    gets the entropy from the list of classes
    """
    p_list = [i for i in p_list if i != 0]

    return sum(-p_i*np.log(p_i)/np.log(2) for p_i in p_list)


def entropy_split(p_list, attribute):
    """
    """
    pass


def info_subset(lens1, lens2):
    """
    """
    info_i = 0
    for set1, set1y in zip(lens1,lens2):
        leny = set1y
        lenn = set1 - leny
        info_i += (leny+lenn)/Dtot*\
            entropy(p_list=[leny/(leny+lenn), lenn/(leny+lenn)])

    return info_i


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


    # example 8b - entropy
    D_total = 30
    p_list = [16/D_total, 14/D_total]
    info_D = entropy(p_list=p_list)
    assert np.allclose(info_D, 0.99, rtol=1e-2)

    # example 8b - info gain
    class1 = 9
    class2 = 5
    D_total = sum((class1, class2))
    p_list = [class1/D_total, class2/D_total]
    info_D = entropy(p_list=p_list)
    assert np.allclose(info_D, 0.940, rtol=1e-2)

    info_age = 5/14*entropy([2/5,3/5]) + 4/14*entropy([4/4,0/4]) + 5/14*entropy([3/5,2/5])
    assert np.allclose(info_age, 0.694, rtol=1e-2)
    gain_age = info_D - info_age
    assert np.allclose(gain_age, 0.246, rtol=1e-2)

    data = pd.read_csv('data/decisiontree_ex.csv', skipinitialspace=True)

    # info D
    Dy = len(data[data['buys_computer'] == 'yes'])
    Dn = len(data[data['buys_computer'] == 'no'])
    Dtot = Dy + Dn
    infoD =  entropy(p_list=[Dy/Dtot, Dn/Dtot])
    assert np.allclose(infoD, 0.940, rtol=1e-2)

    # info age
    ageset1 = data[data['age'] == '<=30']
    ageset1yes = ageset1[ageset1['buys_computer'] == 'yes' ]
    ageset2 = data[data['age'] =='31-40']
    ageset2yes = ageset2[ageset2['buys_computer'] == 'yes' ]
    ageset3 = data[data['age'] =='>40']
    ageset3yes = ageset3[ageset3['buys_computer'] == 'yes' ]
    lens_age = [len(ageset1), len(ageset2), len(ageset3)]
    lens_agey = [len(ageset1yes), len(ageset2yes), len(ageset3yes)]
    info_age = info_subset(lens1=lens_age, lens2=lens_agey)
    gain_age = info_D - info_age

    # info income
    incomeset1 = data[data['income'] == 'high']
    incomeset1yes = incomeset1[incomeset1['buys_computer'] == 'yes' ]
    incomeset2 = data[data['income'] =='medium']
    incomeset2yes = incomeset2[incomeset2['buys_computer'] == 'yes' ]
    incomeset3 = data[data['income'] =='low']
    incomeset3yes = incomeset3[incomeset3['buys_computer'] == 'yes' ]
    lens_income = [len(incomeset1), len(incomeset2), len(incomeset3)]
    lens_incomey = [len(incomeset1yes), len(incomeset2yes), len(incomeset3yes)]
    info_income = info_subset(lens1=lens_income, lens2=lens_incomey)
    gain_income = info_D - info_income

    # info student
    studentset1 = data[data['student'] == 'yes']
    studentset1yes = studentset1[studentset1['buys_computer'] == 'yes' ]
    studentset2 = data[data['student'] =='no']
    studentset2yes = studentset2[studentset2['buys_computer'] == 'yes' ]
    lens_student = [len(studentset1), len(studentset2)]
    lens_studenty = [len(studentset1yes), len(studentset2yes)]
    info_student = info_subset(lens1=lens_student, lens2=lens_studenty)
    gain_student = info_D - info_student

    # info credit_rating
    credit_ratingset1 = data[data['credit_rating'] == 'excellent']
    credit_ratingset1yes = credit_ratingset1[credit_ratingset1['buys_computer'] == 'yes' ]
    credit_ratingset2 = data[data['credit_rating'] =='fair']
    credit_ratingset2yes = credit_ratingset2[credit_ratingset2['buys_computer'] == 'yes' ]
    lens_credit_rating = [len(credit_ratingset1), len(credit_ratingset2)]
    lens_credit_ratingy = [len(credit_ratingset1yes), len(credit_ratingset2yes)]
    info_credit_rating = info_subset(lens1=lens_credit_rating, lens2=lens_credit_ratingy)
    gain_credit_rating = info_D - info_credit_rating

    assert np.allclose(gain_age, 0.246, rtol=1e-2)
    assert np.allclose(gain_income, 0.029, rtol=1e-2)
    assert np.allclose(gain_student, 0.151, rtol=1e-2)
    assert np.allclose(gain_credit_rating, 0.048, rtol=1e-2)

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
    print(gain_food)



    X_yes = {'food': 'good', 'speedy': 'yes', 'price': 'high'}
    class_label = {'bigtip': 'yes'}
    PX_yes = naive_bayesian(df=data, label_dict=X_yes, class_label=class_label)
    print(PX_yes)