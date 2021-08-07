import numpy as np
import pandas as pd


def get_metadata():

    data = pd.read_feather('study_data/integrated_data_old.feather')
    data[:][:3] = data[:][:3].replace(r'\s',' ', regex=True)

    with open('study_data/metadata.csv', 'w') as f:
        for ii, column in enumerate(data.columns):
            if column in ['Num1', 'Num2a', 'Num2b', 'Num3']:
                f.writelines(f'{ii},{column},"{data.iloc[0,ii]}"\n')
            else:
                f.writelines(f'{ii},{column},"{data.iloc[1,ii]}","{data.iloc[2,ii]}"\n')


if __name__ == '__main__':

    # get headers for reference
    get_metadata()

    # get a list of attributes and their types
    with open('study_data/attr_types.txt') as f:
        attrs = dict(x.rstrip().split(None, 1) for x in f)

    # read integrated dataset
    data = pd.read_feather('study_data/integrated_data.feather').\
        astype('string').set_index('index')
    # print(data.head())
    # print(data.columns)

    # how many subset of attributes are in each attribute set in the data
    attributes = {'DemGen': 1,
                  'quota_age': 1,
                  'Residency': 1,
                  'GenSocTrust': 1,
                  'Trustingroups': 13,
                  'COVIDexp': 1,
                  'COVIDeffect': 4,
                  'SARS': 1,
                  'CultCog': 6,
                  'prosocial': 1,
                  'CanadaQ': 3,
                  'FinitePool': 8,
                  'Longitude': 3,
                  'Personal': 8,
                  'Friends': 8,
                  'MediaExp': 7,
                  'PosterstrustQ1': 1,
                  'SocialmediatrustQ1': 1,
                  'journalisttrustQ1': 1,
                  'govtrustQ1': 1,
                  'workplacetrustQ1': 1,
                  'FriendstrustQ1': 1,
                  'WHOtrustQ1': 1,
                  'Soughtinfo': 1,
                  'prep': 1,
                  'Govresponse': 9,
                  'Govrestrict': 3,
                  'Sciunderstand': 1,
                  'KnowledgeCert': 2,
                  'Vaccine': 2,
                  'Num1': 1,
                  'Num2a': 1,
                  'Num2b': 1,
                  'Num3': 1,
                  'NumeracyQ1': 1,
                  'DemAge': 1,
                  'DemEdu': 1,
                  'DemHealthcare': 1,
                  'Ethnic min': 1,
                  'Politics': 1,}

    numeric_attrs = ['Num1', 'Num2a', 'Num2b', 'Num3']
    for num in numeric_attrs:
        data[num] = data[num].astype('float64')
    # print(data.info())

    # which cases to we want to run to see pandas' describe() results?
    all_cases = True
    if all_cases: # all cases
        cases2run = data.columns.tolist()
    else: # set single or set of cases here
        cases2run = ['Trustingroups_1', 'COVIDexp']
    data = data[cases2run].copy()

    # run describe() function on nominal/binary/ordinal attributes
    try:
        dfstats = data.describe(exclude=[np.number]).T
        dfstats.index.name = 'attribute'
        dfstats.to_csv('outputs/dfstats.csv')
    except ValueError:
        print('No categorical data')

    # run describe() function on numerical attributes
    try:
        dfstats_num = data.describe(include=[np.number]).T
        dfstats_num.index.name = 'attribute'
        dfstats_num.to_csv('outputs/dfstats_num.csv')
    except ValueError:
        print('No numerical data')
