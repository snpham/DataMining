import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


def data_gen(dataset):
    for row in dataset.iloc[:,0]:
        yield row.replace(" ", "").split(",")


def row_gen(dataset):
    for row in dataset:
        yield row.replace(" ", "").split(",")


def threshold(data, min_sup):
    return {item:count for item, count in data.items() if count >= min_sup}


def union_set(datadict):
    return set([','.join((i, j)) for i in datadict for j in datadict if i < j])


def apriori_assoc(df, datadict1, datadict2, label):
    # support
    splitlabel = label.split(',')
    for key, _ in datadict2.items():
        if set(splitlabel).issubset(key.split(',')):
            labl = key
        else:
            continue
    supp = datadict2[labl]/len(df.index)

    # confidence
    for key1, val1 in datadict1.items():
        if set(splitlabel[:-1]).issubset(key1.split(',')):
            for key2, val2 in datadict2.items():
                if set(splitlabel).issubset(key2.split(',')):
                    conf = val2/val1
    string = f'{splitlabel[:-1]} -> {splitlabel[-1]}'
    return string, supp, conf


def apriori(dataset, min_sup):

    freq1_items = set()
    for data in data_gen(dataset):
        for item in data:
            freq1_items.add(item)
    # print('itemset:', freq1_items)

    scan1 = dict.fromkeys(freq1_items, 0)
    for data in data_gen(dataset):
        for item in data:
            scan1[item] += 1
    scan1 = threshold(scan1, min_sup)
    scan1 = dict(sorted(scan1.items()))

    unionset1 = union_set(scan1)
    scan2 = dict.fromkeys(unionset1, 0)
    for data in data_gen(dataset):
        for row in scan2:
            if set(row.split(',')).issubset(set(data)):
                scan2[row] += 1
    scan2 = threshold(scan2, min_sup)
    scan2 = dict(sorted(scan2.items()))

    unionset2 = union_set(scan2)
    newunion2 = set()
    for row in unionset2:
        newset = set(sorted(row.split(',')))
        for col in row_gen(scan2):
            if set(col).issubset(newset) and scan2[','.join(col)] >= 3 and len(newset) == 3:
                newunion2.add(','.join(newset))
    scan3 = dict.fromkeys(newunion2, 0)

    for data in data_gen(dataset):
        for row in scan3:
            if set(row.split(',')).issubset(set(data)):
                scan3[row] += 1
    scan3 = threshold(scan3, min_sup)
    scan3 = dict(sorted(scan3.items()))

    # print('frequent-1 itemset:', scan1)
    # print('frequent-2 itemset:', scan2)
    # print('frequent-3 itemset:', scan3)
    
    return scan1, scan2, scan3


if __name__ == '__main__':

    dataset = pd.read_csv('data/apriori_ex.csv', index_col=0, header=0)
    min_sup = 0.50
    min_sup *= len(dataset)
    scan1, scan2, scan3 = apriori(dataset, min_sup)
    # itemset: {'E', 'D', 'C', 'A', 'B'}
    # frequent-1 itemset: {'A': 2, 'B': 3, 'C': 3, 'E': 3}
    # frequent-2 itemset: {'A,C': 2, 'B,C': 2, 'B,E': 3, 'C,E': 2}
    # frequent-3 itemset: {'C,B,E': 2}

    dataset = pd.read_csv('data/apriori_hw.csv', index_col=0, header=0)
    min_sup = 0.60
    min_sup *= len(dataset)
    scan1, scan2, scan3 = apriori(dataset, min_sup)
    # itemset: {'Z', 'N', 'B', 'E', 'D', 'I', 'T', 'O', 'G', 'S', 'F'}
    # frequent-1 itemset: {'B': 4, 'E': 3, 'G': 3, 'I': 3, 'N': 4, 'Z': 3}
    # frequent-2 itemset: {'B,I': 3, 'B,N': 4, 'G,Z': 3, 'I,N': 3}
    # frequent-3 itemset: {'I,B,N': 3}

    dataset = pd.read_csv('data/apriori_hw2.csv', index_col=0, header=0)
    min_sup = 0.60
    min_sup *= len(dataset)
    scan1, scan2, scan3 = apriori(dataset, min_sup)
    # itemset: {'pie', 'bread', 'cereal', 'cheese', 'milk', 'cherry'}
    # frequent-1 itemset: {'bread': 4, 'cheese': 3, 'milk': 4, 'pie': 3}
    # frequent-2 itemset: {'bread,cheese': 3, 'bread,milk': 4, 'bread,pie': 3, 'cheese,milk': 3, 'milk,pie': 3}
    # frequent-3 itemset: {'bread,cheese,milk': 3, 'pie,bread,milk': 3}

    # association rules
    tofind = ['bread,milk,pie', 'milk,pie,bread', 'bread,pie,milk', 
              'bread,milk,cheese','cheese,milk,bread', 'bread,cheese,milk']
    for sets in tofind:
        assoc_str, supp, conf = apriori_assoc(df=dataset, datadict1=scan2, 
                                            datadict2=scan3, label=sets)
        # print(assoc_str, 'support:', supp, 'confidence:', conf)
        # ['bread', 'milk'] -> pie support: 0.75 confidence: 0.75
        # ['milk', 'pie'] -> bread support: 0.75 confidence: 1.0
        # ['bread', 'pie'] -> milk support: 0.75 confidence: 1.0
        # ['bread', 'milk'] -> cheese support: 0.75 confidence: 0.75
        # ['cheese', 'milk'] -> bread support: 0.75 confidence: 1.0
        # ['bread', 'cheese'] -> milk support: 0.75 confidence: 1.0

    dataset = pd.read_csv('data/ex63_apriori.csv', index_col=0, header=0)
    min_sup = 0.22
    min_sup *= len(dataset)
    scan1, scan2, scan3 = apriori(dataset, min_sup)
    # frequent-1 itemset: {'I1': 6, 'I2': 7, 'I3': 6, 'I4': 2, 'I5': 2}
    # frequent-2 itemset: {'I1,I2': 4, 'I1,I3': 4, 'I1,I5': 2, 'I2,I3': 4, 'I2,I4': 2, 'I2,I5': 2}
    # frequent-3 itemset: {'I2,I3,I1': 2, 'I2,I5,I1': 2}
    
    dataset = pd.read_csv('data/midterm_frequent.csv', index_col=0, header=0)
    min_sup = 0.50
    min_sup *= len(dataset)
    scan1, scan2, scan3 = apriori(dataset, min_sup)


    ## project - frequent itemsets
    dataset = pd.read_csv('data/integrated_data.csv', index_col=0, header=0).astype('string')
    dataset_prep = dataset[['prep']].dropna()

    # get all countries with frequent-set 1
    dataset_country = dataset[['Residency','prep']].dropna()
    countries = ['US', 'AU', 'DE', 'ES', 'IT', 'JP', 'KR', 'MX', 'SE', 'UK']
    df_countries = []
    minsup_countries = []
    scans1_countries = []
    scans2_countries = []
    scans3_countries = []
    gobars = []
    for ii, country in enumerate(countries):
        df_country = dataset_country[dataset_country['Residency'] == country]
        minsup_country = 0.10 * len(df_country)
        df_country = df_country.drop('Residency', axis=1)
        scans1_country, _, _ = \
            apriori(df_country, minsup_country)
        scans1_country = {key:value/len(df_country) for (key, value) in scans1_country.items()}
        gobars.append(go.Bar(name=country, x=list(scans1_country.keys()), y=list(scans1_country.values()), base=0))
    fig = go.Figure(data=gobars[:])
    # Change the bar mode
    fig.update_layout(barmode='relative')
    update_traces = dict(marker_line_width=1.5, opacity=0.9)
    fig.update_traces(update_traces)
    fig.write_image("outputs/plots/support_countries.pdf")

    # get global frequentset-1
    min_sup = 0.10
    min_sup *= len(dataset_prep)
    print(min_sup)
    scan1, scan2, scan3 = apriori(dataset_prep, min_sup)
    # print('frequent-1 itemset:', scan1)
    # print('frequent-2 itemset:', scan2)
    # print('frequent-3 itemset:', scan3)
    # CN had some erroneous data for PREP

    dfprep = pd.read_csv('data/metadata.csv', index_col=0, 
                         header=None, usecols=range(1,4), encoding='latin1')

    prep_dict = {}
    for val in dfprep.loc['prep', 3].strip().split(','):
        val = val.split('=')
        prep_dict[val[0].strip()] = val[1].strip()
    
    dfprep_desc = pd.DataFrame.from_dict(prep_dict, orient='index', columns=['Description'])
    table = go.Figure(data=[go.Table(
                            columnwidth=[1, 4],
                            header=dict(values=['Activity', 'Description']),
                            cells=dict(values=[list(prep_dict.keys()),
                                               list(prep_dict.values())]))])
    # print(prep_dict)

    # # to use activity description instead
    # prep_merged = {value:scan1[key] for (key, value) in prep_dict.items()}
    # prep_merged = {key:value/len(dataset_prep) for (key, value) in prep_merged.items()}
    # # print(prep_merged)

    update_layout = dict(barmode='group', title='Support of Preparation Activities',
                        yaxis=dict( titlefont_size=16, tickfont_size=16, range=[0.1, 1]),
                        xaxis=dict( tickangle=-45, titlefont_size=16, tickfont_size=16))
    update_traces = dict(marker_color='rgb(55, 83, 109)', marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5, opacity=0.7)

    prep_merged = {key:value/len(dataset_prep) for (key, value) in scan1.items()}

    prep_merged = pd.DataFrame.from_dict(prep_merged, orient='index', columns=['support'])
    # print(prep_merged)
    fig = px.bar(prep_merged, x=prep_merged.index, y='support', color='support',
                 labels={'index': 'activities'})
    fig.update_traces(update_traces)
    fig.update_layout(update_layout)
    fig.write_image("outputs/plots/frequent_itemset1.pdf")
    # fig.update_xaxes(title_font=dict(size=18, family='Courier', color='crimson'))
    # fig.update_yaxes(title_font=dict(size=18, family='Courier', color='crimson'))

    # increase support and view higher frequentsets (2 and 3)
    update_layout = dict(barmode='relative',
                        yaxis=dict(titlefont_size=16, tickfont_size=16, range=[0.4, 0.65]),
                        xaxis=dict(tickangle=-45, titlefont_size=16, tickfont_size=16))
    min_sup = 0.40
    min_sup *= len(dataset_prep)
    print(min_sup)
    scan1, scan2, scan3 = apriori(dataset_prep, min_sup)
    # print('frequent-1 itemset:', scan1)
    # print('frequent-2 itemset:', scan2)
    # print('frequent-3 itemset:', scan3)
    prep2 =  {key:value/len(dataset_prep) for (key, value) in scan2.items()}
    prep2_supp = pd.DataFrame.from_dict(prep2, orient='index', columns=['support'])
    fig = px.bar(prep2_supp, x=prep2_supp.index, y='support', color='support',
                 labels={'index': 'activities'})
    fig.update_traces(update_traces)
    fig.update_layout(update_layout)
    fig.update_layout(title='Support of Preparation Activities - frequentset-2')
    fig.write_image("outputs/plots/frequent_itemset2.pdf")

    prep3 =  {key:value/len(dataset_prep) for (key, value) in scan3.items()}
    prep3_supp = pd.DataFrame.from_dict(prep3, orient='index', columns=['support'])
    fig = px.bar(prep3_supp, x=prep3_supp.index, y='support')
    fig.update_traces(update_traces)
    fig.update_layout(update_layout)
    fig.update_layout(title='Support of Preparation Activities - frequentset-3')
    fig.write_image("outputs/plots/frequent_itemset3.pdf")
