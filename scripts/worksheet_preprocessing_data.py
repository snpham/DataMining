import numpy as np
import pandas as pd
from data_helpers import min_max_normalization
from distances import manhattan_distance, simple_matching_distance



if __name__ == '__main__':


    # worksheet - pre-processing data - park exercise
    data_conv = pd.read_csv('data/park_exercise.csv', index_col=False)
    travel = {'walk': (0,0), 'car': (1,0), 'bike': (0,1), 'bus': (1,1)}
    activities = {'sport': (1, 0, 0, 0, 0, 0), 'picnic': (0, 1, 0, 0, 0, 0),
                  'read': (0, 0, 1, 0, 0, 0), 'walk': (0, 0, 0, 1, 0, 0),
                  'meditate': (0, 0, 0, 0, 1, 0), 'jog': (0, 0, 0, 0, 0, 1)}
    playground = {'Yes': 1, 'No': 0}
    # pre-process travel
    for row in range(0, len(data_conv['Travel'])):
        data_conv.at[row, 'Travel'] = travel[data_conv.at[row, 'Travel']]
    # pre-process activities
    for row in range(0, len(data_conv['Activities'])):
        out_tup = (0, 0, 0, 0, 0, 0)
        for activ in data_conv.at[row, 'Activities'].split(','):
            act = activ.lstrip().rstrip()
            out_tup = tuple([i+j for i, j in zip(out_tup, activities[act])])
            data_conv.at[row, 'Activities'] = out_tup
    # pre-process satisfaction
    for row in range(0, len(data_conv['Satisfaction'])):
        old_min = -2
        old_max = 2
        val = data_conv.loc[row, 'Satisfaction']
        newval = min_max_normalization(old_min=old_min, old_max=old_max, 
                                  new_min=0, new_max=1, single_val=val)
        data_conv.loc[row, 'Satisfaction'] = newval
    # pre-process playground
    for row in range(0, len(data_conv['Playground'])):
        data_conv.at[row, 'Playground'] = playground[data_conv.at[row, 'Playground']]
    print(data_conv)

    # use manhattan distances
    dfdist = pd.DataFrame(index=data_conv['Family'], columns=data_conv['Family'])
    # time
    for row in range(0, len(dfdist.index)):
        for col in range(0, len(dfdist.columns)):
            dfdist.iloc[row, col] = manhattan_distance(x=[data_conv.loc[row, 'Time']], 
                                                       y=[data_conv.loc[col, 'Time']])
    distmax = dfdist.to_numpy().max()
    distmin = dfdist.to_numpy().min()
    for row in range(0, len(dfdist.index)):
        for col in range(0, len(dfdist.columns)):
            val = dfdist.iloc[row, col]
            dfdist.iloc[row, col] = min_max_normalization(old_min=distmin, old_max=distmax, 
                        new_min=0, new_max=1, single_val=val)
    print(dfdist)
    # Travel
    dfdist = pd.DataFrame(index=data_conv['Family'], columns=data_conv['Family'])
    for row in range(0, len(dfdist.index)):
        for col in range(0, len(dfdist.columns)):
            val = 0
            val += manhattan_distance(x=data_conv.loc[row, 'Travel'], y=data_conv.loc[col, 'Travel'])
            dfdist.iloc[row, col] = val
    distmax = dfdist.to_numpy().max()
    distmin = dfdist.to_numpy().min()
    for row in range(0, len(dfdist.index)):
        for col in range(0, len(dfdist.columns)):
            val = dfdist.iloc[row, col]
            dfdist.iloc[row, col] = min_max_normalization(old_min=distmin, old_max=distmax, 
                        new_min=0, new_max=1, single_val=val)
    print(dfdist)
    # Satisfaction
    for row in range(0, len(dfdist.index)):
        for col in range(0, len(dfdist.columns)):
            dfdist.iloc[row, col] = manhattan_distance(x=[data_conv.loc[row, 'Satisfaction']], 
                                                       y=[data_conv.loc[col, 'Satisfaction']])
    print(dfdist)
    # Activities
    dfdist = pd.DataFrame(index=data_conv['Family'], columns=data_conv['Family'])
    for row in range(0, len(dfdist.index)):
        for col in range(0, len(dfdist.columns)):
            val = 0
            val += manhattan_distance(x=data_conv.loc[row, 'Activities'], y=data_conv.loc[col, 'Activities'])
            dfdist.iloc[row, col] = val
    # normalize
    for row in range(0, len(dfdist.index)):
        for col in range(0, len(dfdist.columns)):
            dfdist.iloc[row, col] = simple_matching_distance(list1=data_conv.loc[row, 'Activities'],
                                                             list2=data_conv.loc[col, 'Activities'])
    print(dfdist)
    # Playground
    dfdist = pd.DataFrame(index=data_conv['Family'], columns=data_conv['Family'])
    for row in range(0, len(dfdist.index)):
        for col in range(0, len(dfdist.columns)):
            dfdist.iloc[row, col] = manhattan_distance(x=[data_conv.loc[row, 'Playground']], 
                                                       y=[data_conv.loc[col, 'Playground']])
    distmax = dfdist.to_numpy().max()
    distmin = dfdist.to_numpy().min()
    for row in range(0, len(dfdist.index)):
        for col in range(0, len(dfdist.columns)):
            val = dfdist.iloc[row, col]
            dfdist.iloc[row, col] = min_max_normalization(old_min=distmin, old_max=distmax, 
                        new_min=0, new_max=1, single_val=val)
    print(dfdist)
    