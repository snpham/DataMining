import pandas as pd
import re
import numpy as np


## fixing combined categorial values from prep attribute
dataset = pd.read_csv('data/integrated_data_v2.csv', index_col=0, header=0).astype('string')
prep = dataset['prep']
# print(dataset['prep'].head(10))
for ii, val in enumerate(prep, 1):
    try:
        float(val)
        if 'E+' in val or np.isnan(float(val)):
            continue
        else:
            # print(ii, val)
            number_with_commas = re.sub(r'([0-9])(?!$)(?!0)', r'\1,', val)
            nums = number_with_commas.split(',')
            if nums.count('1') == 2 and len(nums) == 2:
                nums = ['11']
            if nums.count('1') == 3:
                nums = nums[:-2]
                nums.append('11')
            dataset.loc[ii, 'prep'] = ','.join(nums)
    except ValueError:
        continue
    except TypeError:
        continue

# print(dataset['prep'].head(10))
dataset.to_csv('data/integrated_data_v3.csv')
