import os
import yaml
import numpy as np
import pandas as pd

os.system("cls")
with open('public_config.yml', 'r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)

column = config['evaluateInferenceModel']['column']
sheet = config['evaluateInferenceModel']['sheet']
spreadsheet = config['evaluateInferenceModel']['spreadsheet']

data = pd.read_excel(io=spreadsheet,
                     usecols=[column],
                     dtype=np.float16)

# Print some information about the dataframe
# print('datatype: ', type(data))
# print(f'length: {len(data)}')
# print(data.loc[:20])
# print(data.dtypes)
# print('\n')

# Drop NaN rows
data.dropna(inplace=True)
# print('datatype: ', type(data))
# print(f'length: {len(data)}')
# print(data.loc[:20])
# print(data.dtypes)

# Convert dataframe to numpy array
data = data.to_numpy()
data = np.squeeze(data)
print(f'datatype: {type(data)}')
print(f'shape: {np.shape(data)}')
print(data[:20])

# Sum the elements of the array
print('type: ', type(int(np.sum(data))))
print('sum: ', int(np.sum(data)))

print('len type: ', type(len(data)))
print('len: ', len(data))

# Calculate the accuracy
score = int(np.sum(data))/len(data)
print('score type: ', type(score))
print(score)