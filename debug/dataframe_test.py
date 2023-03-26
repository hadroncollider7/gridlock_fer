import os
import yaml
import pandas as pd

os.system("cls")
with open('public_config.yml', 'r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)

batch_no = config['evaluateInferenceModel']['batch_no']
column = config['evaluateInferenceModel']['column']
sheet = config['evaluateInferenceModel']['sheet']

if batch_no == 2:
    img_path = 'images/batch2/'
    spreadsheet = 'Batch2_Labels.xlsx'
elif batch_no == 3:
    img_path = 'images/batch3/0-995/'
    spreadsheet = 'Batch3_Labels.xlsx'
elif batch_no == 4:
    img_path = 'images/batch4/0-822/'
    spreadsheet = 'Batch4_Labels.xlsx'
elif batch_no == 1:
    img_path = 'images/batch1/0-999/'
    spreadsheet = 'Batch1_Labels.xlsx'

data = pd.read_excel(io=spreadsheet,
                     sheet_name=[sheet],
                     usecols=['Image', column])

print(type(list(data[sheet]['Image'])))