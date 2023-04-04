import os
import yaml
import numpy as np
import pandas as pd


def load_validation_predictions(column, spreadsheet):
    data = pd.read_excel(io=spreadsheet,
                        usecols=[column],
                        dtype=np.float16)

    # Drop NaN rows
    data.dropna(inplace=True)
    # Convert dataframe to numpy array
    data = data.to_numpy()
    data = np.squeeze(data)
    return data

os.system("cls")
with open('public_config.yml', 'r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)

# Load spreadsheet parameters
column = config['evaluateInferenceModel']['column']
spreadsheet = config['evaluateInferenceModel']['spreadsheet']

data = load_validation_predictions(column, spreadsheet)

# Calculate the accuracy
score = int(np.sum(data))/len(data)
# Print score
print("Score for {0}: {1}".format(spreadsheet, score))