import yaml

with open('config.yml', 'r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)
    
print(config['mystring'])
print('type: ', type(config['mystring']))