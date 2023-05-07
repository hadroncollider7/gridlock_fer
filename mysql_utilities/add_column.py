import mysql.connector
from mysql.connector import Error       # Catches exceptions that may occur during this process.
import yaml
import os

with open('config.yml','r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)


try:
    # Connect to the database server.
    connection = mysql.connector.connect(
        host = config['mysql']['host'],
        database = config['mysql']['database'],
        user = config['mysql']['user'],
        password = config['mysql']['password'])
    # Instructions to create the table its columns.
    mySqlQuery = """ALTER TABLE FER_Predictions ADD filename VARCHAR(250) AFTER value;"""
    
    cursor = connection.cursor()
    result = cursor.execute(mySqlQuery)       # Execute the table creation instructions.
    print("Column added successfully ")
    
except mysql.connector.Error as error:
    print("Failed to create table in MySQL: {}".format(error))
finally:
    # Close the connection to the database server.
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")