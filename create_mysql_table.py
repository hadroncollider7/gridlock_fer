"""This program creates a table in the mySQL database."""
import mysql.connector
import yaml

with open('config.yml','r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)

# The name of the table to be created
tableTitle = 'GridlockFER'

try:
    # Connect to the database server.
    connection = mysql.connector.connect(
        host = config['mysql']['host'],
        database = config['mysql']['database'],
        user = config['mysql']['user'],
        password = config['mysql']['password'])
    # Instructions to create the table its columns.
    mySql_Create_Table_Query = """CREATE TABLE {0:s} (
                            id INT unsigned NOT NULL,
                            name VARCHAR(10) NOT NULL,
                            value INT,
                            PRIMARY KEY (id)) """.format(tableTitle)
    
    cursor = connection.cursor()
    result = cursor.execute(mySql_Create_Table_Query)       # Execute the table creation instructions.
    print("{0:s} Table created successfully ".format(tableTitle))
    
except mysql.connector.Error as error:
    print("Failed to create table in MySQL: {}".format(error))
finally:
    # Close the connection to the database server.
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")