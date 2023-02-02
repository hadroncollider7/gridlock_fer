"""This program creates a table in the mySQL database."""
import mysql.connector
import yaml

with open('config.yml','r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)

tableTitle = "Gridlock_FER"

try:
    # Connect to the database server.
    connection = mysql.connector.connect(
        host = config['mysql']['host'],
        database = config['mysql']['database'],
        user = config['mysql']['user'],
        password = config['mysql']['password'])
    # Instructions to create the table its columns.
    mySql_Create_Table_Query = """CREATE TABLE Gridlock_FER (
                            id INT unsigned NOT NULL,
                            predicted VARCHAR(10) NOT NULL,
                            value_argmax INT unsigned NOT NULL,
                            0_neutral_softmax DECIMAL(4,4) NOT NULL,
                            1_happy_softmax DECIMAL(4,4) NOT NULL,
                            2_sad_softmax DECIMAL(4,4) NOT NULL,
                            3_surprise_softmax DECIMAL(4,4) NOT NULL,
                            4_fear_softmax DECIMAL(4,4) NOT NULL,
                            5_disgust_softmax DECIMAL(4,4) NOT NULL,
                            6_anger_softmax DECIMAL(4,4) NOT NULL,
                            7_contempt_softmax DECIMAL(4,4) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (id)) """
    
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