import mysql.connector
from mysql.connector import Error       # Catches exceptions that may occur during this process.
import yaml
import os

with open('config.yml','r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)

def insertIntoTable(id, name, value):
    try:
        connection = mysql.connector.connect(
                                    host = config['mysql']['host'],
                                    database = config['mysql']['database'],
                                    user = config['mysql']['user'],
                                    password = config['mysql']['password'])
        
        if connection.is_connected():
            cursor = connection.cursor()
            mysqlQuery = """INSERT INTO FER_Predictions (id,name,value)
                            VALUES ({0},'{1}',{2})""".format(id,name,value)
            cursor.execute(mysqlQuery)
            connection.commit()
            print("Record successfully inserted into FER_Predictions table")

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("\nMySQL connection is closed")
            
if __name__ == '__main__':
    os.system("cls")
    try:
        connection = mysql.connector.connect(
                                    host = config['mysql']['host'],
                                    database = config['mysql']['database'],
                                    user = config['mysql']['user'],
                                    password = config['mysql']['password'])
        
        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchall()
            print("You are connected to database: ", record)
            
            mysqlQuery = 'SELECT * FROM FER_Predictions;'
            cursor.execute(mysqlQuery)
            record = cursor.fetchall()
            print("Query results:\n", record)

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("\nMySQL connection is closed")
        