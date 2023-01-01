import mysql.connector
from mysql.connector import Error       # Catches exceptions that may occur during this process.
import yaml
import os

if __name__ == '__main__':
    os.system("cls")

    with open('config.yml','r') as ymlConfigFile:
        config = yaml.safe_load(ymlConfigFile)

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
            record = cursor.fetchone()
            print("You are connected to database: ", record)

            mysqlQuery = "SHOW TABLES;"
            cursor.execute(mysqlQuery)
            record = cursor.fetchall()
            print("\nQuery: '{0}'\n{1}".format(mysqlQuery,record))
            
            # mysqlQuery = "DROP TABLE FER_Predictions;"
            # cursor.execute(mysqlQuery)
            # record = cursor.fetchall()
            # print("\nQuery: '{0}'\n{1}".format(mysqlQuery,record))

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("\nMySQL connection is closed")
        