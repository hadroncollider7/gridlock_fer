import mysql.connector
from mysql.connector import Error       # Catches exceptions that may occur during this process.
import yaml
import os

with open('config.yml','r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)
    
def deleteFromTable(id,cursor,connection):
    """This function is only called from the insertIntoTable() function."""
    mysqlQuery = """DELETE FROM FER_Predictions where id={0};""".format(id)
    cursor.execute(mysqlQuery)
    connection.commit()
    print("Record id#{0} successfully deleted from FER_Predictions table".format(id))


def insertIntoTable(id, name, value, filename):
    try:
        connection = mysql.connector.connect(
                                    host = config['mysql']['host'],
                                    database = config['mysql']['database'],
                                    user = config['mysql']['user'],
                                    password = config['mysql']['password'])
        
        if connection.is_connected():
            cursor = connection.cursor()
            deleteFromTable(id,cursor,connection)
                        
            mysqlQuery = """INSERT INTO FER_Predictions (id,name,value,filename)
                            VALUES ({0},'{1}',{2},'{3}');""".format(id,name,value,filename)
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
    """Connect to the MySQL database server, access the FER_Predictions table,
    and retreive records from the table."""
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
            
            # Print the table
            print("\nQuery results (id, name, value, filename):\n")            
            # Loop throught the rows
            for row in record:
                print("{0}\n".format(row))

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("\nMySQL connection is closed")
        