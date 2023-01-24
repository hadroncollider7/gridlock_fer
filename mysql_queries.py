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


def insertIntoTable(connection, cursor, id, name, value, filename):
    deleteFromTable(id,cursor,connection)       # Delete current row entries at id
    mysqlQuery = """INSERT INTO FER_Predictions (id,name,value,filename)
                    VALUES ({0},'{1}',{2},'{3}');""".format(id,name,value,filename)
    cursor.execute(mysqlQuery)
    connection.commit()
    print("Record successfully inserted into FER_Predictions table")

def insertColumn(connection, cursor, query):
    """Inserts a column into the FER_Predictions table.
    Can also be used to delete a column

    Args:
        connection (mySQL object): _description_
        cursor (mySQL object): _description_
        query (string)): This is the query that the database server will run.
    """
    # mysqlQuery = "ALTER TABLE FER_Predictions ADD created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP AFTER value;"
    cursor.execute(query)
    connection.commit()
    print("Column successfully inserted into table")

            
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

            # # Insert timestamp column into table
            # mysqlQuery = "ALTER TABLE FER_Predictions ADD created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP AFTER value;"
            # insertColumn(connection, cursor, mysqlQuery)
            
            # Show tables
            mysqlQuery = 'SHOW TABLES;'
            cursor.execute(mysqlQuery)
            record = cursor.fetchall()
            print('\nShowing tables:')
            i = 1
            for item in record:
                print('{1}. {0}'.format(item[0], i))
                i += 1
                
            # Describe a table
            mysqlQuery = 'DESCRIBE FER_Predictions;'
            cursor.execute(mysqlQuery)
            record = cursor.fetchall()
            print('\nTable description:')
            for row in record:
                print('{0}'.format(row))
            
            
            
            mysqlQuery = 'SELECT * FROM FER_Predictions;'
            cursor.execute(mysqlQuery)
            record = cursor.fetchall()
            
            # Print the table
            print("\nQuery results:\n(id, name, value, created_at, filename)")            
            # Loop throught the rows
            for row in record:
                print("{0}".format(row))

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("\nMySQL connection is closed")
        