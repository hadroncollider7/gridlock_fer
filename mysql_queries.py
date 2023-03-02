import mysql.connector
from mysql.connector import Error       # Catches exceptions that may occur during this process.
import yaml
import os

with open('config.yml','r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)
    
def deleteFromTable(username, cursor, connection, selectTable):
    """This function is only called from the insertIntoTable() function (Update: not anymore)"""
    mysqlQuery = """DELETE FROM {1:s} where username='{0}';""".format(username, selectTable)
    cursor.execute(mysqlQuery)
    connection.commit()
    print("Record username: {0:s} successfully deleted from FER_Predictions table".format(username))


def insertIntoTable(connection, cursor, username, prediction, valueArgmax, prob, filename):
    """Insert a single entry into the Gridlock_FER table.
    
    Args:
        connection (mysql object)
        cursor (mysql object)
        username (string)
        prediction (string): The emotion label of the argmax of prob
        valueArgmax (int): the class argmax
        prob (list of floats): The softmax associated with the inference
        filename (string)
    """
    selectTable = config['selectTable']
    mysqlQuery = """INSERT INTO {0} (username, predicted, value_argmax, 0_neutral_softmax, 1_happy_softmax, 2_sad_softmax, 3_surprise_softmax, 4_fear_softmax, 5_disgust_softmax, 6_anger_softmax, 7_contempt_softmax, filename)
                    VALUES ('{1}','{2}',{3},'{4}','{5}','{6}','{7}','{8}','{9}','{10}','{11}','{12}')
                    ON DUPLICATE KEY UPDATE
                        predicted = '{2}',
                        value_argmax = '{3}',
                        0_neutral_softmax = '{4}',
                        1_happy_softmax = '{5}',
                        2_sad_softmax = '{6}',
                        3_surprise_softmax = '{7}',
                        4_fear_softmax = '{8}',
                        5_disgust_softmax = '{9}',
                        6_anger_softmax = '{10}',
                        7_contempt_softmax = '{11}';""".format(selectTable,username,prediction,valueArgmax,prob[0],prob[1],prob[2],prob[3],prob[4],prob[5],prob[6],prob[7],filename)
    cursor.execute(mysqlQuery)
    connection.commit()
    print("Record successfully inserted into {0:s} table".format(selectTable))

def insertColumn(connection, cursor, query):
    """Inserts a column into the Gridlock_FER table.
    Can also be used to delete a column

    Args:
        connection (mySQL object): _description_
        cursor (mySQL object): _description_
        query (string)): This is the query that the database server will run.
    """
    # mysqlQuery = "ALTER TABLE FER_Predictions ADD created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP AFTER value;"
    selectTable = config['selectTable']
    cursor.execute(query)
    connection.commit()
    print("Column successfully inserted into table")

            
def main_mysqQueries():
    """Connect to the MySQL database server, access the FER_Predictions table,
    and retreive records from the table."""
    select_table = config['selectTable']
    
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
            
            # *************** DELETE TABLE ENTRIES ***************
            # for i in range(3):
            #     deleteFromTable(i+1,cursor,connection,select_table)

            # **************** INSERT COLUMN INTO TABLE ************************
            # mysqlQuery = "ALTER TABLE {0:s} ADD username VARCHAR(50) NOT NULL AFTER id;".format(select_table)
            # insertColumn(connection, cursor, mysqlQuery)
            
            # ****************** DELETE COLUMN *********************
            # mysqlQuery = "ALTER TABLE {0:s} DROP COLUMN id".format(select_table)
            # cursor.execute(mysqlQuery)
            
            # **************** DROP PRIMARY KEY CONSTRAINT *****************
            # mysqlQuery = "ALTER TABLE {0:s} DROP PRIMARY KEY".format(select_table)
            # cursor.execute(mysqlQuery)
            
            # ***************** CREATE A PRIMARY KEY ****************
            # mysqlQuery = "ALTER TABLE {0:s} ADD PRIMARY KEY (username);".format(select_table)
            # cursor.execute(mysqlQuery)
            
            # Show tables
            mysqlQuery = 'SHOW TABLES;'
            cursor.execute(mysqlQuery)
            record = cursor.fetchall()
            print('\nShowing tables:')
            i = 1
            for item in record:
                print('{1}. {0}'.format(item[0], i))
                i += 1
                
            # Describe the selected table
            mysqlQuery = 'DESCRIBE {0:s};'.format(select_table)
            cursor.execute(mysqlQuery)
            record = cursor.fetchall()
            print('\nTable description:')
            for row in record:
                print('{0}'.format(row))
            
            
            
            mysqlQuery = 'SELECT * FROM {0:s};'.format(select_table)
            cursor.execute(mysqlQuery)
            record = cursor.fetchall()
            
            # Print the table
            print("\nQuery results:")            
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

if __name__ == "__main__":
    main_mysqQueries()