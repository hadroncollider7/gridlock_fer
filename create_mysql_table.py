"""This program creates a table in the mySQL database."""
import mysql.connector


try:
    # Connect to the database server.
    connection = mysql.connector.connect(
        host='localhost',
        database='',
        user='',
        password='')
    # Instructions to create the table its columns.
    mySql_Create_Table_Query = """CREATE TABLE FER_Predictions(
                            id int(11) NOT NULL,
                            Predicted varchar(250) NOT NULL,
                            PRIMARY KEY (id)) """
    
    cursor = connection.cursor()
    result = cursor.execute(mySql_Create_Table_Query)       # Execute the table creation instructions.
    print("FER Predictions Table created successfully ")
    
except mysql.connector.Error as error:
    print("Failed to create table in MySQL: {}".format(error))
finally:
    # Close the connection to the database server.
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")