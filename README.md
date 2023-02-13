# gridlock_fer
The purpose of this repository is to develop the FER component of Project Gridlock.

## Dependencies
Because the database server is on Rowan's network, and subject to its firewall, it can only be accessed remotely through Rowan's vpn, or while connected through Rowan's Wifi network.

The configuration file is required to access the database.

 Create an `images` folder and put images in there.

### Modules
 - Pytorch = 1.13.0
 - Python = 3.9.15
 - PyYAML = 6.0
 - mySQL Connector = 8.0.31
 - cuda = 11.7.1
 - watchdog = 2.2.1
    - argh = 26.2


## How It Works
Obtain the config file. Modify the source images filepath in the config file. Finally, run `monitor_directory.py` to begin the program. The program monitors the filepath defined in the config file for .jpg file creation events. When a .jpg file is created in the filepath, it performs an inference and uploads data to the mysql database.

### Real-time FER inferences
`webcam.py` uses a webcam to make real-time FER inferences. The inferences are first stored in a que of a defined size. The statistical mode of the que is the FER prediction. IF `uploadToDatabaseServer = True`, then the prediction is uploaded to the mySQl database server after a set amount of "ticks". Simply set `uploadToDatabaseServer = False` to use the webcam to make real-time inferences without using mySQL. 