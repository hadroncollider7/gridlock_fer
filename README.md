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


## Making FER predictions
Run `inference.py` to make FER predictions on all image files in the `img_path` filepath.

After inference, the prediction string, value (int), and filename (string) associated with each prediction will be stored in the `FER_Predictions` table in the database server at the `id` index of the table.

### Real-time FER inferences
`webcam.py` uses a webcam to make real-time FER inferences. The inferences are first stored in a que of a set size. The statistical mode of the que is the FER prediction. IF `uploadToDatabaseServer = True`, then the prediction is uploaded to the mySQl database server after a set amount of "ticks". Simply set `uploadToDatabaseServer = False` to use the webcam to make real-time inferences without using mySQL. 