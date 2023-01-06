# gridlock_fer
The purpose of this repository is to develop the FER component of Project Gridlock.

## Prerequisites
Because the database server is on Rowan's network, and subject to its firewall, it can only be accessed remotely through Rowan's vpn, or while connected through Rowan's Wifi network.

The configuration file is required to access the database.

 Create an `images` folder and put images in there.

### Modules
 - Pytorch = 1.13.0
 - Python = 3.9.15
 - yaml = 6.0
 - mySQL Connector = 8.0.31
 - cuda = 11.7.1


## Making FER predictions
Run `inference.py` to make FER predictions on all image files in the `img_path` filepath.

After inference, the prediction string, value (int), and filename (string) associated with each prediction will be stored in the `FER_Predictions` table in the database server at the `id` index of the table.