# gridlock_fer
The purpose of this repository is to develop the FER component of Project Gridlock.

The configuration file is required to access the database.

## Prerequisites
Because the database server is on Rowan's network, and subject to its firewall, it can only be accessed remotely through Rowan's vpn, or will connected through Rowan's Wifi.

## Making FER predictions
Run `inference.py` to make the FER prediction on the image in the `img_path` filepath. Create an `images` folder and put the image in there.

After inference, the prediction string and value will be stored in the `FER_Predictions` table in the database server at the `id` index of the table.