# 2023-RPI-Coprocessor

## SD card preparation for the RPI coprocessor:

1) write WPILibPi image to 16 Gb (> 8 Gb) uSD card, use 64 bit image (WPILibPi_64_image-v2023.2.1.zip)
      see: https://github.com/wpilibsuite/WPILibPi/releases
2) insert SD card in RPi, power up RPi, connect ethernet cable to a network with access to internet (internet access is required to install depthAI!)
3) open up a browser, navigate to wpilibpi.local (laptop and RPi have to be on the same, e.g. a home, network)
4) make file system writable in the webbrowser connected to wpilibpi.local
5) open ssh session to wpilibpi.local, use e.g. PuTTY application, log in with username "pi", password "raspberry"
6) update local date and time on RPi. Date and time have to be correct, otherwise git repository credentials may not work, update and execute following command with the current date and time:
```sh
sudo date -s 'fri mar 3 12:07:23 CST 2023'
```
7) install depthai on RPi:
```sh
sudo apt-get install python3-venv
sudo curl -fL https://docs.luxonis.com/install_dependencies.sh | bash
python3 -m pip install depthai
```
      see:
      https://docs.luxonis.com/projects/api/en/latest/install/
      https://docs.luxonis.com/projects/api/en/latest/install/#raspberry-pi-os
      https://docs.luxonis.com/projects/api/en/latest/install/#install-from-pypi)
8) install or update apriltag library:
```sh
pip install robotpy-apriltag
```
      see:
      https://pypi.org/project/robotpy-apriltag/
9) open winscp session with wpilibpi.local
10) create folder "models" under /home/pi
11) copy neural network models to the models folder (see content in models folder in this github repository)
14) upload application and configure team number, camera, etc in the browser connected to the wpilibpi.local webserver
15) connect Shuffleboard either directly to the RPi (address: wpilibpi.local) or through radio to RoboRIO (address is: 10.9.35.2)

## Depth AI camera neural network training
For the 2023 model training we switched from MobileNetSSD to YOLO-v6 due to better support from the DepthAI community maintaining Google Colab notebooks. The notebook used for the training is located in this github repository. Upload it to a RaileRobotics programming google drive to use it for training.

The images for training a YOLO-v6 model need to be square with a size being a multiple of 32. Tests were done with 320x320 and 416x416 pixel images. The interference speed of the 320x320 is better than the 416x416 but because the same images are also used as input for the AprilTag detection it's better to use the 416x416 resolution. With the OAK-D lite camera connected to a laptop via USB-C it was also possible to get a high resolution camera stream for the AprilTags in parallel to the 320x320 object interference stream. On the RPi with only a USB-3 host port the bandwidth was not sufficient.

The DepthAI community maintains the training notebooks at:
https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks

The image annotation was done using Roboflow. After creating a image set version in Roboflow export it in Yolo V6 format and move the images and annotation around into the folder structure required for the Colab notebook. See the documentation inside the Colab notebook for the exaxt structure. Easiest is to copy the images and annotations to the same Google Drive account as the one used for the Colab notebook so the drive can be mounted and images copied in the Colab scripts.
