# 2023-RPI-Coprocessor

How do you prepare an SD card for the RPI coprocessor:

1) write WPILibPi image to 16 Gb uSD card, use 64 bit image (WPILibPi_64_image-v2023.2.1.zip)
      see: https://github.com/wpilibsuite/WPILibPi/releases
2) power up RPi, connect ethernet cable to a network with access to internet (need internet to install depthAI!)
3) open up a browser, navigate to wpilibpi.local, laptop and RPi have to be on the same (home) network
4) make file system writable (in the webbrowser connected to wpilibpi.local)
5) open ssh session to wpilibpi.local (e.g. with PuTTY application, log in with username "pi", password "raspberry")
6) sudo date -s 'fri mar 3 12:07:23 CST 2023'  (update to local date! RPi date/time has to be correct, otherwise repository credentials may not work!)
7) sudo apt-get install python3-venv
      (install depthai, see:
      https://docs.luxonis.com/projects/api/en/latest/install/
      https://docs.luxonis.com/projects/api/en/latest/install/#raspberry-pi-os
      https://docs.luxonis.com/projects/api/en/latest/install/#install-from-pypi)
8) sudo curl -fL https://docs.luxonis.com/install_dependencies.sh | bash
9) python3 -m pip install depthai
10) pip install robotpy-apriltag
      (install robotpy-apriltag, see:
      https://pypi.org/project/robotpy-apriltag/ was already installed, just to check)
11) open winscp session with wpilibpi.local
12) create folder "models" under /home/pi
13) copy neural network models to the models folder (see content in models folder in this github repository)
14) upload application and configure team number, camera, etc in the browser connected to the wpilibpi.local webserver
15) connect Shuffleboard either directly to the RPi (address: wpilibpi.local) or through radio to RoboRIO (address is: 10.9.35.2)

