#!/usr/bin/env python3
#
# based on: https://docs.luxonis.com/projects/api/en/v2.1.0.0/samples/26_1_spatial_mobilenet/
# and: https://docs.luxonis.com/projects/api/en/latest/samples/ObjectTracker/spatial_object_tracker/#spatial-object-tracker-on-rgb
# updated to work on FRC 2022 WPILibPi image and to be uploaded as a vision application
# communicating with shuffleboard and RoboRIO through NetworkTables and CameraServer
# Jaap van Bergeijk, 2022

# First Created by Jaap van Bergijk during frc 2022 build season
# updated in 2023 to include apriltag processing 
# based on: https://www.chiefdelphi.com/t/using-apriltag-on-raspberry-pi/423250/15
# Erik Kaufman, 2023

# from operator import truediv
from pathlib import Path
from cscore import CameraServer
from ntcore import NetworkTableInstance

import json
import time
import sys
import cv2
import depthai as dai
import numpy as np

# Apriltag
import robotpy_apriltag
from wpimath.geometry import Transform3d
import math

configFile = "/boot/frc.json"

team = None
server = False

def parseError(str):
    """Report parse error."""
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

def readConfig():
    """Read configuration file."""
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))


    return True

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTableInstance.getDefault()
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClient4("wpilibpi")
        ntinst.setServerTeam(team)
        ntinst.startDSClient()
    sd = ntinst.getTable("Vision")
    
    # MobilenetSSD label texts
    labelMap = ["background", "blue ball", "person", "red ball", "robot"]

    syncNN = True
    streamDepth = False

    # Static setting for model BLOB, this runs on a RPi with a RO filesystem
    nnBlobPath = str((Path(__file__).parent / Path('models/frc2022_openvino_2021.4_5shave_20220118.blob')).resolve().absolute())

    # Create a CameraServer for ShuffleBoard visualization
    cs = CameraServer
    cs.enableLogging()

    # Width and Height have to match Neural Network settings: 300x300 pixels
    width = 300
    height = 300
    output_stream_front_nn = cs.putVideo("FrontNN", width, height)
    if streamDepth:
        output_stream_front_depth = cs.putVideo("FrontDepth", width, height)

    # Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
    pipeline = dai.Pipeline()

    #First, we want the Color camera as the output
#    colorCam = pipeline.createColorCamera()
    colorCam = pipeline.create(dai.node.ColorCamera)
    manip = pipeline.create(dai.node.ImageManip)
    spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    objectTracker = pipeline.createObjectTracker()

    xoutRgb = pipeline.createXLinkOut()
    if streamDepth:
        xoutDepth = pipeline.createXLinkOut()
    xoutTracker = pipeline.createXLinkOut()

    xoutRgb.setStreamName("rgb")
    if streamDepth:
        xoutDepth.setStreamName("depth")
    xoutTracker.setStreamName("tracklets")

    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    colorCam.setInterleaved(False)
    colorCam.setIspScale(1,5) # 4056x3040 -> 812x608
    colorCam.setPreviewSize(812, 608)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    colorCam.setFps(25)

    # Use ImageManip to resize to 300x300 with letterboxing: enables a wider FOV
    manip.setMaxOutputFrameSize(270000) # 300x300x3
    manip.initialConfig.setResizeThumbnail(300, 300)
    colorCam.preview.link(manip.inputImage)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    spatialDetectionNetwork.setBlobPath(nnBlobPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(15000)

    objectTracker.setDetectionLabelsToTrack([1, 2, 3, 4])  # track red balls, blue balls, persons and robots
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    manip.out.link(spatialDetectionNetwork.input)
    objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
    objectTracker.out.link(xoutTracker.input)

    if (syncNN):
        spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
    else:
        manip.out.link(xoutRgb.input)

    spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    spatialDetectionNetwork.out.link(objectTracker.inputDetections)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    # Pipeline is now finished, and we need to find an available device to run our pipeline
    # we are using context manager here that will dispose the device after we stop using it
    #with dai.Device(pipeline, True) as device:
    with dai.Device(pipeline) as device:
        # From this point, the Device will be in "running" mode and will start sending data via XLink

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=6, blocking=False)
        if streamDepth:
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        trackletsQueue = device.getOutputQueue(name="tracklets", maxSize=6, blocking=False)

        # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
        frame = None
        detections = []

#        startTime = time.monotonic()
#        counter = 0
#        fps = 0
        color = (255, 255, 255)
        image_output_bandwidth_limit_counter = 0

        prev_time = time.time()
        while True:
            start_time = time.time()
            inPreview = previewQueue.get()
            track = trackletsQueue.get()

#            counter+=1
#            current_time = time.monotonic()
#            if (current_time - startTime) > 1 :
#                fps = counter / (current_time - startTime)
#                counter = 0
#                startTime = current_time

            frame = inPreview.getCvFrame()
            if streamDepth:
                depth = depthQueue.get()
                depthFrame = depth.getFrame()
                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
            
            #---------------------------------------
            # Apriltag Processing
            #---------------------------------------

            # Coordinates of found targets, for NT output:
            x_list = []
            y_list = []
            id_list = []

            # April Tag detection:
            detector = robotpy_apriltag.AprilTagDetector()
            detector.addFamily("tag16h5")
            estimator = robotpy_apriltag.AprilTagPoseEstimator(
                robotpy_apriltag.AprilTagPoseEstimator.Config(
                    0.2, 500, 500, width / 2.0, height / 2.0
                )
            )

            # Detect apriltag
            DETECTION_MARGIN_THRESHOLD = 100
            DETECTION_ITERATIONS = 50

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tag_info = detector.detect(gray)
            filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]

            # OPTIONAL: Ignore any tags not in the set used on the 2023 FRC field:
            #filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 9))]

            for tag in filter_tags:

                est = estimator.estimateOrthogonalIteration(tag, DETECTION_ITERATIONS)
                pose = est.pose1

                tag_id = tag.getId()
                center = tag.getCenter()
                #hamming = tag.getHamming()
                #decision_margin = tag.getDecisionMargin()
                print(f"{tag_id}: {pose}")

                # Highlight the edges of all recognized tags and label them with their IDs:

                if ((tag_id > 0) & (tag_id < 9)):
                    col_box = (0,255,0)
                    col_txt = (255,255,255)
                else:
                    col_box = (0,0,255)
                    col_txt = (0,255,255)

                # Draw a frame around the tag:
                corner0 = (int(tag.getCorner(0).x), int(tag.getCorner(0).y))
                corner1 = (int(tag.getCorner(1).x), int(tag.getCorner(1).y))
                corner2 = (int(tag.getCorner(2).x), int(tag.getCorner(2).y))
                corner3 = (int(tag.getCorner(3).x), int(tag.getCorner(3).y))
                cv2.line(frame, corner0, corner1, color = col_box, thickness = 1)
                cv2.line(frame, corner1, corner2, color = col_box, thickness = 1)
                cv2.line(frame, corner2, corner3, color = col_box, thickness = 1)
                cv2.line(frame, corner3, corner0, color = col_box, thickness = 1)

                # Label the tag with the ID:
                cv2.putText(frame, f"{tag_id}", (int(center.x), int(center.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col_txt)

                x_list.append((center.x - width / 2) / (width / 2))
                y_list.append((center.y - width / 2) / (width / 2))
                id_list.append(tag_id)
            
            sd.putNumberArray('target_x', x_list)
            sd.putNumberArray('target_y', y_list)
            sd.putNumberArray('target_id', id_list)


            #---------------------------------------
            # Object Dedetion
            #---------------------------------------
            
            trackletsData = track.tracklets
            for t in trackletsData:
                roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)

                try:
                    label = labelMap[t.label]
                except:
                    label = t.label

                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)

                ssd=sd.getSubTable(f"FrontCam/Object[{t.id}]")

                ssd.putString("Label", str(label))
                ssd.putString("Status", t.status.name)
    #            sd.putNumber("Confidence", int(detection.confidence * 100))
                ssd.putNumberArray("Location", [int(t.spatialCoordinates.x), int(t.spatialCoordinates.y), int(t.spatialCoordinates.z)])
 
            processing_time = start_time - prev_time
            prev_time = start_time

            fps = 1 / processing_time
            cv2.putText(frame, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))

            # After all the drawing is finished, we show the frame on the screen
            # we can't do that on a headless RPi....           cv2.imshow("preview", frame)
            # Instead publish to CameraServer output stream for NetworkTables or MJPEG http stream\
            # ... and lower the refresh rate to comply with FRC robot wireless bandwidth regulations
            image_output_bandwidth_limit_counter += 1
            if image_output_bandwidth_limit_counter > 1:
                image_output_bandwidth_limit_counter = 0
                output_stream_front_nn.putFrame(frame)

            if streamDepth:
                output_stream_front_depth.putFrame(depthFrameColor)

            # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
            if cv2.waitKey(1) == ord('q'):
                break