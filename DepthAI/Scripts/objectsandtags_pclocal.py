#!/usr/bin/env python3
#
# based on: https://docs.luxonis.com/projects/api/en/v2.1.0.0/samples/26_1_spatial_mobilenet/
# and: https://docs.luxonis.com/projects/api/en/latest/samples/ObjectTracker/spatial_object_tracker/#spatial-object-tracker-on-rgb
# updated to work on FRC 2022 WPILibPi image and to be uploaded as a vision application
# communicating with shuffleboard and RoboRIO through NetworkTables and CameraServer
# Jaap van Bergeijk, 2022

#from operator import truediv
from pathlib import Path

#for DepthAI processing
import json
import time
import sys
import cv2
import depthai as dai
import numpy as np

#for AprilTag processing
import robotpy_apriltag
from wpimath.geometry import Transform3d
import math

if __name__ == "__main__":

    # Label texts
    labelMap = ["cone", "cube", "robot"]
    syncNN = True
    wideFOV = True
    stereoDepth = True
    streamDepth = True

    # Static setting for model BLOB, this runs on a RPi with a RO filesystem
    nnPath = str((Path(__file__).parent / Path('../models/yolov6ntrained_openvino_2021.4_6shave.blob')).resolve().absolute())

    # Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
    pipeline = dai.Pipeline()

    # First, we want the Color camera as the output
    camRgb = pipeline.create(dai.node.ColorCamera)
    if wideFOV:
        manip = pipeline.create(dai.node.ImageManip)
    if stereoDepth:
        detectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
    else:
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    objectTracker = pipeline.create(dai.node.ObjectTracker)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
#    nnOut = pipeline.create(dai.node.XLinkOut)
    if stereoDepth:
        if streamDepth:
            xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutTracker = pipeline.create(dai.node.XLinkOut)
    xoutRgbPreview = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
#    nnOut.setStreamName("nn")
    if stereoDepth:
        if streamDepth:
            xoutDepth.setStreamName("depth")
    xoutTracker.setStreamName("tracklets")
    xoutRgbPreview.setStreamName("rgbpreview")
       
    # Properties
    if wideFOV:
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        camRgb.setInterleaved(False)
        camRgb.setIspScale(1,5) # 4056x3040 -> 812x608
        camRgb.setPreviewSize(812, 608)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(25)
    else:
        camRgb.setPreviewSize(320, 320)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(40)

    # Network specific settings
    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setNumClasses(3)
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setAnchors([])
    detectionNetwork.setAnchorMasks({})
    detectionNetwork.setIouThreshold(0.5)
    detectionNetwork.setBlobPath(nnPath)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # Use ImageManip to resize to 320x320 with letterboxing: enables a wider FOV
    if wideFOV:
        manip.setMaxOutputFrameSize(307200) # 320x320x3
        manip.initialConfig.setResizeThumbnail(320, 320)
        camRgb.preview.link(manip.inputImage)

    if stereoDepth:
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # setting node configs
    if stereoDepth:
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

#    spatialDetectionNetwork.setBlobPath(nnBlobPath)
#    spatialDetectionNetwork.setConfidenceThreshold(0.5)
#    spatialDetectionNetwork.input.setBlocking(False)
#    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
#    detectionNetwork.setDepthLowerThreshold(100)
#    detectionNetwork.setDepthUpperThreshold(15000)

    objectTracker.setDetectionLabelsToTrack([0, 1, 2])  # track cones, cubes, robots
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

    # Linking
    if stereoDepth:
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

    if wideFOV:
        manip.out.link(detectionNetwork.input)
    else:
        camRgb.preview.link(detectionNetwork.input)

    # BEGIN insert for object tracking test
    objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
    objectTracker.out.link(xoutTracker.input)

    detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

    detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    detectionNetwork.out.link(objectTracker.inputDetections)
    stereo.depth.link(detectionNetwork.inputDepth)
    if streamDepth:
        detectionNetwork.passthroughDepth.link(xoutDepth.input)

    camRgb.preview.link(xoutRgbPreview.input)
    # END insert for object tracking test

    # BEGIN disabled for object tracking test
#    if syncNN:
#        detectionNetwork.passthrough.link(xoutRgb.input)
#    else:
#        if wideFOV:
#            manip.out.link(xoutRgb.input)
#        else:
#            camRgb.preview.link(xoutRgb.input)
        
#    if stereoDepth:
#        stereo.depth.link(detectionNetwork.inputDepth)
#        detectionNetwork.passthroughDepth.link(xoutDepth.input)

#    detectionNetwork.out.link(nnOut.input)
    # END disabled for object tracking test

    # Create the apriltag detector
    detector = robotpy_apriltag.AprilTagDetector()
    detector.addFamily("tag16h5")
    detector.Config.quadDecimate = 1
    estimator = robotpy_apriltag.AprilTagPoseEstimator(
        robotpy_apriltag.AprilTagPoseEstimator.Config(
            0.2, 500, 500, 812 / 2.0, 608 / 2.0
        )
    )
    # Detect apriltag
    DETECTION_MARGIN_THRESHOLD = 100
    DETECTION_ITERATIONS = 50

    # Pipeline is now finished, and we need to find an available device to run our pipeline
    # we are using context manager here that will dispose the device after we stop using it
    #with dai.Device(pipeline, True) as device:
    with dai.Device(pipeline) as device:
        # From this point, the Device will be in "running" mode and will start sending data via XLink

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
#        detectionNNQueue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        tracklets = device.getOutputQueue(name="tracklets", maxSize=4, blocking=False)
        if streamDepth:
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        rgbPreviewQueue = device.getOutputQueue(name="rgbpreview", maxSize=4, blocking=False)

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)

        while True:
            inPreview = previewQueue.get()
#            inDet = detectionNNQueue.get()
            track = tracklets.get()
            if streamDepth:
                depth = depthQueue.get()
            inFrame = rgbPreviewQueue.get()

            frame = inPreview.getCvFrame()
            if streamDepth:
                depthFrame = depth.getFrame() # depthFrame values are in millimeters, 640x400 pixels
                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
#                print(f"depth frame: {depthFrame.shape[1]} : {depthFrame.shape[0]}")

            # Feed gray scale image into AprilTag library
            gray = cv2.cvtColor(inFrame.getCvFrame(), cv2.COLOR_BGR2GRAY)
            # Coordinates of found AprilTags, for NT output:
            apriltag_x_list = []
            apriltag_y_list = []
            apriltag_id_list = []

            tag_info = detector.detect(gray)
            apriltags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]

            # OPTIONAL: Ignore any tags not in the set used on the 2023 FRC field:
            #filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 9))]

            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            trackletsData = track.tracklets
#            detections = inDet.detections

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

            for tag in apriltags:

                est = estimator.estimateOrthogonalIteration(tag, DETECTION_ITERATIONS)
                pose = est.pose1

                tag_id = tag.getId()
                center = tag.getCenter()
                #hamming = tag.getHamming()
                #decision_margin = tag.getDecisionMargin()
                
                # Look up depth at center of Tag from depth image
                # depthFrame = 640 x 400
                # rgbPreview = 812 x 608
#                tag_distance = depthFrame[int(center.x * 640 / 812),int(center.y * 400 / 608)]
                
#                print(f"{tag_id}: {center} , {tag_distance}")
#                print(f"{tag_id}: {center} ,{pose}")

                # Highlight the edges of all recognized tags and label them with their IDs:
                if ((tag_id > 0) & (tag_id < 9)):
                    col_box = (0,255,0)
                    col_txt = (0,255,0)
                else:
                    col_box = (0,0,255)
                    col_txt = (0,255,255)

                # Draw a frame around the tag, output frame is 320x320, AprilTag uses the higher 812x608 resolution RGB preview
                frame_size_adjust = 320.0 / 812.0
                frame_vertical_shift = frame_size_adjust * (812 - 608) * 0.5
                corner0 = (int(tag.getCorner(0).x * frame_size_adjust), int(tag.getCorner(0).y * frame_size_adjust + frame_vertical_shift))
                corner1 = (int(tag.getCorner(1).x * frame_size_adjust), int(tag.getCorner(1).y * frame_size_adjust + frame_vertical_shift))
                corner2 = (int(tag.getCorner(2).x * frame_size_adjust), int(tag.getCorner(2).y * frame_size_adjust + frame_vertical_shift))
                corner3 = (int(tag.getCorner(3).x * frame_size_adjust), int(tag.getCorner(3).y * frame_size_adjust + frame_vertical_shift))
                cv2.line(frame, corner0, corner1, color = col_box, thickness = 1)
                cv2.line(frame, corner1, corner2, color = col_box, thickness = 1)
                cv2.line(frame, corner2, corner3, color = col_box, thickness = 1)
                cv2.line(frame, corner3, corner0, color = col_box, thickness = 1)

                # Label the tag with the ID:
                cv2.putText(frame, f"{tag_id}", (int(center.x  * frame_size_adjust), int(center.y * frame_size_adjust + frame_vertical_shift)), cv2.FONT_HERSHEY_SIMPLEX, 1, col_txt, 2)

#                apriltag_x_list.append((center.x - 812 / 2) / (812 / 2))
#                apriltag_y_list.append((center.y - 608 / 2) / (608 / 2))
#                apriltag_id_list.append(tag_id)

            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)

            cv2.imshow("tracker", frame)
            if streamDepth:
                cv2.imshow("depth", depthFrameColor)

            if cv2.waitKey(1) == ord('q'):
                break