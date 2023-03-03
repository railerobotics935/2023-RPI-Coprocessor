#!/usr/bin/env python3

import cv2
import depthai as dai
import time

import robotpy_apriltag
from wpimath.geometry import Transform3d
import math

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
manip = pipeline.create(dai.node.ImageManip)

xoutRgb = pipeline.create(dai.node.XLinkOut)
#xoutAprilTag = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
#xoutAprilTag.setStreamName("aprilTagData")

# Properties
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.setInterleaved(False)
camRgb.setIspScale(1,5) # 4056x3040 -> 812x608
camRgb.setPreviewSize(812, 608)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(25)

# camera resolution settings
width = 812
height = 608

manip.setMaxOutputFrameSize(307200) # 320x320x3
manip.initialConfig.setResizeThumbnail(320, 320)
camRgb.preview.link(manip.inputImage)

# Create the apriltag detector
detector = robotpy_apriltag.AprilTagDetector()
detector.addFamily("tag16h5")
detector.Config.quadDecimate = 1
estimator = robotpy_apriltag.AprilTagPoseEstimator(
    robotpy_apriltag.AprilTagPoseEstimator.Config(
        0.2, 500, 500, width / 2.0, height / 2.0
    )
)

# Linking
camRgb.isp.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the mono frames from the outputs defined above
    rgbQueue = device.getOutputQueue("rgb", 8, False)

    color = (0, 255, 0)

    startTime = time.monotonic()
    counter = 0
    fps = 0

    # Detect apriltag
    DETECTION_MARGIN_THRESHOLD = 100
    DETECTION_ITERATIONS = 50

    while(True):
        inFrame = rgbQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        output_img = inFrame.getCvFrame()
        gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # Coordinates of found targets, for NT output:
        x_list = []
        y_list = []
        id_list = []

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
            cv2.line(output_img, corner0, corner1, color = col_box, thickness = 2)
            cv2.line(output_img, corner1, corner2, color = col_box, thickness = 2)
            cv2.line(output_img, corner2, corner3, color = col_box, thickness = 2)
            cv2.line(output_img, corner3, corner0, color = col_box, thickness = 2)

            # Label the tag with the ID:
            cv2.putText(output_img, f"{tag_id}", (int(center.x), int(center.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, col_txt, 2)

            x_list.append((center.x - width / 2) / (width / 2))
            y_list.append((center.y - width / 2) / (width / 2))
            id_list.append(tag_id)

        cv2.putText(output_img, "Fps: {:.2f}".format(fps), (2, output_img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))

#        output_img = cv2.resize(output_img, (width // 3, height // 3))
#        output_stream.putFrame(output_img)

        cv2.imshow("rgb", output_img)

        if cv2.waitKey(1) == ord('q'):
            break
