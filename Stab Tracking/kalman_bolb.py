import cv2
import numpy as np
import chooseROI

# set up blob detector parameters
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 30 #cut out dark objects
params.maxThreshold= 255
params.filterByArea = True
params.minArea = 1
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

# create blob detector object
detector = cv2.SimpleBlobDetector_create(params)

# initialize Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

# read video file
cap = cv2.VideoCapture('Input.mp4')

# define green color range in HSV
red_lower = np.array([0, 160, 70])
red_upper = np.array([10, 160, 250])
#try with yellow
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
#choose roi
frame_nr,roi = chooseROI.chooseRoi.get_roi("Input.mp4")

#load video and frames
vidcap= cv2.VideoCapture("Input.mp4")
vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
x, y, w, h = roi[0],roi[1],roi[2],roi[3]

# read first frame to set up ROI
flag, frame= vidcap.read()

# initialize Kalman filter state
kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)

while True:
    # read frame
    ret, frame = cap.read()
    if not ret:
        break

    # extract ROI using color thresholding
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    roi_frame = mask[int(y):int(y+h), int(x):int(x+w)]

    # perform blob detection
    keypoints = detector.detect(roi_frame)
    print("hier sind die keypoint")
    print(keypoints)

    if keypoints:
        # get center of first detected blob
        x, y = keypoints[0].pt
        x += roi[0]
        y += roi[1]

        # update Kalman filter
        kalman.correct(np.array([[x], [y]], np.float32))

    # predict next state using Kalman filter
    prediction = kalman.predict()
    print(prediction.ravel())
    print(prediction.shape)

    # draw predicted position on frame
    x, y = prediction.ravel()[0],prediction.ravel()[1]
    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()