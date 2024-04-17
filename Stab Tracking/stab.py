import cv2
import numpy as np
import sys 
from random import randint



tracker_types= ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "MOSSE", "CSRT"]
tracker_type = tracker_types[6]

if tracker_type == "BOOSTING":
    tracker = cv2.legacy.TrackerBoosting_create()
elif tracker_type == "MIL":
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == "KCF":
    tracker = cv2.legacy.TrackerKCF_create()
elif tracker_type == "TLD":
    tracker = cv2.legacy.TrackerTLD_create()
elif tracker_type == "MEDIANFLOW":
    tracker = cv2.legacy.TrackerMedianFlow_create()
elif tracker_type == "MOSSE":
    tracker = cv2.legacy.TrackerMOSSE_create()
elif tracker_type == "CSRT":
    tracker = cv2.legacy.TrackerCSRT_create()

#---------------------------------------------------------------------------------------------------------------------------------------
# -----init    

vidcap = cv2.VideoCapture("Input.mp4")
if not vidcap.isOpened():
    print("Error while loading video")
    sys.exit()

flag,frame = vidcap.read()
if not flag:
    print("Error while loading frame")
    sys.exit()
colors= (randint(0,225), randint(0,225), randint(0,225)) #BGR!


while True:

    # Read first frame.
    ok, frame = vidcap.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()
    # Retrieve an image and Display it.
    if((0xFF & cv2.waitKey(10))==ord('p')): # Press key `p` to pause the video to start tracking
        break
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", frame)
cv2.destroyWindow("Image");

# select the bounding box
bbox = (287, 23, 86, 320)

# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)




while True: #loop over frames 
    flag, frame = vidcap.read()
    if not flag:
        break
    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    flag, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if flag:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1])) #corner coordinates
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])) # bbox[2] width of box bbox[3] height
        cv2.rectangle(frame, p1, p2, colors, 2, 1) #add rectangel to frame at position (p1,p2) with random color, 2 size of boundig boxes and 1 type of line
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

     # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break # 27 equals escape key in keyboard


