import numpy as np
import cv2
import chooseROI
import sys
#init params
paramters_s= dict(maxCorners= 100, qualityLevel = 0.4, minDistance = 1)
paramers_lk = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#choose ROI
frame_nr,roi = chooseROI.chooseRoi.get_roi("Input.mp4")
x,y,w,h =roi[0],roi[1],roi[2],roi[3]

# define green color range in HSV
red_lower = np.array([0, 160, 70])
red_upper = np.array([10, 160, 250])

#try with yellow
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])

#yellow-orange-red
yor_lower = np.array([0, 100, 200])
yor_upper = np.array([60, 255, 255])

#grey excluding orange yellow including
nogray_lower = np.array([0, 0, 200])
nogray_upper = np.array([60, 30, 255])

#brightness filter
bright_lower = np.array([0, 0, 220])
bright_upper = np.array([360, 50, 255])


#load video and frames
vidcap= cv2.VideoCapture("Input.mp4")
vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
print("framenumber")
print(frame_nr)

flag, frame= vidcap.read(frame_nr)

#create empty mask
mask =np.zeros_like(frame)


#create roi mask with color filter
# read first frame to set up ROI
roi_mask =np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
roi_mask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = 1

# extract ROI using color thresholding
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
color_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
color_roi_mask=cv2.bitwise_and(color_mask,roi_mask,None)
while True:
    cv2.imshow("ColorMask", color_mask)
    cv2.imshow("firstFrame", frame)
    if cv2.waitKey(1) == 13: #stop when enter is pressed
            break

print("color:")
print(color_roi_mask.sum())

#find possible features to start with, filtered by color and roi area
old = cv2.goodFeaturesToTrack(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask=color_roi_mask, **paramters_s)

if old is None or old.size <1:
    print("no suitable points could be found for tracking, try a different color range or a different roi")
    sys.exit()


frame_init=frame

while True:

    flag,frame=vidcap.read()
    if not flag:
        break

    # filter out motion vectors outside of the color thresholded area
    new_points, status, err = cv2.calcOpticalFlowPyrLK(frame_init,frame,  old, None, **paramers_lk)
    
    new = new_points[status == 1]
    old = old[status ==1]


    for i, (new, old) in enumerate(zip(new,old)):
        a,b = new.ravel()
        c,d = old.ravel()

        
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

        #display image with lines of movement
        img = cv2.add(frame, mask)
        cv2.imshow("Optical Flow", img)

        if cv2.waitKey(1) == 13: #stop when enter is pressed
            break
        frame_init = frame.copy()
        old=new.reshape(-1,1,2)

vidcap.release()
cv2.destroyAllWindows()
    




