import cv2
import numpy as np
import chooseROI



#choose ROI
frame_nr,roi = chooseROI.chooseRoi.get_roi("Input.mp4")


#load video and frames
vidcap= cv2.VideoCapture("Input.mp4")
vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
flag, frame= vidcap.read()
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#define matrix
hsv = np.zeros_like(frame)
hsv[...,1]= 255 #3 dimensions, changing second dimension that represents speed of object (Saturation)

#define roi mask
x,y,w,h=roi[0], roi[1],roi[2],roi[3]
roi_mask = np.zeros_like(frame[:,:,0])
roi_mask[y:y+h, x:x+w] = 1

# Apply the ROI mask to the first frame
frame1_roi = cv2.bitwise_and(frame_gray_init, frame_gray_init, mask=roi_mask)


while True:
    flag,frame=vidcap.read()
    if not flag:
        break
    frame_gray=  cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Apply the ROI mask to the second frame
    frame2_roi = cv2.bitwise_and(frame_gray, frame_gray, mask=roi_mask)


     #0.5 scaling , 3 layers, 15 pixel in each layer,3 number of iterations at each pyramide level, 5 polynominal extension, poly_sigma by recommendation 1.1
    flow = cv2.calcOpticalFlowFarneback(frame1_roi, frame2_roi, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1]) # X, Y
    hsv[...,0] = angle * (180 / (np.pi / 2))
    hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    frame_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Dense optical flow', frame_rgb)
    if cv2.waitKey(1) == 13: # enter
        break

    frame_gray_init = frame_gray