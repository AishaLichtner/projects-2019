import numpy as np
import cv2
import chooseROI
#init params
paramters_s= dict(maxCorners= 100, qualityLevel = 0.2, minDistance = 7)
paramers_lk = dict(winSize=(15, 15), maxLevel=4,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
colors = np.random.randint(0,255, (100, 3)) #for 100 edges 3 color channels
#choose ROI
frame_nr,roi = chooseROI.chooseRoi.get_roi("Input.mp4")

#load video and frames
vidcap= cv2.VideoCapture("Input.mp4")
vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
flag, frame= vidcap.read()
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#cv2.imshow("first frame",frame_gray_init)

mask =np.zeros_like(frame)
#create roi mask
roi_mask = np.zeros_like(frame, dtype=np.uint8)
roi_mask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = 1
roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY) 

#find possible features to track within the roi mask
edges= cv2.goodFeaturesToTrack(frame_gray_init, mask=roi_mask, **paramters_s)

while True:
    flag,frame=vidcap.read()
    if not flag:
        break
    frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_edges,status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init,frame_gray, edges, None, **paramers_lk)
    

    new= new_edges[status==1] #status one "can keep tracking that edge"
    old=edges[status==1]
    for i, (new, old) in enumerate(zip(new,old)):
        a,b = new.ravel()
        c,d = old.ravel()


        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)


        img = cv2.add(frame, mask)
        cv2.imshow("Optical Flow", img)

        if cv2.waitKey(1) == 13: #stop when enter is pressed
            break
        frame_gray_init=frame_gray.copy()
        edges=new.reshape(-1,1,2)


vidcap.release()
cv2.destroyAllWindows()
    




