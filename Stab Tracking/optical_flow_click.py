import numpy as np
import cv2

#init params

paramers_lk = dict(winSize=(15, 15), maxLevel=4,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



#helper function
def select_point(event,x,y, flags,params):
    global point, selected_point,old_points
    if event == cv2.EVENT_LBUTTONDOWN: #rechtsklick
        point =(x,y)
        selected_point = True
        old_points = np.array([[x,y]], dtype=np.float32)


#load video and frames
vidcap= cv2.VideoCapture("Input.mp4")
flag, frame= vidcap.read()
print(flag)
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#cv2.imshow("first frame",frame_gray_init)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",select_point)
selected_point = False
point= ()
old_points = np.array([[]])
mask =np.zeros_like(frame)


while True:
    
    flag,frame=vidcap.read()
    if not flag:
        break
   
    frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if selected_point:

        cv2.circle(frame, point, 5, (0,0,255), 2)

        new_points,status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init,frame_gray, old_points, None, **paramers_lk)
        frame_gray_init=frame_gray.copy()

        old_points=new_points

        a,b = new_points.ravel()
        c,d = old_points.ravel()

        mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), (0,255,255), 2)
        frame = cv2.circle(frame, (int(a),int(b)), 5, (0,255,255), -1)

    img = cv2.add(frame, mask)
    cv2.imshow("Frame", img)
    #cv2.imshow("Frame 2", mask)

    key=cv2.waitKey(1)
    if cv2.waitKey(1) == 27: #stop when esc
            break

vidcap.release()
cv2.destroyAllWindows()
    




