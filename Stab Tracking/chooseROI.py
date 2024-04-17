import cv2
import numpy as np

class chooseRoi:

    @staticmethod
    def get_roi(path='Input.mp4'):
    # Read the video sequence
        cap = cv2.VideoCapture('Input.mp4')

        # Create an empty list to store the frames
        frames = []
        frame_nr =0

        # Loop through the frames and add them to the list
        while True:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add the frame to the list
            frames.append(frame)
            
            # Show the frame
            cv2.imshow('frame', frame)
            frame_nr=frame_nr+1
            if cv2.waitKey(0) == ord('r'):
                r=cv2.selectROI(frame)
                break
            key = cv2.waitKey(0)
            while key != ord('n'):
                key = cv2.waitKey(0)
            
            #exit when press esc
            if key == 27:
                break

        # Release the video capture
        cap.release()
        # Show the final canvas with the chosen points
        cv2.waitKey(0)

        # Close all windows
        cv2.destroyAllWindows()
        print("here comes the frame")
        print(frame_nr)
        return frame_nr,r