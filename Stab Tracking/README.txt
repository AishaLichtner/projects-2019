There is different algorithms and different versions to be found in this folder.
The only package you need to install before is cv2.

The version that works best if dense.py, you can also have a look on optical_flow_roi_color

1. To execute dense.py: 
    - press "N" when the first window opens until the glowing stick appears (approximately 10 times)
    -when stick has appreared press "R" and choose the WHOLE area in which the stick will move
    - you can then see a black screen with the pixels in motion represented in different colors depending on direction and speed
2. To excute optical_flow_ROI/ optical_flow_ROI_colors/kalman_bolb:
    - press "N" when the first window opens until the glowing stick appears (approximately 10 times)
    -when stick has appreared press "R" and choose the area where the stick appears first 
    -results will be shown in seperate window
3. To execute stab.py (collection of different tracking algorithms of open cv) 
   -you can switch algorithms by chanigng index in line 9 ( but non of them really works)
   - and you just run it