import cv2
import numpy as np
import math
from parser import in_video

in_video_path = '../Naver_video_02.mp4'
#out_video_path = '../Result_Naver_video_01.mp4'
out_video_path = '../output_video02_.mp4'
cap = cv2.VideoCapture(in_video_path)
back_cap = cv2.VideoCapture(in_video_path)##
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

colors = [
    (0,0,255), #red
    (0,255,0), #green
    (255,0,0), #blue
    (0,255,255), #yellow
    (255,0,255), #pink
    (255,255,0) #skyblue
]

print("Wait...")

# hand point stack
left_hand = []
right_hand = []
for i in range(in_video.hum_cnt):
    left_hand.append([i+1])
    right_hand.append([i+1])

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    back_ret, back_frame = back_cap.read() # original frame / It's for opacity

    if ret == False:
        print("Oops... ")
        break

    #Skip the unrecognized frame
    if in_video.frames[i] == 'empty_frame':
        i += 1
        continue

    # Short Test
    if i > 2000 :
        break

    fr_humans = in_video.frames[i].humans
    
    # Draw a point for each person.
    for j in range(len(fr_humans)):
        human_color = colors[fr_humans[j].id - 1]
    
        # handneck anchor
        human_id = fr_humans[j].id - 1
        anchors = fr_humans[j].pose_pos

        # left handneck
        point = (anchors[9][0], anchors[9][1]) 
        left_hand[human_id].append(point)
        if (len(left_hand[human_id])>10): # 1th~10th points tracked
            for k in range(1,10): 
                if -80 < (left_hand[human_id][k+1][0] - left_hand[human_id][k][0]) < 80: # To elimate bad point
                    if  -80 < (left_hand[human_id][k+1][1] - left_hand[human_id][k][1]) < 80:
                        frame = cv2.line(frame, left_hand[human_id][k], left_hand[human_id][k+1], human_color, 2+k)
                left_hand[human_id][k] = left_hand[human_id][k+1]
            del left_hand[human_id][-1]

        # right handneck
        point = (anchors[10][0], anchors[10][1]) 
        right_hand[human_id].append(point)
        if (len(right_hand[human_id])>10):
            for k in range(1,10):
                if -80 < (right_hand[human_id][k+1][0] - right_hand[human_id][k][0]) < 80:
                    if -80 < (right_hand[human_id][k+1][1] - right_hand[human_id][k][1]) < 80: 
                        frame = cv2.line(frame, right_hand[human_id][k], right_hand[human_id][k+1], human_color, 2+k)
                right_hand[human_id][k] = right_hand[human_id][k+1]
            del right_hand[human_id][-1]

    ## opacity version ##
    # for k in range(1,11):
    #     for h in range(len(left_hand)):
    #         human_color = colors[h]
    #         if (len(left_hand[h])==12):
    #             if -40 < (left_hand[h][k+1][0] - left_hand[h][k][0]) < 40:
    #                     if  -40 < (left_hand[h][k+1][1] - left_hand[h][k][1]) < 40:
    #                         frame = cv2.line(frame, left_hand[h][k], left_hand[h][k+1], human_color, 1+k)
    #         if (len(right_hand[h])==12):
    #             if -40 < (right_hand[h][k+1][0] - right_hand[h][k][0]) < 40:
    #                     if -40 < (right_hand[h][k+1][1] - right_hand[h][k][1]) < 40: 
    #                         frame = cv2.line(frame, right_hand[h][k], right_hand[h][k+1], human_color, 1+k)
    #     frame = cv2.addWeighted(back_frame, 0.7-0.05*k, frame, 0.3+0.05*k, 0)

    # Give Opacity
    frame = cv2.addWeighted(back_frame,0.4,frame,0.6,0)
    # write output frame
    out.write(frame)

    i += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done!")