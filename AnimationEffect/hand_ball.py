import cv2
import numpy as np
import math
from parser import in_video

in_video_path = '../../Naver_video_01.mp4'
#out_video_path = '../Result_Naver_video_01.mp4'
out_video_path = '../../output_hand_ball.mp4'
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
left_hand = []
right_hand = []
for i in range(in_video.hum_cnt):
    left_hand.append([i+1])
    right_hand.append([i+1])

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    ret, back_frame = back_cap.read()

    if ret == False:
        print("Oops... ")
        break

    #Skip the unrecognized frame
    if in_video.frames[i] == 'empty_frame':
        i += 1
        continue

    # Short Test
    if i > 500 :
        break

    fr_humans = in_video.frames[i].humans
    ##
    for j in range(len(fr_humans)):
        human_color = colors[fr_humans[j].id - 1]

        #draw hand points
        anchors = fr_humans[j].pose_pos
        point = (anchors[9][0], anchors[9][1])
        frame = cv2.line(frame, point, point, human_color, 40)
        point = (anchors[10][0], anchors[10][1])
        frame = cv2.line(frame, point, point, human_color, 40)


    frame = cv2.addWeighted(back_frame,0.7,frame,0.3,0)
    # write output frame
    out.write(frame)

    i += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done!")