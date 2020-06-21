import cv2
import numpy as np
import math

def handneck_effect (cap, frame, back_cap, back_frame, out, in_video, effect_path, i, hm = 0) :
    colors = [
    (0,0,255), #red
    (0,255,0), #green
    (255,0,0), #blue
    (0,255,255), #yellow
    (255,0,255), #pink
    (255,255,0), #skyblue
    (255,255,255) #white
    ]
    start = i
    term = 30
    print("handTracking...")

    # hand point stack
    left_hand = []
    right_hand = []
    for n in range(in_video.hum_cnt):
        left_hand.append([n+1])
        right_hand.append([n+1])

    while(cap.isOpened()):
        

        #Skip the unrecognized frame
        if in_video.frame.get(i) == 'empty_frame':
            i += 1
            continue

        # Short Test
        if i == start + term  :
            break

        fr_humans = in_video.frame.get(i).humans
        
        # Draw a point for each person.
        for j in range(len(fr_humans)):
            # if (hm > 0 and fr_humans[j].id != hm):
            #     continue
            # print (fr_humans[j], fr_humans[j].id)
            human_color = colors[fr_humans[j].id - 1]
        
            # handneck anchor
            human_id = fr_humans[j].id - 1
            anchors = fr_humans[j].pose_pos

            # left handneck
            point = (int(anchors[9][0]), int(anchors[9][1]))
            # print (point)
            left_hand[human_id].append(point)
            if (len(left_hand[human_id])>10): # 1th~10th points tracked
                for k in range(1,10): 
                    if -80 < (left_hand[human_id][k+1][0] - left_hand[human_id][k][0]) < 80: # To elimate bad point
                        if  -80 < (left_hand[human_id][k+1][1] - left_hand[human_id][k][1]) < 80:
                            # print (type(frame), type(human_id), human_color, type(2+k*7))
                            frame = cv2.line(frame, left_hand[human_id][k], left_hand[human_id][k+1], human_color, 2+k*7)
                    left_hand[human_id][k] = left_hand[human_id][k+1]
                del left_hand[human_id][-1]

            # right handneck
            point = (int(anchors[10][0]), int(anchors[10][1]))
            right_hand[human_id].append(point)
            if (len(right_hand[human_id])>10):
                for k in range(1,10):
                    if -80 < (right_hand[human_id][k+1][0] - right_hand[human_id][k][0]) < 80:
                        if -80 < (right_hand[human_id][k+1][1] - right_hand[human_id][k][1]) < 80: 
                            frame = cv2.line(frame, right_hand[human_id][k], right_hand[human_id][k+1], human_color, 2+k*7)
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
        frame = cv2.addWeighted(back_frame,0.3,frame,0.7,0)
        
        # write output frame
        out.write(frame)
        i += 1

        #
        ret, frame = cap.read()
        back_ret, back_frame = back_cap.read() # original frame / It's for opacity

        if ret == False:
            print("Oops... ")
            break

        

    return i, frame, back_frame