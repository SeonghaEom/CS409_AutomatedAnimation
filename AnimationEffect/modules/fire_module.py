import cv2
import numpy as np
import math
def ani_effect(y,x,fr,effect):
    rows, cols, channels = effect.shape
    roi = fr[x:rows+x, y:cols+y]
    
    effect_gray = cv2.cvtColor(effect, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(effect_gray, 10 ,255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    fr_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    effect_fg = cv2.bitwise_and(effect, effect, mask=mask)

    dst = cv2.add(fr_bg, effect_fg)
    fr[x:rows+x, y:cols+y] = dst

    return fr

def fire_effect (cap, frame, back_cap, back_frame, out, in_video, effect_path, i) :

    start = i
    term = 80
    print("fire...")

    # point stack
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
        
            # handneck anchor
            human_id = fr_humans[j].id - 1
            anchors = fr_humans[j].pose_pos

            standard_height = int((anchors[13][1]-anchors[2][1])*0.4) 
            eff = cv2.imread(effect_path+'/fire/animation_fire-'+str((i-start)%64).zfill(4)+'.jpg')
            eff = cv2.resize(eff, dsize=(eff.shape[1]*standard_height//eff.shape[0], standard_height), interpolation=cv2.INTER_LINEAR)
                
            # left handneck
            point = (anchors[9][0], anchors[9][1]) 
            left_hand[human_id].append(point)
            if (len(left_hand[human_id])>3): # 1th~10th points tracked
                for k in range(1,3): 
                    if -80 < (left_hand[human_id][k+1][0] - left_hand[human_id][k][0]) < 80: # To elimate bad point
                        if  -80 < (left_hand[human_id][k+1][1] - left_hand[human_id][k][1]) < 80:
                            # frame = cv2.line(frame, left_hand[human_id][k], left_hand[human_id][k+1], human_color, 2+k*7)
                            if (eff.shape[1]//2 <  left_hand[human_id][k][0] < frame.shape[1] - eff.shape[1]) and (left_hand[human_id][k][1] < frame.shape[0] - eff.shape[0]):
                                frame = ani_effect(left_hand[human_id][k][0]-eff.shape[1]//2,left_hand[human_id][k][1]-eff.shape[0]//2, frame, eff)
                    left_hand[human_id][k] = left_hand[human_id][k+1]
                del left_hand[human_id][-1]

            # right handneck
            point = (anchors[10][0], anchors[10][1]) 
            right_hand[human_id].append(point)
            if (len(right_hand[human_id])>3):
                for k in range(1,3):
                    if -80 < (right_hand[human_id][k+1][0] - right_hand[human_id][k][0]) < 80:
                        if -80 < (right_hand[human_id][k+1][1] - right_hand[human_id][k][1]) < 80: 
                            # frame = cv2.line(frame, right_hand[human_id][k], right_hand[human_id][k+1], human_color, 2+k*7)
                            if (eff.shape[1]//2 <  right_hand[human_id][k][0] < frame.shape[1] - eff.shape[1]) and (right_hand[human_id][k][1] < frame.shape[0] - eff.shape[0]):
                                frame = ani_effect(right_hand[human_id][k][0]-eff.shape[1]//2,right_hand[human_id][k][1]-eff.shape[0]//2, frame, eff)
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
        frame = cv2.addWeighted(back_frame,0.2,frame,0.8,0)
        
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