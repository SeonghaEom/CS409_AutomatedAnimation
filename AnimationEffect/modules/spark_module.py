import cv2
import numpy as np
import math

def ani_effect(y,x,fr,effect):
    rows, cols, channels = effect.shape
    roi = fr[x:rows+x, y:cols+y]
    
    effect_gray = cv2.cvtColor(effect, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(effect_gray, 245 ,255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    fr_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    effect_fg = cv2.bitwise_and(effect, effect, mask=mask)

    dst = cv2.add(fr_bg, effect_fg)
    fr[x:rows+x, y:cols+y] = dst

    return fr

def spark_effect (cap, frame, back_cap, back_frame, out, in_video, effect_path, i) :
    
    print("spark...")

    n = 16 # number of frames
    start = i
    ani_start = []
    test = []
    while(cap.isOpened()):

        #Skip the unrecognized frame
        if in_video.frame.get(i) == 'empty_frame':
            i += 1
            continue

        # Short Test
        if i == start + n  :
            break

        fr_humans = in_video.frame.get(i).humans
        
        # Draw a point for each person.
        for j in range(1):
        
            human_id = fr_humans[j].id - 1
            anchors = fr_humans[j].pose_pos

            # set position and size
            if i == start:
                ani_start.append((anchors[1][0], anchors[1][1]))
                standard_height = anchors[13][1]-anchors[2][1] #  knee - eye
        
        #draw
        for j in range(len(ani_start)):
            if start <= i < start+n :
                eff = cv2.imread(effect_path+'/spark/spark'+str(i-start).zfill(4)+'.jpg')
                eff = cv2.resize(eff, dsize=(eff.shape[1]*standard_height//eff.shape[0], standard_height), interpolation=cv2.INTER_LINEAR)

                if (eff.shape[1]//2 <  ani_start[j][0] < frame.shape[1] - eff.shape[1]) and (ani_start[j][1] < frame.shape[0] - eff.shape[0]):
                    frame = ani_effect(ani_start[j][0]-eff.shape[1]//2,ani_start[j][1], frame, eff)
    
                
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