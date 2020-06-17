import cv2
import numpy as np
import math

def ani_effect(y,x,fr,effect):
    rows, cols, channels = effect.shape
    roi = fr[x:rows+x, y:cols+y]
    
    effect_gray = cv2.cvtColor(effect, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(effect_gray, 80 ,255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    fr_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    effect_fg = cv2.bitwise_and(effect, effect, mask=mask)

    dst = cv2.add(fr_bg, effect_fg)
    fr[x:rows+x, y:cols+y] = dst

    return fr

def wing_effect (cap, frame, back_cap, back_frame, out, in_video, effect_path, i) :
    
    print("Wing...")

    n = 82 # number of frames
    start = i

    while(cap.isOpened()):

        #Skip the unrecognized frame
        if in_video.frame.get(i) == 'empty_frame':
            i += 1
            continue

        # Short Test
        if i == start + n  :
            break

        center_id = in_video.frame.get(i).center
        # Draw a point for each person.
        for human in in_video.frame.get(i).humans: 
            if human.id != center_id: continue
            # anchor
            anchors = human.pose_pos
            
            left_shoulder = (anchors[5][0], anchors[5][1])
            right_shoulder = (anchors[6][0], anchors[6][1])

            # draw prepared img
            if start <= i < start+n :
                eff = cv2.imread(effect_path+'/wing_left/animation_wing-'+str(i-start).zfill(4)+'.jpg')
                eff2 = cv2.imread(effect_path+'/wing_right/animation_wing-'+str(i-start).zfill(4)+'.jpg')

                # resize
                sizing = 0.8
                eff = cv2.resize(eff, dsize=(0, 0), fx=sizing, fy=sizing, interpolation=cv2.INTER_AREA)
                eff2 = cv2.resize(eff2, dsize=(0, 0), fx=sizing, fy=sizing, interpolation=cv2.INTER_AREA)

                if (left_shoulder[0] < frame.shape[1] - eff.shape[1]) and (left_shoulder[1] < frame.shape[0] - eff.shape[0]):
                    frame = ani_effect(left_shoulder[0],left_shoulder[1]-eff.shape[0]//2, frame, eff)
                
                if (eff2.shape[1] < right_shoulder[0] < frame.shape[1]) and (right_shoulder[1] < frame.shape[0] - eff2.shape[0]):
                    frame = ani_effect(right_shoulder[0]-eff2.shape[1],right_shoulder[1]-eff2.shape[0]//2, frame, eff2)
    
                
        # Give Opacity
        frame = cv2.addWeighted(back_frame,0.07,frame,0.93,0)

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