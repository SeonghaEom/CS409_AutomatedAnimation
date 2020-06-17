import cv2
import numpy as np
import math

def ani_effect(y,x,fr,effect):
    rows, cols, channels = effect.shape
    roi = fr[x:rows+x, y:cols+y]
    
    effect_gray = cv2.cvtColor(effect, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(effect_gray, 90 ,255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    fr_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    effect_fg = cv2.bitwise_and(effect, effect, mask=mask)

    dst = cv2.add(fr_bg, effect_fg)
    fr[x:rows+x, y:cols+y] = dst

    return fr

def back_streak_effect (cap, frame, back_cap, back_frame, out, in_video, effect_path, i) :
    
    eff_path = effect_path + '/back/Streaks.mp4'
    eff_video = cv2.VideoCapture(eff_path)

    print("streak...")

    n = 100 # number of frames
    start = i
    ani_start = []

    while(cap.isOpened()):

        #Skip the unrecognized frame
        if in_video.frame.get(i) == 'empty_frame':
            i += 1
            continue

        # Short Test
        if i == start + n  :
            break


        if start <= i < start+n :
            r_eff, eff = eff_video.read()
            eff = cv2.resize(eff, dsize=(frame.shape[1],frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            frame = ani_effect(0,0, frame, eff)
    
                
        # Give Opacity
        frame = cv2.addWeighted(back_frame,0.4,frame,0.6,0)

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