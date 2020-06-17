import cv2
import numpy as np
import math
import random

def back_light4_effect (cap, frame, back_cap, back_frame, out, in_video, effect_path, i) :
   
    eff_path = effect_path + '/back/light4.mp4'
    eff_video = cv2.VideoCapture(eff_path)

    print("stage light4...")

    n = 250 # number of frames
    start = i

    rand_num = random.randrange(0, 220, 1)
    
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
            if start+rand_num < i < start+rand_num+30:
                if i < int(start+n*2/5):
                    if i%12<5:
                        frame = eff
                elif i < int(start+n*4/5):
                    if i%8<4:
                        frame = eff
                else:
                    if i%2<1:
                        frame = eff
                

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