import cv2
import numpy as np
import math
from .segment import draw_segment_outline
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
from matplotlib import pyplot as plt
from gluoncv.utils.viz import get_color_pallete, cv_plot_image
import matplotlib.image as mpimg
from scipy import ndimage

def ani_effect(y,x,fr,effect):
    rows, cols, channels = effect.shape
    roi = fr[x:rows+x, y:cols+y]
    
    effect_gray = cv2.cvtColor(effect, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(effect_gray, 250 ,255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    fr_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    effect_fg = cv2.bitwise_and(effect, effect, mask=mask)

    dst = cv2.add(fr_bg, effect_fg)
    fr[x:rows+x, y:cols+y] = dst

    return fr

def black_outline_effect (cap, frame, back_cap, back_frame, out, in_video, effect_path, i) :

    colors = [
    (0,0,255), #red
    (0,255,0), #green
    (255,0,0), #blue
    (0,255,255), #yellow
    (255,0,255), #pink
    (255,255,0) #skyblue
    ]

    print("outline...")

    ctx = mx.cpu(0)
    model = gluoncv.model_zoo.get_model('icnet_resnet50_mhpv1', pretrained=True)

    n = 30 # number of frames
    start = i

    boxes = []
    for l in range(in_video.hum_cnt):
        boxes.append([l+1])
        

    while(cap.isOpened()):

        #Skip the unrecognized frame
        if in_video.frame.get(i) == 'empty_frame':
            i += 1
            continue

        # Short Test
        if i == start + n  :
            break

        if start <= i < start+n :
            fr_humans = in_video.frame.get(i).humans
            img = draw_segment_outline(model, ctx, frame, thickness=6, black=True, color="white", alpha=1)

            for j in range(len(fr_humans)):
                human_color = colors[fr_humans[j].id - 1]
                
                human_id = fr_humans[j].id - 1
                anchors = fr_humans[j].box_pos
                
                point = ((anchors[0],anchors[2]), (anchors[1],anchors[3]))
                boxes[human_id].append(point)
                if (len(boxes[human_id])>2): # 1th~10th points tracked
                    for k in range(1,2): 
                        frame = cv2.rectangle(frame, boxes[human_id][k][0], boxes[human_id][k][1], human_color, -1)
                        boxes[human_id][k] = boxes[human_id][k+1]
                    del boxes[human_id][-1]
            
            frame = ani_effect(0,0, frame, img)
    
                
        # Give Opacity
        # frame = cv2.addWeighted(back_frame,0.8,frame,0.2,0)

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