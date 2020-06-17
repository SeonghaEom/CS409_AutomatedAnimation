import cv2
import numpy as np
import math
from .segment import segment_mask
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
    ret, mask = cv2.threshold(effect_gray, 10 ,255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    fr_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    effect_fg = cv2.bitwise_and(effect, effect, mask=mask)

    dst = cv2.add(fr_bg, effect_fg)
    fr[x:rows+x, y:cols+y] = dst

    return fr

def back_tunnel_effect (cap, frame, back_cap, back_frame, out, in_video, effect_path, i) :
   
    eff_path = effect_path + '/back/turnnel.mp4'
    eff_video = cv2.VideoCapture(eff_path)

    print("tunnel...")

    ctx = mx.cpu(0)
    model = gluoncv.model_zoo.get_model('icnet_resnet50_mhpv1', pretrained=True)

    n = 100 # number of frames
    start = i
    
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
            img = segment_mask(model, ctx, frame, inverse_mask=True, color="black", alpha=1)
            frame = ani_effect(0,0, eff, img)
    
                
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