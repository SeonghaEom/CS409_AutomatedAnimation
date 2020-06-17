import cv2
import numpy as np
import math
import time
import mxnet as mx
import gluoncv
import random
import moviepy.editor as mpe

from .modules.handneck_module import handneck_effect
from .modules.heart1_module import heart1_effect
from .modules.heart2_module import heart2_effect, heart2_effects
from .modules.spark_module import spark_effect
from .modules.fire_module import fire_effect
from .modules.wing_module import wing_effect
from .modules.ribbon_module import ribbon_effect
from .modules.mirrorball import mirrorball_effect
from .modules.back_streak import back_streak_effect
from .modules.back_glowing import back_glowing_effect
from .modules.back_light import back_light_effect
from .modules.back_tunnel import back_tunnel_effect
from .modules.back_light2 import back_light2_effect
from .modules.back_light3 import back_light3_effect
from .modules.back_light4 import back_light4_effect
from .modules.back_stagelight import back_stagelight_effect
from .modules.color_outline import outline_effect
from .modules.color_outline_black import black_outline_effect
from .modules.foot3_module import foot3_effect
from .modules.foot2_module import foot2_effect
from .modules.foot_module import foot_effect

# BGR style
colors = [
    (0,0,255), #red
    (0,255,0), #green
    (255,0,0), #blue
    (0,255,255), #yellow
    (255,0,255), #pink
    (255,255,0), #skyblue
    (255, 255, 255) #white
]

def get_effect_range(video, interval):
    effect_range = {}
    effect_start_list = []
    for start, end in video.danceFormationKeys:
        step = int((end - start) / interval)
        for t in range(step):
            effect_start = random.randrange(start+interval*t, start+interval*(t+1), 1)
            effect_start_list.append(effect_start)
            effect_end = effect_start + 30
            effect_range[effect_start] = effect_end

    return effect_range, effect_start_list

def get_foot_list(video):
    left_start_list = []
    right_start_list = []
    foot_start_list = []
    for start, end in video.framesForLeftFoot.keys():
        left_start_list.append(start)
        foot_start_list.append(start)
    for start, end in video.framesForRightFoot.keys():
        right_start_list.append(start)
        foot_start_list.append(start)
    
    return left_start_list, right_start_list, foot_start_list


def animation_effect(video, args):
    ctx = mx.cpu(0)
    model = gluoncv.model_zoo.get_model('icnet_resnet50_mhpv1', pretrained=True)
    
    cap = cv2.VideoCapture(args.input_video_path)
    back_cap = cv2.VideoCapture(args.input_video_path)
    effect_path = 'AnimationEffect/Effects/'

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("This video is {} width {} height {} fps" .format(width, height, fps))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_nosound_video_path, fourcc, fps, (width, height))

    effect_start_list = []
    foot_start_list = []

    if args.random_effect and not args.step_detection:
        effect_range, effect_start_list = get_effect_range(video, 100)

    if args.step_detection:
        left_start_list, right_start_list, foot_start_list = get_foot_list(video)
    
        if args.random_effect:
            foot_start_list.sort()
            
            for t in range(len(foot_start_list)):
                if t == len(foot_start_list)-2: break
                step = int((foot_start_list[t+1] - foot_start_list[t])/100)
                for s in range(step):
                    effect_start = random.randrange(foot_start_list[t]+120*s, foot_start_list[t]+100*(s+1), 1)
                    effect_start_list.append(effect_start)


    start = time.time()
    print("Start animation effect generation")

    effect_function_list = [handneck_effect, heart1_effect, heart2_effects, spark_effect, fire_effect, wing_effect, ribbon_effect, 
                        back_glowing_effect, back_light_effect, back_light2_effect, back_light2_effect, back_light3_effect,
                        back_light3_effect, back_light4_effect, back_tunnel_effect, back_stagelight_effect, outline_effect,
                        black_outline_effect,]


    i=0
    left_foot = []
    right_foot = []
    for n in range(video.hum_cnt):
        left_foot.append([n+1])
        right_foot.append([n+1])

    while cap.isOpened():
        ret, frame = cap.read()
        back_ret, back_frame = back_cap.read()

        if i % 50 == 0: print(i)
        frame_info = video.frame.get(i)

        if args.step_detection and foot_start_list[0]>100 and i == foot_start_list[0]-60:
            effect_function = back_streak_effect
            i, frame, back_frame = effect_function(cap, frame, back_cap, back_frame, out, video, effect_path, i)

        if i in foot_start_list:
            effect_function = foot_effect
            i, frame, back_frame = effect_function(cap, frame, back_cap, back_frame, out, video, effect_path, i)

        if i in effect_start_list:
            if i == effect_start_list[0]:
                effect_function = back_streak_effect
            else:
                effect_function = random.choice(effect_function_list)

            i, frame, back_frame = effect_function(cap, frame, back_cap, back_frame, out, video, effect_path, i)


        out.write(frame)
        i += 1

        if i > video.frame.getLength(): break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    my_clip = mpe.VideoFileClip(args.output_nosound_video_path)
    audio_background = mpe.VideoFileClip(args.input_video_path)
    my_clip.audio = audio_background.audio
    my_clip.write_videofile(args.output_video_path,
        codec='libx264', 
        audio_codec='aac', 
        temp_audiofile='temp-audio.m4a', 
        remove_temp=True)

    my_clip.close()
    audio_background.close()

    end = time.time()
    print("Complete animation effect generation")
    print("It took {} sec" .format(end - start))

