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

def get_effect_range(video):
    effect_range = {}
    effect_start_list = []
    for start, end in video.danceFormationKeys:
        step = int((end - start) / 150)
        for t in range(step):
            effect_start = random.randrange(start+150*t, start+150*(t+1), 1)
            effect_start_list.append(effect_start)
            effect_end = effect_start + 30
            effect_range[effect_start] = effect_end

    return effect_range, effect_start_list

def animation_effect(video, in_video_path, out_video_path, out_nosound_path):
    ctx = mx.cpu(0)
    model = gluoncv.model_zoo.get_model('icnet_resnet50_mhpv1', pretrained=True)
    
    cap = cv2.VideoCapture(in_video_path)
    back_cap = cv2.VideoCapture(in_video_path)
    effect_path = 'AnimationEffect/Effects/'

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("This video is {} width {} height {} fps" .format(width, height, fps))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_nosound_path, fourcc, fps, (width, height))

    effect_range, effect_start_list = get_effect_range(video)

    print(effect_range)

    start = time.time()
    print("Start animation effect generation")

    effect_function_list = [handneck_effect, heart1_effect, heart2_effects, spark_effect, fire_effect, wing_effect, ribbon_effect, 
                        back_glowing_effect, back_light_effect, back_light2_effect, back_light2_effect, back_light3_effect,
                        # back_light3_effect, back_light4_effect, back_tunnel_effect, back_stagelight_effect, outline_effect,
                        # black_outline_effect,]
    ]

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

        if i in effect_start_list:
            if i == effect_start_list[0]:
                effect_function = back_streak_effect
            else:
                effect_function = random.choice(effect_function_list)

            # effect_function = foot3_effect
            i, frame, back_frame = effect_function(cap, frame, back_cap, back_frame, out, video, effect_path, i)

        # for j in range(len(frame_info.humans)):
        #     if video.leftFeetMov[frame_info.humans[j].id][i] > 5:
        #         human_color = colors[frame_info.humans[j].id - 1]
        #         human_id = frame_info.humans[j].id - 1
        #         anchors = frame_info.humans[j].pose_pos

        #         point = (anchors[15][0], anchors[16][1]) 
        #         left_foot[human_id].append(point)
        #         if (len(left_foot[human_id])>10): # 1th~10th points tracked
        #             for k in range(1,10): 
        #                 if -80 < (left_foot[human_id][k+1][0] - left_foot[human_id][k][0]) < 80: # To elimate bad point
        #                     if  -80 < (left_foot[human_id][k+1][1] - left_foot[human_id][k][1]) < 80:
        #                         frame = cv2.line(frame, left_foot[human_id][k], left_foot[human_id][k+1], human_color, 2+k*7)
        #                 left_foot[human_id][k] = left_foot[human_id][k+1]
        #             del left_foot[human_id][-1]

        #         # right handneck
        #         point = (anchors[16][0], anchors[16][1]) 
        #         right_foot[human_id].append(point)
        #         if (len(right_foot[human_id])>10):
        #             for k in range(1,10):
        #                 if -80 < (right_foot[human_id][k+1][0] - right_foot[human_id][k][0]) < 80:
        #                     if -80 < (right_foot[human_id][k+1][1] - right_foot[human_id][k][1]) < 80: 
        #                         frame = cv2.line(frame, right_foot[human_id][k], right_foot[human_id][k+1], human_color, 2+k*7)
        #                 right_foot[human_id][k] = right_foot[human_id][k+1]
        #             del right_foot[human_id][-1]

        # frame = cv2.addWeighted(back_frame,0.3,frame,0.7,0)

        out.write(frame)
        i += 1

        if i>1000:break
        if i > video.frame.getLength(): break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    my_clip = mpe.VideoFileClip(out_nosound_path)
    audio_background = mpe.VideoFileClip(in_video_path)
    my_clip.audio = audio_background.audio
    my_clip.write_videofile(out_video_path,
        codec='libx264', 
        audio_codec='aac', 
        temp_audiofile='temp-audio.m4a', 
        remove_temp=True)

    my_clip.close()
    audio_background.close()

    end = time.time()
    print("Complete animation effect generation")
    print("It took {} sec" .format(end - start))

