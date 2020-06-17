import json

from DataParse.parse import Video
from BehaviorAnalysis.formation import getCenterChunk, getFormationChunk
from BehaviorAnalysis.behavior import getLeftFeetMov, getRightFeetMov
from AnimationEffect.animation_effect import animation_effect

in_video_path = 'dance-effect-source/(G)I-DLE-01.mp4'
out_video_path = '(G)I-DLE-01-output.mp4'

json_path = 'dance-effect-source/(G)I-DLE-01_matching.json'

print("Generating json deserialization")

video = Video(json_path)

print("Complete json deserialization")

print("There are {} people in this video" .format(video.hum_cnt))
print("This video consists of {} frames" .format(video.frame.getLength()))

formation = getFormationChunk(video, 100)
centerChunk = getCenterChunk(video)

rightFeetMov = getRightFeetMov(video)
leftFeetMov = getLeftFeetMov(video)

animation_effect(video, in_video_path, out_video_path)
