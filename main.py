import json
import argparse

from DataParse.parse import Video
from BehaviorAnalysis.formation import getCenterChunk, getFormationChunk
from BehaviorAnalysis.behavior import getFootVector, selectFootVectorChunk, getLeftFeetMov, getRightFeetMov
from AnimationEffect.animation_effect import animation_effect

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_video_path",
        default= None,
        type=str,
        required=True,
        help="path of input video file",
    )

    parser.add_argument(
        "--input_json_path",
        default= None,
        type=str,
        required=True,
        help="path of input json file",
    )

    parser.add_argument(
        "--output_video_path",
        default= None,
        type=str,
        required=True,
        help="path of output video file",
    )

    parser.add_argument(
        "--output_nosound_video_path",
        default= None,
        type=str,
        required=True,
        help="path of output video without sound file",
    )

    parser.add_argument(
        "--step_detection",
        action="store_true",
        help="detects step of dancers",
    )

    parser.add_argument(
        "--random_effect",
        action="store_true",
        help="generate random effects",
    )

    args = parser.parse_args()

    print("Generating json deserialization")

    video = Video(args.input_json_path)

    print("Complete json deserialization")

    print("There are {} people in this video" .format(video.hum_cnt))
    print("This video consists of {} frames" .format(video.frame.getLength()))

    formation = getFormationChunk(video, 100)
    centerChunk = getCenterChunk(video)

    getFootVector(video)
    selectFootVectorChunk(video, 30)
    # rightFeetMov = getRightFeetMov(video)
    # leftFeetMov = getLeftFeetMov(video)

    animation_effect(video, args, formation)


if __name__ == "__main__":
    main()