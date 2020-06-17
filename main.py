import json
import argparse

from DataParse.parse import Video
from BehaviorAnalysis.formation import getCenterChunk, getFormationChunk
from BehaviorAnalysis.behavior import getLeftFeetMov, getRightFeetMov
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

    args = parser.parse_args()

    in_video_path = args.input_video_path
    out_video_path = args.output_video_path
    json_path = args.input_json_path
    out_nosound_path = args.output_nosound_video_path

    print("Generating json deserialization")

    video = Video(json_path)

    print("Complete json deserialization")

    print("There are {} people in this video" .format(video.hum_cnt))
    print("This video consists of {} frames" .format(video.frame.getLength()))

    formation = getFormationChunk(video, 100)
    centerChunk = getCenterChunk(video)

    rightFeetMov = getRightFeetMov(video)
    leftFeetMov = getLeftFeetMov(video)

    animation_effect(video, in_video_path, out_video_path, out_nosound_path)


if __name__ == "__main__":
    main()