#!/usr/bin/env bash

python3 main.py --input_video_path dance-effect-source/boywithluv.mp4 \
    --input_json_path dance-effect-source/boywithluv_matching.json \
    --output_video_path dance-effect-output/boywithluv.mp4 \
    --output_nosound_video_path dance-effect-output/boywithluv_nosoud.mp4 \
    --step_detection \
    --random_effect \