#!/usr/bin/env bash

python3 main.py --input_video_path dance-effect-source/I-DLE-02.mp4 \
    --input_json_path dance-effect-source/I-DLE-02_matching.json \
    --output_video_path dance-effect-output/I-DLE-02-output.mp4 \
    --output_nosound_video_path dance-effect-output/I-DLE-02_nosoud.mp4 \
    --step_detection \
    --random_effect \