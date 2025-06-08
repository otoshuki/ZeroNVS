#!/bin/bash
# Set these yourself!

# IMAGE_PATH='rob.png'
# FOV=60.0
# ELEVATION_DEG=30.0
# SCALE=0.8

IMAGE_PATH='smallmoto.png'
FOV=52.55
ELEVATION_DEG=31.0
SCALE=0.7

python launch.py --config configs/custom_config.yaml --train --gpu 0 \
    data.image_path=$IMAGE_PATH \
    data.default_elevation_deg=$ELEVATION_DEG \
    data.default_fovy_deg=$FOV \
    data.random_camera.fovy_range="[$FOV,$FOV]" \
    data.random_camera.eval_fovy_deg=$FOV \
    system.guidance.precomputed_scale=$SCALE \
