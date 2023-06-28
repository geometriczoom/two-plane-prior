#!/bin/bash

source experiments/path.sh

expName="KDE_TPP"
cfgDir="$reposDir/two-plane-prior/configs"
nGPU=2

python tools/test.py \
    --gpu-test-pre \
    --data-dir $dataDir \
    --out-dir "$outDir/$expName/test_FT" \
    --weights "checkpoints/KDE_TPP.pth" \
    --dataset-config "$cfgDir/datasets/avhd_gt.py" \
    --model-config "$cfgDir/models/fovea_faster_rcnn_r50_fpn.py" \
    --schedule-config "$cfgDir/schedules/schedule_poly.py" \
    --runtime-config "$cfgDir/default_runtime.py" \
    --preprocess-scale "1" \
    --reg-decoded-bbox \
    --use-fovea \
    --grid-net-cfg "$cfgDir/models/cuboid_global_kde_grid.py" \
    # --vis-options \
    #     input_image="$outDir/$expName/test_FT/vis/input_image" \
    #     warped_image="$outDir/$expName/test_FT/vis/warped_image" \
    #     saliency="$outDir/$expName/test_FT/vis/saliency" \
