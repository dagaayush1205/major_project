#!/bin/bash

set -e

echo "======================================"
echo "Starting YOLO11"
echo "======================================"

python detect.py \
  --model yolo11 \
  --images val2017 \
  --out yolo11_val2017.json

echo
echo "Evaluating YOLO11"

python evaluate.py \
  --gt annotations/instances_val2017.json \
  --pred yolo11_val2017.json

echo "======================================"
echo "Starting Faster R-CNN"
echo "======================================"

python detect.py \
  --model fasterrcnn \
  --images val2017 \
  --out fasterrcnn_val2017.json

echo
echo "Evaluating Faster R-CNN"

python evaluate.py \
  --gt annotations/instances_val2017.json \
  --pred fasterrcnn_val2017.json

echo "======================================"
echo "Starting RT-DETR"
echo "======================================"

python detect.py \
  --model rtdetr \
  --images val2017 \
  --out rtdetr_val2017.json

echo
echo "Evaluating RT-DETR"

python evaluate.py \
  --gt annotations/instances_val2017.json \
  --pred rtdetr_val2017.json

echo
echo "======================================"
echo "ALL THREE MODELS COMPLETE"
echo "======================================"
