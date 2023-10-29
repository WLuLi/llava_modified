#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /mnt/disk0/wangqijie/checkpoints/llava-v1.5-13b \
    --question-file /mnt/disk0/wangqijie/eval/vizwiz/llava_test.jsonl \
    --image-folder /mnt/disk0/wangqijie/eval/vizwiz/test \
    --answers-file /mnt/disk0/wangqijie/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /mnt/disk0/wangqijie/eval/vizwiz/llava_test.jsonl \
    --result-file /mnt/disk0/wangqijie/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --result-upload-file /mnt/disk0/wangqijie/eval/vizwiz/answers_upload/llava-v1.5-13b.json
