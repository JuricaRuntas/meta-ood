#!/bin/bash

N=10
MODEL_NAME=DeepLabV3+_WideResNet38
IO=/media/jurica/c2d76687-c280-496a-91dc-acee1afa83ab1/io/$MODEL_NAME
LOG_FOLDER_LAF=$IO/laf_eval/logs
LOG_FOLDER_FS=$IO/fs_eval/logs

mkdir -p $LOG_FOLDER_LAF
mkdir -p $LOG_FOLDER_FS

for ((i=1; i<=N; ++i)); do
  mkdir -p $LOG_FOLDER_LAF/$i
  mkdir -p $LOG_FOLDER_FS/$i

  python3 ood_training.py | tee $LOG_FOLDER_LAF/$i/train_output.txt
  python3 meta_classification.py --metaseg_prepare | tee
  python3 meta_classification.py --metaseg_prepare --VALSET=Fishyscapes | tee
  python3 evaluation.py --pixel_eval | tee $LOG_FOLDER_LAF/$i/pixel_evaluation.txt
  python3 evaluation.py --pixel_eval --VALSET=Fishyscapes | tee $LOG_FOLDER_LAF/$i/pixel_evaluation.txt
  python3 cityscapes_valset_eval.py --TRAIN_NUM=$i | tee
done

