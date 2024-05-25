#!/bin/bash

N=50
MODEL_NAME=DeepLabV3+_WideResNet38
IO=/media/jurica/c2d76687-c280-496a-91dc-acee1afa83ab1/io/$MODEL_NAME
IO_LAF=$IO/laf_eval
IO_FS=$IO/fs_eval

LOG_FOLDER_LAF=$IO_LAF/logs
LOG_FOLDER_FS=$IO_FS/logs

RESULTS_FOLDER_LAF=$IO_LAF/results/entropy_counts_per_pixel
RESULTS_FOLDER_FS=$IO_FS/results/entropy_counts_per_pixel

METRICS_IO_FOLDER_LAF=$IO_LAF/metaseg_io/metrics/epoch_4_alpha_0.9_t0.7

mkdir -p $LOG_FOLDER_LAF
mkdir -p $LOG_FOLDER_FS

for ((i=38; i<=N; ++i)); do
  mkdir -p $LOG_FOLDER_LAF/$i
  mkdir -p $LOG_FOLDER_FS/$i

  python3 ood_training.py | tee $LOG_FOLDER_LAF/$i/train_output.txt
  
  python3 meta_classification.py --START_NUM_IMAGES=0 --STOP_NUM_IMAGES=600 | tee
  python3 meta_classification.py --START_NUM_IMAGES=600 --STOP_NUM_IMAGES=1203 | tee
 
  python3 meta_classification.py --segment_search --fp_removal | tee
  cp $METRICS_IO_FOLDER_LAF/meta_classified_logistic.p $LOG_FOLDER_LAF/$i/
  cp $METRICS_IO_FOLDER_LAF/meta_classifier_predictions_logistic.p $LOG_FOLDER_LAF/$i/
  python3 evaluation.py --pixel_eval | tee $LOG_FOLDER_LAF/$i/pixel_evaluation.txt
  python3 evaluation.py --segment_eval | tee $LOG_FOLDER_LAF/$i/segment_evaluation_logistic.txt
  
  python3 meta_classification.py --fp_removal --METACLASSIFIER=NN | tee
  cp $METRICS_IO_FOLDER_LAF/meta_classifier_predictions_nn.p $LOG_FOLDER_LAF/$i/
  cp $METRICS_IO_FOLDER_LAF/meta_classified_nn.p $LOG_FOLDER_LAF/$i/
  python3 evaluation.py --segment_eval | tee $LOG_FOLDER_LAF/$i/segment_evaluation_nn.txt
  
  python3 meta_classification.py --metaseg_prepare --START_NUM_IMAGES=0 --STOP_NUM_IMAGES=30 --VALSET=Fishyscapes | tee
  python3 evaluation.py --pixel_eval --VALSET=Fishyscapes | tee $LOG_FOLDER_FS/$i/pixel_evaluation.txt
  
  python3 cityscapes_valset_eval.py --TRAIN_NUM=$i | tee
 
  cp $RESULTS_FOLDER_LAF/epoch_4_alpha_0.9.p $LOG_FOLDER_LAF/$i/
  cp $RESULTS_FOLDER_FS/epoch_4_alpha_0.9.p $LOG_FOLDER_FS/$i/

  find $RESULTS_FOLDER_LAF -name '*.p' -delete
  find $RESULTS_FOLDER_FS -name '*.p' -delete
done

