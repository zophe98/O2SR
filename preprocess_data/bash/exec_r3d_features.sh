#!/bin/bash
server=$1
start_idx=$2
end_idx=$3
gpus=$4
#dataset="tgif-qa"
dataset=$5
num_bboxes=5
model_name="resnext101"
if [ $server = 2080 ]; then
  model_ckpt="/data2/zf/Datasets/VQA/pretrained_model/resnext/resnext-101-kinetics.pth";
  if [ $dataset = "tgif-qa" ]; then
      sample_nums=35
      video_dir="/data2/zf/Datasets/VQA/TGIF_QA/tgif_full/gifs";
      out_dir="/data2/zf/Datasets/VQA/TGIF_QA/output";
      detect_dir="/data2/zf/Datasets/VQA/TGIF_QA/output/maskrcnn_old_renew";
  elif [ $dataset = "msvd-qa" ]; then
      sample_nums=20
      video_dir="/data2/zf/Datasets/VQA/MSVD/YoutubeClips"
      out_dir="/data2/zf/Datasets/VQA/MSVD-QA/output"
      detect_dir="/data2/zf/Datasets/VQA/MSVD-QA/output/maskrcnn_old_renew"
  elif [ $dataset = "msrvtt-qa" ]; then
      sample_nums=20
      video_dir="/data2/zf/Datasets/VQA/MSRVTT-QA/video"
      out_dir="/data2/zf/Datasets/VQA/MSRVTT-QA/output"
      detect_dir="/data2/zf/Datasets/VQA/MSRVTT-QA/output/maskrcnn_old_renew"
  fi
elif [ $server = 3090 ]; then
    model_ckpt="/data1/zf/datasets/pre-trained-model/resnext-101-kinetics.pth";
    if [ $dataset = "tgif-qa" ]; then
        sample_nums=35
        video_dir="/data1/zf/datasets/QA/TGIF_QA/tgif_full/gifs";
        out_dir="/data1/zf/datasets/QA/TGIF_QA/output";
        detect_dir="/data1/zf/datasets/QA/TGIF_QA/output/maskrcnn_old_renew";
    elif [ $dataset = "msvd-qa" ]; then
        sample_nums=20
        video_dir="/data1/zf/datasets/QA/MSVD/YoutubeClips"
        out_dir="/data1/zf/datasets/QA/MSVD-QA/output"
        detect_dir="/data1/zf/datasets/QA/MSVD-QA/output/maskrcnn_old_renew"
    elif [ $dataset = "msrvtt-qa" ]; then
        sample_nums=20
        video_dir="/data1/zf/datasets/QA/MSRVTT-QA/video"
        out_dir="/data1/zf/datasets/QA/MSRVTT-QA/output"
        detect_dir="/data1/zf/datasets/QA/MSRVTT-QA/output/maskrcnn_old_renew"
    fi
elif [ $server = 2082 ]; then
    model_ckpt="/data2/zf/Pre_trained_model/resnext-101-kinetics.pth";
    if [ $dataset = "next-qa" ]; then
      sample_nums=16
#      video_dir="/data2/zf/VQA/NExT-QA/VidOR_DATASET/all_video"
      video_dir="/data2/zf/VQA/NExT-QA/VidOR_DATASET/MissingVideos/all_videos"
      detect_dir="/data2/zf/VQA/NExT-QA/output/maskrcnn_new3"
      out_dir="/data2/zf/VQA/NExT-QA/output"
    fi
fi

CUDA_VISIBLE_DEVICES=${gpus} \
  python ../scripts/extract_r3d101_features_with_RoIAlign.py \
  --dataset ${dataset} \
  --video_dir ${video_dir} \
  --out_dir ${out_dir} \
  --start_idx ${start_idx} \
  --end_idx ${end_idx} \
  --detect_dir ${detect_dir} \
  --model_ckpt ${model_ckpt} \
  --num_clips ${sample_nums} \
  --num_bboxes ${num_bboxes} \
  --model ${model_name} \
