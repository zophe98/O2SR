server=$1
start_idx=$2
end_idx=$3
gpus=$4
#dataset="tgif"
dataset=$5
num_bboxes=5
model_name="resnet152"

if [ $dataset = "tgif" ]; then
    video_ext="gif"
    sample_nums=35
    model_name="resnet152"
    if [ $server -eq 2080 ]; then
      video_root="/data2/zf/Datasets/VQA/TGIF_QA/tgif_full/gifs"
      detect_root="/data2/zf/Datasets/VQA/TGIF_QA/output/maskrcnn_old_renew"
      output_root="/data2/zf/Datasets/VQA/TGIF_QA/output"
    elif [ $server -eq 3090 ]; then
      video_root="/data1/zf/datasets/QA/TGIF_QA/tgif_full/gifs"
      detect_root="/data1/zf/datasets/QA/TGIF_QA/output/maskrcnn_old_renew"
      output_root="/data1/zf/datasets/QA/TGIF_QA/output/"
    fi
elif [ $dataset = "msvd" ]; then
    video_ext="avi"
    sample_nums=20
    model_name="resnet152"
    if [ $server -eq 2080 ]; then
      video_root="/data2/zf/Datasets/VQA/MSVD/YoutubeClips"
      detect_root="/data2/zf/Datasets/VQA/MSVD-QA/output/maskrcnn"
      output_root="/data2/zf/Datasets/VQA/MSVD-QA/output"
    elif [ $server -eq 3090 ]; then
      video_root="/data1/zf/datasets/QA/MSVD/YoutubeClips"
      detect_root="/data1/zf/datasets/QA/MSVD-QA/output/maskrcnn_old_renew"
      output_root="/data1/zf/datasets/QA/MSVD-QA/output"
    fi
elif [ $dataset = "msrvtt" ]; then
    video_ext="mp4"
    sample_nums=20
    model_name="resnet152"
    if [ $server -eq 2080 ]; then
      video_root="/data2/zf/Datasets/VQA/MSRVTT-QA/video"
      detect_root="/data1/zf/datasets/QA/MSRVTT-QA/output/maskrcnn_old_renew"
      output_root="/data2/zf/Datasets/VQA/MSRVTT-QA/output"
    elif [ $server -eq 3090 ]; then
      video_root="/data1/zf/datasets/QA/MSRVTT-QA/video"
      output_root="/data1/zf/datasets/QA/MSRVTT-QA/output"
      detect_root="/data1/zf/datasets/QA/MSRVTT-QA/output/maskrcnn_old_renew"
    fi
elif [ $dataset = "next" ]; then
    video_ext="mp4"
    sample_nums=16
    model_name="resnet152"
    if [ $server -eq 2082 ]; then
#      video_root="/data2/zf/VQA/NExT-QA/VidOR_DATASET/all_video"
      video_root="/data2/zf/VQA/NExT-QA/VidOR_DATASET/MissingVideos/all_videos"
      detect_root="/data2/zf/VQA/NExT-QA/output/maskrcnn_new3"
      output_root="/data2/zf/VQA/NExT-QA/output"
    fi
fi

CUDA_VISIBLE_DEVICES=${gpus} \
  python ../scripts/extract_resnet152_features_with_RoIAlign.py \
    --dataset $dataset \
    --video_ext ${video_ext} \
    --video_root ${video_root} \
    --detect_root ${detect_root} \
    --output_root ${output_root} \
    --start_idx ${start_idx} \
    --end_idx ${end_idx} \
    --sample_nums ${sample_nums} \
    --num_bboxes ${num_bboxes} \
    --model_name ${model_name}