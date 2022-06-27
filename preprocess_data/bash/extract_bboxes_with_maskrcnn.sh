server=$1
start_idx=$2
end_idx=$3
gpus=$4
confidence_threshold=0.0
step_detection=1
dataset=$5
if [ $dataset = "tgif" ]; then
    video_ext="gif"
    sample_nums=35
    if [ $server -eq 3090 ]; then
      video_root="/data1/zf/datasets/QA/TGIF_QA/tgif_full/gifs"
      output_root="/data1/zf/datasets/QA/TGIF_QA/output"
    elif [ $server -eq 2080 ]; then
      video_root="/data2/zf/Datasets/VQA/TGIF_QA/tgif_full/gifs"
      output_root="/data2/zf/Datasets/VQA/TGIF_QA/output"
    fi
elif [ $dataset = "msvd" ]; then
    video_ext="avi"
    sample_nums=20
    if [ $server -eq 3090 ]; then
        video_root="/data1/zf/datasets/QA/MSVD/YoutubeClips"
        output_root="/data1/zf/datasets/QA/MSVD-QA/output"
    fi
elif [ $dataset = "msrvtt" ]; then
    video_ext="mp4"
    sample_nums=20
    if [ $server -eq 3090 ]; then
        video_root="/data1/zf/datasets/QA/MSRVTT-QA/video"
        output_root="/data1/zf/datasets/QA/MSRVTT-QA/output"
    fi
elif [ $dataset = 'next' ]; then
    video_ext="mp4"
    sample_nums=16
    if [ $server -eq 2082 ]; then
#        video_root="/data2/zf/VQA/NExT-QA/VidOR_DATASET/all_video"
        video_root="/data2/zf/VQA/NExT-QA/VidOR_DATASET/MissingVideos/all_videos"
        output_root="/data2/zf/VQA/NExT-QA/output"
    fi
fi


CUDA_VISIBLE_DEVICES=${gpus} \
  python ../scripts/extract_bboxes_with_maskrcnn.py \
    --dataset $dataset \
    --video_ext ${video_ext} \
    --video_root ${video_root} \
    --output_root ${output_root} \
    --start_idx ${start_idx} \
    --end_idx ${end_idx} \
    --sample_nums ${sample_nums} \
    --step_detection ${step_detection} \
    --confidence-threshold ${confidence_threshold}