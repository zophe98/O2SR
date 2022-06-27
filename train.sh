server=$1
task=$2
gpus=$3
model_name=$4
dataset_name=$5
exp_name=$6
debug=$7
exp_date="0118"
log_file="${dataset_name}_${exp_name}_train_${exp_date}_${task}_${model_name}.log"
if [ $dataset_name = 'tgif-qa' ]; then
    cfg_file="configs/tgif_qa_${task}.yml"
elif [ $dataset_name = 'msvd-qa' ]; then
    cfg_file="configs/msvd_qa_frameqa.yml"
elif [ $dataset_name = 'msrvtt-qa' ]; then
    cfg_file="configs/msrvtt_qa_frameqa.yml"
elif [ $dataset_name = 'next-qa' ]; then
    cfg_file="configs/next_qa_action.yml"
else
  echo "wrong dataset " + $dataset_name
fi

if [ $debug = 'debug' ]; then
    DEBUG=1
else
    DEBUG=0
fi
if [ $DEBUG = 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} \
      python ./main.py --cfg ${cfg_file} \
      --dataset ${dataset_name} \
      --server ${server} \
      --gpus ${gpus} \
      --exp_prefix ${exp_name} \
      --model_name ${model_name} 2>&1
else
  CUDA_VISIBLE_DEVICES=${gpus} \
    nohup python ./main.py --cfg ${cfg_file} \
    --dataset ${dataset_name} \
    --server ${server} \
    --gpus ${gpus} \
    --exp_prefix ${exp_name} \
    --model_name ${model_name} 2>&1 \
     >> ${log_file} &
fi


