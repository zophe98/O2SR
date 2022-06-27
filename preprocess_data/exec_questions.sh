answer_top=4000
#dataset="tgif-qa"
#server=2080
dataset=$1    # "tgif-qa", "msvd-qa", "msrvtt-qa", "next-qa"
server=$2
split=$3      # train,test, (val)
question_type=$4 # frameqa, count, action, transition

if [ $server -eq 2080 ]; then
  glove_pt="/data2/zf/Datasets/VQA/pretrained_model/Glove/glove.840.300d.pkl"
elif [ $server -eq 2082 ]; then
  glove_pt="/data2/zf/Pre_trained_model/glove.840.300d.pkl"
else
  echo "不支持的服务器"
fi

echo "bash preprocess ${dataset}"

input_ann="../datasets/orignal/{}"
output_pt="../datasets/output/{}"
vocab_json="../datasets/output/{}/{}_vocab.json"

python preprocess_questions.py \
  --dataset $dataset \
  --glove_pt $glove_pt \
  --input_ann $input_ann \
  --output_pt $output_pt \
  --vocab_json $vocab_json \
  --split $split \
  --question_type $question_type >> "0315_preprocess_${dataset}.log"

