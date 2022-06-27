import argparse
import numpy as np
import os

from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa
from datautils import next_qa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='tgif-qa',
        choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa','next-qa'],
        type=str)

    parser.add_argument(
        '--answer_top',
        default=4000,
        type=int)
    parser.add_argument(
        '--glove_pt',
        help='glove pickle file, '
        'should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')

    parser.add_argument(
        '--input_ann',
        type=str,
        default='。。/datasets/{}',
        help='include all the original data, like datasets/orignal/{tgif-qa}/Train_action_question.csv')

    parser.add_argument(
        '--output_pt',
        type=str, default='../datasets/output/{}/{}_{}_questions.pkl',
        help="datasets/output/{tgif-qa}/{question_type}_{data_split}_questions.pkl")

    parser.add_argument(
        '--vocab_json',
        type=str,
        default='../datasets/{}/{}_vocab.json',
        help="datasets/{tgif-qa}/{question_type}_vocab.json")

    parser.add_argument(
        '--split', choices=['train', 'val', 'test'])

    parser.add_argument('--question_type',
        choices=['frameqa', 'action', 'transition', 'count', 'none'], default='none')
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)

    if not os.path.exists(args.output_pt.format(args.dataset)):
        os.mkdir(args.output_pt.format(args.dataset))

    # dataset/{}
    args.output_pt = args.output_pt + "/{}_{}_questions.pkl"

    if args.dataset == 'tgif-qa':
        # '/{datasetname}/{split}_{question_type}_question.csv'
        args.annotation_file = args.input_ann + '/{}_{}_question.csv'

        print("input annotion file {}".format(args.annotation_file))
        print("preprocess dataset:{}, type:{}, split:{}".format(args.dataset,
                                                                args.question_type,
                                                                args.split))
        if args.question_type in ['frameqa', 'count']:
            tgif_qa.process_questions_openended(args)
        else:
            tgif_qa.process_questions_mulchoices(args)

    elif args.dataset == 'msrvtt-qa':
        # '/{datasetname}/{split}_qa.json'
        args.annotation_file = args.input_ann + '/{}_qa.json'
        args.annotation_file = args.annotation_file.format(args.dataset,args.split)

        print("preprocess dataset:{}, type:{}, split:{}".format(args.dataset,
                                                                args.question_type,
                                                                args.split))
        print("input annotion file {}".format(args.annotation_file))
        msrvtt_qa.process_questions(args)

    elif args.dataset == 'msvd-qa':
        # '/{datasetname}/{split}_qa.json'
        args.annotation_file = args.input_ann + '/{}_qa.json'
        args.annotation_file = args.annotation_file.format(args.dataset, args.split)

        print("preprocess dataset:{}, type:{}, split:{}".format(args.dataset,
                                                                args.question_type,
                                                                args.split))
        print("input annotion file {}".format(args.annotation_file))

        msvd_qa.process_questions(args)
    elif args.dataset == 'next-qa':
        # '/{datasetname}/{split}.csv'
        args.annotation_file = args.input_ann + '/{}.csv'

        print("input annotion file {}".format(args.annotation_file))
        print("preprocess dataset:{}, type:{}, split:{}".format(args.dataset,
                                                                args.question_type,
                                                                args.split))
        if args.question_type in ['frameqa']:
            raise NotImplementedError
        else:
            next_qa.process_questions_mulchoices(args)

    print("preprocess {} finished.".format(args.split))