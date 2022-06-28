import math
import os.path as osp
import os
import pandas as pd
import json
import text_utils as  utils
import nltk

import pickle
import numpy as np
from copy import deepcopy


base_vocab = {'PAD': 0, 'UNK' : 1,
              'SOS': 2, 'EOS' : 3,
              'SEQ': 4, 'CLS' : 5,
              'QUE': 6,  'ANS' : 7}

def load_file(file_name):
    annos = None
    if osp.splitext(file_name)[-1] == '.csv':
        return pd.read_csv(file_name)
    with open(file_name, 'r') as fp:
        if osp.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if osp.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos

def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    input_paths = []
    try:
        annotation = pd.read_csv(args.annotation_file.format(args.question_type), encoding='utf-8',delimiter='\t')
    except UnicodeDecodeError as e:
        print('Use gbk code')
        annotation = pd.read_csv(args.annotation_file.format(args.question_type), encoding='gbk', delimiter='\t')
    gif_names = list(annotation['gif_name'])
    keys = list(annotation['key'])
    print("Number of questions: {}".format(len(gif_names)))
    for idx, gif in enumerate(gif_names):
        gif_abs_path = os.path.join(args.video_dir, ''.join([gif, '.gif']))
        input_paths.append((gif_abs_path, keys[idx]))
    input_paths = list(set(input_paths))
    print("Number of unique videos: {}".format(len(input_paths)))

    return input_paths


def multichoice_encoding_data(args,
                              vocab, questions, video_names, video_ids,
                              answers, ans_candidates,captions,
                              question_ids,
                              question_types,
                              mode='train',
                              padding = True):
    # Encode all questions
    print('Encoding data')

    # question encode
    questions_encoded = []
    questions_len = []
    # questio id for denote question type with video_id
    question_ids_tbw = []
    question_types_tbw = []

    # caption encode
    captions_encoded = []
    re_captions_encoded = []  # 反向caption
    captions_len = []

    # answer candidate encode
    all_answer_cands_encoded = []
    all_answer_cands_len = []

    # [question;candidate] condate the question and candidate
    # (N, 5, q_len)
    all_question_answer_encoded = []
    # (N, 5,)
    all_question_answer_len = []

    # video information
    video_ids_tbw = []
    video_names_tbw = []

    correct_answers = []
    for idx, question in enumerate(questions):
        question = question.lower()
        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))

        caption = captions[idx].lower()
        caption_tokens = nltk.word_tokenize(caption)
        caption_encoded = utils.encode(caption_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        captions_encoded.append(caption_encoded)
        captions_len.append(len(caption_encoded))

        # re-caption  '<EOS> answer question <SOS>'
        re_caption = caption[::-1]
        re_caption_tokens = nltk.word_tokenize(re_caption)
        re_caption_encoded = utils.encode(re_caption_tokens, vocab['question_token_to_idx'], allow_unk=True)
        re_captions_encoded.append(re_caption_encoded)

        video_names_tbw.append(video_names[idx])
        video_ids_tbw.append(video_ids[idx])

        question_types_tbw.append(question_types[idx])
        question_ids_tbw.append(question_ids[idx])

        # grounthtruth
        answer = int(answers[idx])
        correct_answers.append(answer)

        # answer candidates
        candidates = ans_candidates[idx]
        candidates_encoded = []
        candidates_len = []

        question_candidate_encoded = []
        question_candidate_len = []

        for ans in candidates:
            if not isinstance(ans, str) and math.isnan(ans):
                ans = "UNK"
            else:
                ans = ans.lower()
            ans_tokens = nltk.word_tokenize(ans)
            cand_encoded = utils.encode(ans_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            candidates_encoded.append(cand_encoded)
            candidates_len.append(len(cand_encoded))

            # encode the question candidate
            question_candidate = question + ' SEQ ' + ans
            question_candidate_tokens = nltk.word_tokenize(question_candidate)
            question_candidate_encoded.append(utils.encode(question_candidate_tokens,
                                                           vocab['question_answer_token_to_idx'], allow_unk=True))
            question_candidate_len.append(len(question_candidate_tokens))

        all_question_answer_encoded.append(question_candidate_encoded)
        all_question_answer_len.append(question_candidate_len)

        all_answer_cands_encoded.append(candidates_encoded)
        all_answer_cands_len.append(candidates_len)

    if padding:
        # Pad encoded questions
        max_question_length = max(len(x) for x in questions_encoded)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['question_answer_token_to_idx']['PAD'])

        # Pad encoded captions
        max_caption_length = max(len(x) for x in captions_encoded)
        for cap in captions_encoded:
            while len(cap) < max_caption_length:
                cap.append(vocab['question_answer_token_to_idx']['PAD'])

        max_caption_length = max(len(x) for x in re_captions_encoded)
        for cap in re_captions_encoded:
            while len(cap) < max_caption_length:
                cap.append(vocab['question_answer_token_to_idx']['PAD'])

        questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
        questions_len = np.asarray(questions_len, dtype=np.int32)

        captions_encoded = np.asarray(captions_encoded, dtype=np.int32)
        captions_len = np.asarray(captions_len, dtype=np.int32)
        re_captions_encoded = np.asarray(re_captions_encoded, dtype=np.int32)

        print("max question length", max_question_length)
        print("max caption length", max_caption_length)
        print("padded questions_encoded shape",questions_encoded.shape)
        print("padded captions_encoded shape", captions_encoded.shape)
        print("padded reverse caption encoded", re_captions_encoded.shape)

        # Pad encoded answer candidates
        max_answer_cand_length = max(max(len(x) for x in candidate) for candidate in all_answer_cands_encoded)
        for ans_cands in all_answer_cands_encoded:
            for ans in ans_cands:
                while len(ans) < max_answer_cand_length:
                    ans.append(vocab['question_answer_token_to_idx']['PAD'])
        all_answer_cands_encoded = np.asarray(all_answer_cands_encoded, dtype=np.int32)
        all_answer_cands_len = np.asarray(all_answer_cands_len, dtype=np.int32)
        print("max answer length", max_answer_cand_length)
        print("padded answer shape ",all_answer_cands_encoded.shape)

        # Pad encoded question_answer
        max_question_answer_length = max(max(len(x) for x in candidate) for candidate in all_question_answer_encoded)
        for ans_cands in all_question_answer_encoded:
            for ans in ans_cands:
                while len(ans) < max_question_answer_length:
                    ans.append(vocab['question_answer_token_to_idx']['PAD'])

        all_question_answer_encoded = np.asarray(all_question_answer_encoded, dtype=np.int32)
        all_question_answer_len = np.asarray(all_question_answer_len, dtype=np.int32)
        print("max question_answer length", max_question_answer_length)
        print("padded question_answer shape ", all_question_answer_encoded.shape)


    glove_matrix = None
    if mode in ['train']:
        token_itow = {i: w for w, i in vocab['question_answer_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print("glove_matrix.shape" ,glove_matrix.shape)

    print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids_tbw,
        'question_type': question_types_tbw,
        'captions' : captions_encoded,
        're_captions': re_captions_encoded,
        'captions_len' : captions_len,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'question_answes' : all_question_answer_encoded,
        'question_answer_len' : all_question_answer_len,
        'ans_candidates': all_answer_cands_encoded,
        'ans_candidates_len': all_answer_cands_len,
        'answers': correct_answers,
        'glove': glove_matrix,
    }

    with open(args.output_pt.format(args.dataset, args.question_type, mode), 'wb') as f:
        pickle.dump(obj, f)

def make_video_caption(questions, answers, candidates) -> list:
    captions = []
    for i, question in enumerate(questions):
        answer_sentence = candidates[i][int(answers[i])]
        # print(question, answer_sentence)
        caption = 'SOS ' + question + ' SEP ' + answer_sentence + ' EOS'
        captions.append(caption)
    return captions

def process_questions_mulchoices(args):
    print('Loading data')
    try:
        csv_data = pd.read_csv(args.annotation_file.format(args.dataset, args.split), encoding='utf-8')
    except UnicodeDecodeError as e:
        print('error {}, use gbk code'.format(e))
        csv_data = pd.read_csv(args.annotation_file.format(args.dataset, args.split), encoding='gbk')

    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]

    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['video'])
    video_ids = list(csv_data['video'])
    question_ids = list(csv_data['qid'])
    question_types = list(csv_data['type'])
    ans_candidates = np.asarray(
        [csv_data['a0'], csv_data['a1'], csv_data['a2'], csv_data['a3'], csv_data['a4']])
    # (5, num_ques) -> (num_ques,5)
    ans_candidates = ans_candidates.transpose()
    # (num_ques,)
    captions = make_video_caption(questions, answers, ans_candidates)
    print('number of questions: %s' % len(questions))

    # Either create the vocab or load it from disk
    if args.split in ['train']:
        print('Building vocab')

        answer_token_to_idx = deepcopy(base_vocab)
        question_answer_token_to_idx = deepcopy(base_vocab)

        for candidates in ans_candidates:
            for ans in candidates:
                if not isinstance(ans,str) and math.isnan(ans):
                    ans = "UNK"
                else:
                    ans = ans.lower()
                for token in nltk.word_tokenize(ans):
                    if token not in answer_token_to_idx:
                        answer_token_to_idx[token] = len(answer_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = deepcopy(base_vocab)

        for i, q in enumerate(questions):
            question = q.lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)

        print('Get question_token_to_idx {}'.format(len(question_token_to_idx)))
        print('Get question_answer_token_to_idx {}'.format(len(question_answer_token_to_idx)))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': question_answer_token_to_idx,
        }

        print('Write into %s' % args.vocab_json.format(args.dataset, args.question_type))
        with open(args.vocab_json.format(args.dataset, args.question_type), 'w') as f:
            json.dump(vocab, f, indent=4)

        multichoice_encoding_data(args, vocab,
                                  questions, video_names,
                                  video_ids, answers,
                                  ans_candidates,captions,
                                  question_ids,
                                  question_types,
                                  mode=args.split)
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.question_type), 'r') as f:
            vocab = json.load(f)
        multichoice_encoding_data(args, vocab,
                                  questions, video_names,
                                  video_ids, answers,
                                  ans_candidates, captions,
                                  question_ids,
                                  question_types,
                                  mode=args.split)

