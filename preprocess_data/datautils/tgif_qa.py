import os
import pandas as pd
import json
from datautils import text_utils
import nltk

import pickle
import numpy as np
from copy import deepcopy

base_vocab = {'PAD': 0, 'UNK' : 1,
              'SOS': 2, 'EOS' : 3,
              'SEQ': 4, 'CLS' : 5,
              'QUE': 6,  'ANS' : 7}

def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    input_paths = []
    annotation = pd.read_csv(args.annotation_file.format(args.question_type), delimiter='\t')
    gif_names = list(annotation['gif_name'])
    keys = list(annotation['key'])
    print("Number of questions: {}".format(len(gif_names)))
    for idx, gif in enumerate(gif_names):
        gif_abs_path = os.path.join(args.video_dir, ''.join([gif, '.gif']))
        input_paths.append((gif_abs_path, keys[idx]))
    input_paths = list(set(input_paths))
    print("Number of unique videos: {}".format(len(input_paths)))

    return input_paths

def openeded_encoding_data(args,
                           vocab, questions, video_names,
                           video_ids, answers, captions, mode='train', padding=True):
    ''' Encode question tokens'''
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    question_ids = []
    captions_encoded = []
    re_captions_encoded = [] # 反向caption
    captions_len = []

    for idx, question in enumerate(questions):

        question = question.lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = text_utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)

        video_names_tbw.append(video_names[idx])
        video_ids_tbw.append(video_ids[idx])


        caption = captions[idx].lower()

        caption_tokens = nltk.word_tokenize(caption)
        caption_encoded = text_utils.encode(caption_tokens, vocab['question_token_to_idx'], allow_unk=True)
        captions_encoded.append(caption_encoded)
        captions_len.append(len(caption_encoded))


        re_caption = caption[::-1]
        re_caption_tokens = nltk.word_tokenize(re_caption)
        re_caption_encoded = text_utils.encode(re_caption_tokens, vocab['question_token_to_idx'], allow_unk=True)
        re_captions_encoded.append(re_caption_encoded)


        answer = answers[idx]
        if args.question_type == 'frameqa':
            if answer in vocab['answer_token_to_idx']:
                answer = vocab['answer_token_to_idx'][answer]
            else:
                answer = vocab['answer_token_to_idx']['UNK'] # 0

            # elif mode in ['train']:
            #     answer = 0
            # elif mode in ['val', 'test']:
            #     answer = 1
        else:
            # The original answer is [0,2,3,4,5,6,7,8,9,10]
            # In order to ensure the continuity of the answer, we set 0 -> 1
            answer = max(int(answers[idx]), 1)

        all_answers.append(answer)

    # It can also be put into the pad when batch is loaded later
    if padding:
        # Pad encoded questions
        max_question_length = max(len(x) for x in questions_encoded)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['question_token_to_idx']['PAD'])

        if len(captions_encoded) > 0:
            max_caption_length = max(len(x) for x in captions_encoded)
            for cpe in captions_encoded:
                while len(cpe) < max_caption_length:
                    cpe.append(vocab['question_token_to_idx']['PAD'])

        if len(re_captions_encoded) > 0:
            max_caption_length = max(len(x) for x in re_captions_encoded)
            for cpe in re_captions_encoded:
                while len(cpe) < max_caption_length:
                    cpe.append(vocab['question_token_to_idx']['PAD'])

        questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
        questions_len = np.asarray(questions_len, dtype=np.int32)

        captions_encoded = np.asarray(captions_encoded, dtype=np.int32)
        captions_len = np.asarray(captions_len, dtype=np.int32)

        re_captions_encoded = np.asarray(re_captions_encoded, dtype=np.int32)

        print("padded question encoded", questions_encoded.shape)
        print("padded caption encoded",  captions_encoded.shape)
        print("padded reverse caption encoded", re_captions_encoded.shape)

    glove_matrix = None
    if mode == 'train':
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print("glove_matrix.shape",glove_matrix.shape)

    print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'captions' : captions_encoded,
        're_captions' : re_captions_encoded,
        'captions_len' : captions_len,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'answers': all_answers,
        'glove': glove_matrix,
    }

    with open(args.output_pt.format(args.dataset, args.question_type, mode), 'wb') as f:
        pickle.dump(obj, f)

def multichoice_encoding_data(args,
                              vocab, questions, video_names, video_ids,
                              answers, ans_candidates,captions, mode='train',
                              padding=True):
    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    question_ids = []

    captions_encoded = []
    re_captions_encoded = []
    captions_len = []

    all_answer_cands_encoded = []
    all_answer_cands_len = []

    # [question;candidate] condate the question and candidate
    # (N, 5, q_len)
    all_question_answer_encoded = []
    # (N, 5,)
    all_question_answer_len = []

    video_ids_tbw = []
    video_names_tbw = []

    correct_answers = []
    for idx, question in enumerate(questions):
        question = question.lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = text_utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)

        caption = captions[idx].lower()
        caption_tokens = nltk.word_tokenize(caption)
        caption_encoded = text_utils.encode(caption_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        captions_encoded.append(caption_encoded)
        captions_len.append(len(caption_encoded))

        re_caption = caption[::-1]
        re_caption_tokens = nltk.word_tokenize(re_caption)
        re_caption_encoded = text_utils.encode(re_caption_tokens, vocab['question_token_to_idx'], allow_unk=True)
        re_captions_encoded.append(re_caption_encoded)

        video_names_tbw.append(video_names[idx])
        video_ids_tbw.append(video_ids[idx])
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
            ans = ans.lower()
            ans_tokens = nltk.word_tokenize(ans)
            cand_encoded = text_utils.encode(ans_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            candidates_encoded.append(cand_encoded)
            candidates_len.append(len(cand_encoded))

            # encode the question candidate
            question_candidate = question + ' SEQ ' + ans
            question_candidate_tokens = nltk.word_tokenize(question_candidate)
            question_candidate_encoded.append(text_utils.encode(question_candidate_tokens,
                                                                vocab['question_answer_token_to_idx'], allow_unk=True))
            question_candidate_len.append(len(question_candidate_tokens))

        all_question_answer_encoded.append(question_candidate_encoded)
        all_question_answer_len.append(question_candidate_len)

        all_answer_cands_encoded.append(candidates_encoded)
        all_answer_cands_len.append(candidates_len)

    # Pad encoded questions
    if padding:
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
        'question_id': question_ids,
        'captions' : captions_encoded,
        're_captions': re_captions_encoded,
        'captions_len' : captions_len,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'question_answes': all_question_answer_encoded,
        'question_answer_len': all_question_answer_len,
        'ans_candidates': all_answer_cands_encoded,
        'ans_candidates_len': all_answer_cands_len,
        'answers': correct_answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.dataset, args.question_type, mode), 'wb') as f:
        pickle.dump(obj, f)

def make_video_description(questions, answers, type='count', candidates = None, _replace=False):
    captions = []

    if type == 'count' or type == 'frameqa':
        for i,question in enumerate(questions):
            answer = str(answers[i])
            if _replace:
                prefix = [ "How many times does ",
                           "How many times do "]
                assert question.startswith(prefix[0]) or question.startswith(prefix[1])
                if question.startswith(prefix[0]):
                    question = question[len(prefix[0]):-1]
                else:
                    question = question[len(prefix[1]):-1]
            if type == 'count':
                caption = 'SOS' + ' ' + question + ' ' + answer + ' times ' + 'EOS'
            else:
                caption = 'SOS' + ' ' + question + ' ' + answer + ' EOS'
            # print(caption)
            captions.append(caption)
    elif type == 'action' or type == 'transition':
        assert candidates is not None
        for i, question in enumerate(questions):
            # What does
            if _replace:
                prefix = [ 'What does', 'What do']
                assert question.startswith(prefix[0]) or question.startswith(prefix[1])
                if question.startswith(prefix[0]):
                    question = question[len(prefix[0]):-1]
                else:
                    question = question[len(prefix[1]):-1]

            answer_sentence = candidates[i][int(answers[i])]

            if _replace:
                assert 'do' in question
                caption = question.replace("do", answer_sentence)
            else:
                caption = 'SOS' + ' ' + question + ' ' + answer_sentence + ' EOS'

            # print(caption)
            captions.append(caption)
    else:
        raise RuntimeError('{}'.format(type))

    return captions

def process_questions_openended(args):
    print('Loading data')
    if args.split in ["train"]:
        csv_data = pd.read_csv(args.annotation_file.format(
            args.dataset, "Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format(
            args.dataset, "Test", args.question_type), delimiter='\t')

    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])

    if args.question_type == 'frameqa':
        # video_captions = list(csv_data["description"])
        video_captions = make_video_description(questions, answers, 'frameqa')
    else:
        video_captions = make_video_description(questions, answers, 'count')

    print('number of questions: %s' % len(questions))

    # Either create the vocab or load it from disk
    if args.split in ['train']:
        print('Building vocab')
        answer_cnt = {}

        answer_token_to_idx = {'UNK': 0}
        question_answer_token_to_idx = deepcopy(base_vocab)

        if args.question_type == "frameqa":
            for i, answer in enumerate(answers):
                answer_cnt[answer] = answer_cnt.get(answer, 0) + 1
                if len(answer_cnt) >= args.answer_top:
                    break
            for token in answer_cnt:
                answer_token_to_idx[token] = len(answer_token_to_idx)
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)

            print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))
        elif args.question_type == 'count':
            answer_token_to_idx = {'UNK': 0}

        question_token_to_idx = deepcopy(base_vocab)

        for i, q in enumerate(questions):
            question = q.lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)

                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)

        for i, caption in enumerate(video_captions):
            for token in nltk.word_tokenize(caption):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)

        print('Get question_token_to_idx')
        print('question vocab size ',len(question_token_to_idx))
        print('answer vocab size ', len(answer_token_to_idx))
        print('question answer vocab size ', len(question_answer_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': question_answer_token_to_idx
        }

        print('Write into %s' % args.vocab_json.format(args.dataset, args.question_type))
        with open(args.vocab_json.format(args.dataset, args.question_type), 'w') as f:
            json.dump(vocab, f, indent=4)

        # split 10% of questions for evaluation
        split = int(0.9 * len(questions))
        train_questions = questions[:split]
        train_answers = answers[:split]
        train_video_names = video_names[:split]
        train_video_ids = video_ids[:split]
        train_captions = video_captions[:split]

        val_questions = questions[split:]
        val_answers = answers[split:]
        val_video_names = video_names[split:]
        val_video_ids = video_ids[split:]
        val_captions = video_captions[split:]


        openeded_encoding_data(args, vocab,
                               train_questions, train_video_names,
                               train_video_ids, train_answers,
                               train_captions,mode='train')

        openeded_encoding_data(args, vocab,
                               val_questions, val_video_names,
                               val_video_ids, val_answers,
                               val_captions,mode='val')

    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.question_type), 'r') as f:
            vocab = json.load(f)
        openeded_encoding_data(args, vocab, questions,
                               video_names, video_ids, answers,
                               video_captions,mode='test')

def process_questions_mulchoices(args):
    print('Loading data')
    if args.split in ["train", "val"]:
        csv_data = pd.read_csv(args.annotation_file.format(args.dataset,"Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format(args.dataset ,"Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    ans_candidates = np.asarray(
        [csv_data['a1'], csv_data['a2'], csv_data['a3'], csv_data['a4'], csv_data['a5']])
    # (5, num_ques) -> (num_ques,5)
    ans_candidates = ans_candidates.transpose()
    print("ans_candidates shape",ans_candidates.shape)
    # ans_candidates: (num_ques, 5)
    print('number of questions: %s' % len(questions))
    captions = make_video_description(questions,answers,'action', ans_candidates)

    # Either create the vocab or load it from disk
    if args.split in ['train']:
        print('Building vocab')

        answer_token_to_idx = deepcopy(base_vocab)
        question_answer_token_to_idx = deepcopy(base_vocab)

        for candidates in ans_candidates:
            for ans in candidates:
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

        # split 10% of questions for evaluation
        split = int(0.9 * len(questions))
        train_questions = questions[:split]
        train_answers = answers[:split]
        train_video_names = video_names[:split]
        train_video_ids = video_ids[:split]
        train_ans_candidates = ans_candidates[:split, :]
        train_captions = captions[:split]

        val_questions = questions[split:]
        val_answers = answers[split:]
        val_video_names = video_names[split:]
        val_video_ids = video_ids[split:]
        val_ans_candidates = ans_candidates[split:, :]
        val_captions = captions[split:]

        multichoice_encoding_data(args, vocab,
                                  train_questions, train_video_names,
                                  train_video_ids, train_answers,
                                  train_ans_candidates,train_captions,
                                  mode='train')
        multichoice_encoding_data(args, vocab,
                                  val_questions, val_video_names,
                                  val_video_ids, val_answers,
                                  val_ans_candidates, val_captions,
                                  mode='val')
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.question_type), 'r') as f:
            vocab = json.load(f)
        multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers,
                                  ans_candidates, captions, mode='test')