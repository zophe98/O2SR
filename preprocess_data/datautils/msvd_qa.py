import json
from datautils import text_utils as  utils
import nltk
from collections import Counter

import pickle
import numpy as np
from copy import deepcopy

base_vocab = {'PAD': 0, 'UNK' : 1,
              'SOS': 2, 'EOS' : 3,
              'SEQ': 4, 'CLS' : 5,
              'QUE': 6,  'ANS' : 7}

def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    video_paths = []
    video_ids = []
    modes = ['train', 'val', 'test']
    for mode in modes:
        with open(args.annotation_file.format(mode), 'r') as anno_file:
            instances = json.load(anno_file)
        [video_ids.append(instance['video_id']) for instance in instances]
    video_ids = set(video_ids)
    with open(args.video_name_mapping, 'r') as mapping:
        mapping_pairs = mapping.read().split('\n')
    mapping_dict = {}
    for idx in range(len(mapping_pairs)):
        cur_pair = mapping_pairs[idx].split(' ')
        mapping_dict[cur_pair[1]] = cur_pair[0]
    for video_id in video_ids:
        video_paths.append((args.video_dir + 'YouTubeClips/{}.avi'.format(mapping_dict['vid' + str(video_id)]), video_id))
    return video_paths

def load_video_name(args):
    if args.dataset == 'msvd-qa':
        # {datastename}/youtube_mapping.txt
        video_name2id_path = (args.input_ann + "/youtube_mapping.txt").format(args.dataset)
        video_id2name = {}
        with open(video_name2id_path, 'r') as file:
            for line in file:
                video_name, video_id = line.split(' ')
                video_id = int(video_id[len("vid"):])
                video_id2name[video_id] = video_name
    else:
        video_id2name = {}
        for i in range(11000):
            video_id2name[i] = "video" + str(i)
    return video_id2name


def process_questions(args, padding=True):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        instances = json.load(dataset_file)

    # Either create the vocab or load it from disk
    if args.split in ['train']:
        print('Building vocab')
        answer_cnt = {}
        for instance in instances:
            answer = instance['answer']
            answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

        question_token_to_idx = deepcopy(base_vocab)
        question_answer_token_to_idx = deepcopy(base_vocab)
        # answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
        answer_token_to_idx = {'UNK': 0}

        answer_counter = Counter(answer_cnt)
        frequent_answers = answer_counter.most_common(args.answer_top)
        total_ans = sum(item[1] for item in answer_counter.items())
        total_freq_ans = sum(item[1] for item in frequent_answers)
        print("Number of unique answers:", len(answer_counter))
        print("Total number of answers:", total_ans)
        print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans * 100.0 / total_ans))

        for token, cnt in Counter(answer_cnt).most_common(args.answer_top):
            answer_token_to_idx[token] = len(answer_token_to_idx)
            if token not in question_answer_token_to_idx:
                question_answer_token_to_idx[token] = len(question_answer_token_to_idx)

        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        for i, instance in enumerate(instances):
            question = instance['question'].lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)

        print('Get question_token_to_idx')
        print(len(question_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': question_answer_token_to_idx
        }

        print('Write into %s' % args.vocab_json.format(args.dataset, args.question_type))
        with open(args.vocab_json.format(args.dataset, args.question_type), 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.question_type), 'r') as f:
            vocab = json.load(f)

    # Encode all questions
    print('Encoding data')

    video_id2names = load_video_name(args)
    questions_encoded = []
    questions_len = []
    question_ids = []

    captions_encoded = []   # question + <SEQ> + <answer>
    re_captions_encoded = []
    captions_len = []

    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []

    for idx, instance in enumerate(instances):
        question = instance['question'].lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        # question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
        question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)

        caption = 'SOS ' + instance['question'].lower()[:-1] + ' SEQ ' + instance['answer'] + ' EOS'
        caption_tokens = nltk.word_tokenize(caption)
        caption_encoded = utils.encode(caption_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        captions_encoded.append(caption_encoded)
        captions_len.append(len(caption_encoded))

        re_caption = caption[::-1]
        re_caption_tokens = nltk.word_tokenize(re_caption)
        re_caption_encoded = utils.encode(re_caption_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        re_captions_encoded.append(re_caption_encoded)

        vid = instance['video_id']
        video_ids_tbw.append(vid)
        video_names_tbw.append(video_id2names[vid])

        if instance['answer'] in vocab['answer_token_to_idx']:
            answer = vocab['answer_token_to_idx'][instance['answer']]
        elif args.split in ['train']:
            # answer = 0
            answer = vocab['answer_token_to_idx']['UNK']
        elif args.split in ['val', 'test']:
            # answer = 1
            answer = vocab['answer_token_to_idx']['UNK']
        else:
            answer = 0

        all_answers.append(answer)
    if padding:
        max_question_length = max(len(x) for x in questions_encoded)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['question_answer_token_to_idx']['PAD'])
                # qe.append(vocab['question_token_to_idx']['<PAD>'])
        # caption
        if len(captions_encoded) > 0:
            max_caption_length = max(len(x) for x in captions_encoded)
            for cpe in captions_encoded:
                while len(cpe) < max_caption_length:
                    cpe.append(vocab['question_answer_token_to_idx']['PAD'])
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

    print("max question length", max_question_length)
    print("max caption length", max_caption_length)
    print("padded question encoded", questions_encoded.shape)
    print("padded caption encoded", captions_encoded.shape)

    glove_matrix = None
    if args.split == 'train':
        # token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        token_itow = {i: w for w, i in vocab['question_answer_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print("glove_matrix.shape",glove_matrix.shape)

    print('Writing', args.output_pt.format(args.dataset, args.question_type, args.split))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'captions': captions_encoded,
        're_captions': re_captions_encoded,
        'captions_len': captions_len,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'answers': all_answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.dataset, args.question_type , args.split), 'wb') as f:
        pickle.dump(obj, f)

