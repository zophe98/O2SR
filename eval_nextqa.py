
import json
import os
import os.path as osp
import pickle as pkl
import pandas as pd

map_name = {'CW': 'Why', 'CH': 'How',
            'TN': 'Bef&Aft', 'TC': 'When',
            'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other',
            'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}
CSV_DIR = "./datasets/orignal/next-qa/{}.csv"


def load_file(filename):
    """
    load obj from filename
    :param filename:
    :return:
    """
    cont = None
    if not osp.exists(filename):
        print('{} not exist'.format(filename))
        return cont
    if osp.splitext(filename)[-1] == '.csv':
        # return pd.read_csv(filename, delimiter= '\t', index_col=0)
        return pd.read_csv(filename, delimiter=',')
    with open(filename, 'r') as fp:
        if osp.splitext(filename)[1] == '.txt':
            cont = fp.readlines()
            cont = [c.rstrip('\n') for c in cont]
        elif osp.splitext(filename)[1] == '.json':
            cont = json.load(fp)
    return cont

def accuracy_metric(sample_list_file, result_list):
    """
    :param sample_list_file: 原始的csv数据文件
    :param result_list :     预测的答案 "video_qid":{"prediction":*, "answer": *,}
    :return:
    """
    sample_list = load_file(sample_list_file)
    group = {'CW':[], 'CH':[], 'TN':[], 'TC':[], 'DC':[], 'DL':[], 'DO':[]}
    # 将一个问题类型的问题的 vid_qid 放在一起
    for id, row in sample_list.iterrows():
        qns_id = str(row['video']) + '_' + str(row['qid'])
        qtype = str(row['type'])
        #(combine temporal qns of previous and next as 'TN')
        if qtype == 'TP': qtype = 'TN'
        group[qtype].append(qns_id)

    # 预测的答案和真值
    preds = result_list

    # 各个问题类型的预测正确的个数
    group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    # 各个问题类型的 问题个数
    group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    # 一个大类问题的 预测正确的个数
    overall_acc = {'C':0, 'T':0, 'D':0}
    # 一个大类问题的 问题个数
    overall_cnt = {'C':0, 'T':0, 'D':0}
    all_acc = 0
    all_cnt = 0
    for qtype, qns_ids in group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids:
            cnt += 1
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']

            if answer == pred:
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt

    # 合并到分组的字典里面
    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    for qtype in group_acc:
        print(map_name[qtype], end='\t')
    print('')
    for qtype, acc in group_acc.items():
        print('{:.2f}'.format(acc*100.0/group_cnt[qtype]), end ='\t')
    print('')
    # 打印各个类型占总的比例
    for qtype, cnt in group_cnt.items():
        print('{:.2f}%'.format(group_cnt[qtype] * 100.0 / all_cnt), end='\t')
    print('')
    print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))

def compute_nextqa_acc(vid_list,qid_list, pred_list, ans_list, mode):
    """
    先制作词典
    {
        "vid_qid":{
            "prediction": 4,
            "answer": 0
        }
    }
    """
    result_dict = {}
    for i, vid in enumerate(vid_list):
        qid = qid_list[i]
        res = {
            "prediction": pred_list[i],
            "answer": ans_list[i]
        }
        key_id = str(vid) + '_' + str(qid)
        result_dict[key_id] = res

    # 计算各个类型的准确率
    csv_file = CSV_DIR.format(mode)

    accuracy_metric(csv_file, result_dict)

if __name__ == '__main__':
    csv_file = './datasets/orignal/next-qa/val.csv'
    accuracy_metric(csv_file, None)
