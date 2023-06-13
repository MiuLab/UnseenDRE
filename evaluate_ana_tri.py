import json
import numpy as np
import argparse
from sklearn.metrics import f1_score
from difflib import SequenceMatcher
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from difflib import SequenceMatcher
import csv


def similar(a, b):
    if b=="[CLS]" and a=="[CLS]":
        return 0
    if b=="[CLS]" and a!="[CLS]":
        return -1
    if b!="[CLS]" and a=="[CLS]":
        return 0
    return SequenceMatcher(None, a, b).ratio()


set_a = [1,7,21,24,28,29,30,3,4,9,10,11,12,5,13,32,19,20,22,36]
set_b = [6,34,18,23]
# set_c = [2,8,14,15,16,17,25,26,27,31,33,35]
# set_c = [1,7,13,14,15,16,24,25,26,30,32,34]
set_c = [15,33,35,8,14,16,17,2,25,26,27,31]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def getresult(fn):
    result = []
    with open(fn, "r") as f:
        l = f.readline()
        while l:
            l = l.strip().split()
            # print(type(l))
            # print(len(l)) 54864
            # exit()
            
            for i in range(len(l)):
                l[i] = float(l[i])
            result += [l]
            l = f.readline()
    # print(type(result))
    # print(len(result))
    # print(len(result[0]))
    # exit()
    result = np.asarray(result)
    return list(1 / (1 + np.exp(-result)))

def evaluate(devp, data, tri_pos_pred, tri_pos_gt):
    index = 0
    correct_sys, all_sys = 0, 0
    correct_gt = 0
    rel_output = []
    tri_output = []
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            if data[i][1][j]["rid"][0]==36:
                continue

            if (data[i][1][j]["rid"][0]+1) in set_a:
                index+=1
                continue
            if (data[i][1][j]["rid"][0]+1) in set_b:
                index+=1
                continue
            # if (data[i][1][j]["rid"][0]+1) in set_c:
            #     index+=1
            #     continue
            # print(data[i][1][j]["rid"],devp[index])
            for id in data[i][1][j]["rid"]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[index]:
                        correct_sys += 1
            for id in devp[index]:
                if id != 36:
                    all_sys += 1
            
            # print(index)
            # print(devp[index]) # predict candidate
            # print(data[i][1][j]["rid"][0]) # ground truth
            # rel_output.append([devp[index][0], data[i][1][j]["rid"][0]])
            # print(tri_pos_pred[index])
            # print(tri_pos_gt[index])
            # print(similar(tri_pos_pred[index], tri_pos_gt[index]))
            # print()

            # print(devp[index])
            # print(len(devp[index]))
            if len(devp[index])==1:
                tri_output.append([devp[index][0], data[i][1][j]["rid"][0], similar(tri_pos_pred[index], tri_pos_gt[index]),tri_pos_pred[index],tri_pos_gt[index]])
            else:
                tri_output.append([devp[index][0], devp[index][1], data[i][1][j]["rid"][0], similar(tri_pos_pred[index], tri_pos_gt[index]),tri_pos_pred[index],tri_pos_gt[index]])
            # exit()

            index += 1
    if args.rank_num == 1:
        with open('tri_output_C_first_binary_v1.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(["predict","ground truth","similar score","predict_tri","gt_tri"])
            write.writerows(tri_output)
    elif args.rank_num == 2:
        with open('tri_output_C_first_binary_v1.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(["predict_A", "predict_B","ground truth","similar score","predict_tri","gt_tri"])
            write.writerows(tri_output)
    # exit()
    # with open('rel_output.csv', 'w') as f:
    #     write = csv.writer(f)
    #     write.writerow(["predict", "ground truth"])
    #     write.writerows(rel_output)
    
    # with open('tri_output_C_first_binary_v1.csv', 'w') as f:
    #     write = csv.writer(f)
    #     write.writerow(["predict","ground truth","similar score","predict_tri","gt_tri"])
    #     write.writerows(tri_output)

    precision = correct_sys/all_sys if all_sys != 0 else 1
    recall = correct_sys/correct_gt if correct_gt != 0 else 0
    f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    return precision, recall, f_1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument("--f1dev",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="Dev logits (f1).")
    
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--rank_num", type=int)
    parser.add_argument("--dev_or_test", type=int)
    args = parser.parse_args()
    
    # f1dev = args.f1dev
    # f1test = args.f1test
    # f1cdev = args.f1cdev
    # f1ctest = args.f1ctest
    if args.dev_or_test == 0:
        with open("ori_data/dev.json", "r", encoding='utf8') as f:
            datadev = json.load(f)
    else:
        with open("ori_data/test.json", "r", encoding='utf8') as f:
            datadev = json.load(f)

    
    ans_ref = []
    for i in range(len(datadev)):
        for j in range(len(datadev[i][1])):
            # ans_ref.append(datadev[i][1][j]["rid"])
            for k in range(len(datadev[i][1][j]["rid"])):
                datadev[i][1][j]["rid"][k] -= 1 # relation value minus 1 (eg: 29->28)
            # if datadev[i][1][j]["rid"][0]!=36 and datadev[i][1][j]["rid"][0]+1 not in set_a and datadev[i][1][j]["rid"][0]+1 not in set_b:
            if datadev[i][1][j]["rid"][0]!=36:
                # print(datadev[i][1][j]["rid"])
                ans_ref.append(datadev[i][1][j]["rid"])
            # if datadev[i][1][j]["rid"][0] in set_c:
            #     ans_ref.append(datadev[i][1][j]["rid"])
    # exit()

    tri_gt = []
    # with open("tri_analy_last/_trig_gt_last_cls", "r") as f:
    with open("train_on_A_true_t_idx_2_23/_trig_gt_binary_cls", "r") as f:
        for line in f:
            tri_gt.append(str(line.strip()))
    # print(tri_gt[0])
    tri_pred = []
    # with open("tri_analy_last/_trig_pred_last_cls", "r") as f:
    with open("train_on_A_true_t_idx_2_23/_trig_pred_binary_cls", "r") as f:
        for line in f:
            tri_pred.append(str(line.strip()))
    # print(tri_pred[0])

    # tri_pos_pred=[]
    tri_pos_gt = []
    for h in range(len(tri_gt)):
        if h%36==0:
            tri_pos_gt.append(tri_gt[h])
    
    

    # dev = []
    # with open(args.input_file, "r") as f:
    #     for line in f:  
    #         dev.append(float(line.strip()))
    dev = getresult(args.input_file)
    dev = list(dev[0])

    
    all_cnt=0
    devp = []
    devplogits = []
    tmp = []
    logits = []
    cnt=0
    trigs = []
    for logit in dev:
        # print(all_cnt)
        logits.append(logit)
        if cnt>34:
            # print(logits)
            logit_sort = logits.copy()
            logit_sort.sort(reverse=True)
            num=1
            for idx in range(len(logit_sort)):
                if idx==0:
                    continue
                if logit_sort[idx]*2<logit_sort[idx-1]:
                    break
                num+=1
            num=args.rank_num
            for i in logits:
                if i in logit_sort[:num]:
                # if i > -1:
                    tmp.append(1)
                else:
                    tmp.append(0)
            # print(logits)
            # print(max(logits))
            # print(tmp)

            devp.append(tmp)
            devplogits.append(logits)
            tmp = []
            logits = []
            cnt=-1
        cnt+=1
        # all_cnt+=1
    # exit()
    # print(len(devp)) # 1470
    # exit()
    # random sampling [Important]
    # map_list = []
    # for i in range(len(datadev)):
    #     for j in range(len(datadev[i][1])):
    #         if datadev[i][1][j]["rid"][0]==36:
    #             continue
    #         map_list.append(datadev[i][1][j]["rid"])
    
    if args.dev_or_test == 0:
        with open("trend/dev.json", "r", encoding='utf8') as f:
            shu_data = json.load(f)  
    else:
        with open("trend/test.json", "r", encoding='utf8') as f:
            shu_data = json.load(f)  

    # with open("trend/test.json", "r", encoding='utf8') as f:
    #     shu_data = json.load(f)
    
    shu_map=[]
    shu_tmp=[]
    for i in range(len(shu_data)):
        for j in range(len(shu_data[i][1])):
            shu_tmp.append(shu_data[i][1][j]["flag"])
            if len(shu_tmp)==36:
                shu_map.append(shu_tmp.copy())
                shu_tmp=[]
    # print(len(shu_map))
    # print(len(devp))





    # random sampling [Important]
    devps = []
    cnt=0
    for i in range(len(devp)):
        tmp = []
        for j in range(len(devp[i])):
            if devp[i][j]==1:
                tmp.append(shu_map[i][j]-1)
        cnt+=1
        devps.append(tmp)





    tri_pos_pred=[]
    hh_cnt=0
    for i in range(len(devp)):
        tmp = []
        for j in range(len(devp[i])):
            if devp[i][j]==1:
                tri_pos_pred.append(tri_pred[hh_cnt])
            hh_cnt+=1
    




    # print(devp[295])
    # print(tri_pos_pred[295])
    # print(tri_pos_gt[295])
    # exit()
    # print(devp[1])
    # print(tri_pos_pred[1])
    # print(tri_pos_gt[1])
    # print(devp[2])
    # print(tri_pos_pred[2])
    # print(tri_pos_gt[2])
    # print(devp[3])
    # print(tri_pos_pred[3])
    # print(tri_pos_gt[3])
    # exit()
    # print()
    # exit()
    # 37 in order [Important]
    # devps = []
    # for i in range(len(devp)):
    #     tmp = []
    #     for j in range(len(devp[i])):
    #         if devp[i][j]==1:
    #             tmp.append(j)
    #     devps.append(tmp)
    

    # for j in range(len(devps)):
    #     print(ans_ref[j],end=" ")
    #     print(devps[j])
    
    # print("check ans len")
    # print(len(ans_ref))

    # print(len(devps))
    # exit()
    precision, recall, f_1 = evaluate(devps, datadev, tri_pos_pred, tri_pos_gt)
    # precision, recall, f_1 = evaluate(devps, datadev)
    # print("dev (P R F1)", precision, recall, f_1)
    # print("dev (P R F1)", f_1)
    print(f_1)

    