__author__ = 'jake221'

import numpy as np

def PrecisionAndRecall(rec_list,correct_items,ignore_items,AtN):
    precision = np.zeros((1,len(AtN)))
    recall = np.zeros((1,len(AtN)))
    for i in range(len(AtN)):

        precision[0,i] = hitAt(rec_list,correct_items,ignore_items,AtN[i]) / (AtN[i] * 1.0)
        recall[0,i] = hitAt(rec_list,correct_items,ignore_items,AtN[i]) / (correct_items.size * 1.0)
    Map = AP(rec_list,correct_items,ignore_items)

    return precision, recall, Map

def AP(rec_list,correct_items,ignore_items):
    # compute the average precision (AP) of a list of ranked items
    hit_count = 0
    avg_prec_sum = 0
    left_out = 0
    for i in range(rec_list.size):
        item_id = rec_list[i]
        if item_id in ignore_items:
            left_out = left_out + 1
            continue

        if item_id not in correct_items:
            continue

        hit_count = hit_count + 1

        avg_prec_sum = avg_prec_sum + (hit_count / ((i + 1 - left_out) * 1.0))

    if hit_count != 0:
        map = avg_prec_sum / (hit_count * 1.0)
    else:
        map = 0
    return map

def hitAt(rec_list,correct_items,ignore_items,n):
    hit_count = 0
    left_out = 0

    for i in range(rec_list.size):
        # print 'rec_list', rec_list.shape, i
        item_id = rec_list[i]
        if item_id in ignore_items:
            left_out = left_out + 1
            continue

        if item_id not in correct_items:
            continue

        if i < (n + left_out):
            hit_count = hit_count + 1
        else:
            break
    return hit_count