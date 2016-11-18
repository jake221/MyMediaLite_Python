__author__ = 'jake221'

import math

def NDCG(rec_list,correct_items,ignore_items):
    dcg = 0
    idcg = computeIDCG(correct_items.size)
    left_out = 0

    for i in range(rec_list.size):
        item_id = rec_list[i]
        if item_id in ignore_items:
            left_out = left_out + 1
            continue

        if item_id not in correct_items:
            continue

        rank = i + 1 - left_out
        dcg = dcg + math.log(2) / (math.log(rank+1))
    ndcg = dcg / idcg
    return ndcg

def computeIDCG(n):
    idcg = 0
    for i in range(n):
        idcg = idcg + math.log(2) / (math.log(i+2))
    return idcg























