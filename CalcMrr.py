__author__ = 'jake221'

def MRR(rec_list,correct_items,ignore_items):
    pos = 0
    for i in range(rec_list.size):
        if rec_list[i] in ignore_items:
            continue

        pos = pos+1

        if rec_list[pos] in correct_items:
            mrr = 1 / (pos * 1.0)
            break
    return mrr
