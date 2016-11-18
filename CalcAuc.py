__author__ = 'jake221'

def AUC(rec_list,correct_items,ignore_items):
    num_eval_items = rec_list.size - ignore_items.size
    num_correct_items = correct_items.size
    num_eval_pairs = (num_eval_items - num_correct_items) * num_correct_items

    if (num_eval_pairs == 0):
      auc = 0.5

    num_correct_pairs = 0
    hit_count = 0
    for i in range(rec_list.size):
        if rec_list[i] in ignore_items:
            continue

        if rec_list[i] not in correct_items:
            num_correct_pairs = num_correct_pairs + hit_count
        else:
            hit_count = hit_count + 1
    auc = num_correct_pairs / (num_eval_pairs * 1.0)
    return auc