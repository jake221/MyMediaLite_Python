__author__ = 'jake221'

import numpy as np
import CalcPreRec
import CalcAuc
import CalcNdcg
import CalcMrr
import time

class Evaluate():
    '''
    evaluate the effectiveness of the recommendation model using eight different information retrieval metrics
    '''
    def __init__(self,user_vecs,item_vecs,train_matrix,test_matrix,test_users,candidate_items):
        self.user_vecs = user_vecs
        self.item_vecs = item_vecs
        self.train_matrix = train_matrix
        self.test_matrix = test_matrix
        self.test_users = test_users
        self.candidate_items = candidate_items

    def CalcMetrics(self):
        '''
        :return: eight metrics
        '''
        num_users = 0
        ret = np.zeros((8,1))
        user_num = len(self.test_users)

        precision = np.zeros((user_num,2))
        recall = np.zeros((user_num,2))
        map = np.zeros((user_num,1))
        auc = np.zeros((user_num,1))
        ndcg = np.zeros((user_num,1))
        mrr = np.zeros((user_num,1))
        AtN = [5,10]

        t0 = time.time()
        print 'Start evaluating...'

        for i in xrange(user_num):
            user_id = self.test_users[i]
            # print 'user_id',user_id

            # find items that user has rated in the test set
            test_nonzero_idx = self.test_matrix[user_id,:].nonzero()
            test_items_idx = test_nonzero_idx[0]
            correct_items = np.intersect1d(test_items_idx,self.candidate_items)

            # find items that user has rated in the train set
            train_nonzero_idx = self.train_matrix[user_id,:].nonzero()
            # print 'train_nonzero_idx',train_nonzero_idx
            train_items_idx = train_nonzero_idx[0]
            # print 'self.candidate_items',self.candidate_items
            candidate_items_in_train = np.intersect1d(train_items_idx,self.candidate_items)

            num_eval_items = self.candidate_items.size - candidate_items_in_train.size

            # if user has not rated any items in test set or all items in test set are relevant then continue
            if correct_items.size == 0 | num_eval_items - correct_items.size == 0:
                continue

            # generate a item recommendation list for user_id
            recommendation_list = self.GenerateLists(self.user_vecs,self.item_vecs,user_id,self.candidate_items)

            ignore_items = train_items_idx

            precision[i,:],recall[i,:],map[i] = CalcPreRec.PrecisionAndRecall(recommendation_list, correct_items, ignore_items, AtN)
            auc[i] = CalcAuc.AUC(recommendation_list, correct_items, ignore_items)
            ndcg[i] = CalcNdcg.NDCG(recommendation_list, correct_items, ignore_items)
            mrr[i] = CalcMrr.MRR(recommendation_list, correct_items, ignore_items)
            num_users = num_users + 1

        t1 = time.time()
        print 'Evaluation finished in %f seconds' %  (t1 - t0)

        ret[0] = sum(auc) / (num_users * 1.0)
        ret[1] = sum(precision[:,0]) / (num_users * 1.0)
        ret[2] = sum(precision[:,1]) / (num_users * 1.0)
        ret[3] = sum(map) / (num_users * 1.0)
        ret[4] = sum(recall[:,0]) / (num_users * 1.0)
        ret[5] = sum(recall[:,1]) / (num_users * 1.0)
        ret[6] = sum(ndcg) / (num_users * 1.0)
        ret[7] = sum(mrr) / (num_users * 1.0)

        return ret

    def GenerateLists(self,user_vecs,item_vecs,user_id,candidate_items):
        predict_list = np.zeros((candidate_items.size,1))
        for i in range(candidate_items.size):
            predict_list[i] = np.dot(user_vecs[user_id,:], item_vecs[candidate_items[i],:])
        list_asc = np.argsort(predict_list,axis=0)
        sorted_list = list_asc[::-1]

        return sorted_list