__author__ = 'jake221'

'''
 * Weighted matrix factorization method proposed by Hu et al. and Pan et al..
 *
 * We use the fast learning method proposed by Hu et al. (alternating least squares),
 * and we use a global weight to penalize observed/unobserved values.
 *
 * Literature:
 *
 *     Y. Hu, Y. Koren, C. Volinsky: Collaborative filtering for implicit feedback datasets.
 *     ICDM 2008.
 *     http://research.yahoo.net/files/HuKorenVolinsky-ICDM08.pdf
 *
 *     R. Pan, Y. Zhou, B. Cao, N. N. Liu, R. M. Lukose, M. Scholz, Q. Yang:
 *     One-class collaborative filtering,
 *     ICDM 2008.
 *     http://www.hpl.hp.com/techreports/2008/HPL-2008-48R1.pdf
 *
 * This recommendation framework is the Python version of MyMediaLite and only integrates the WRMF model.
'''

import LoadData
from IMF import *
from Evaluate import *

class ItemRecommendation():
    '''
    This class includes three components: load train data and test data; train the recommendation model; evaluate the model by eight different metrics
    '''
    def __init__(self,TRAIN_FILE,TEST_FILE,NUM_USERS,NUM_ITEMS,TRAIN_SIZE,TEST_SIZE):
        '''
        :param TRAIN_FILE:
        :param TEST_FILE:
        :param NUM_USERS:
        :param NUM_ITEMS:
        :param TRAIN_SIZE:
        :param TEST_SIZE:
        :return:
        '''
        self.TRAIN_FILE = TRAIN_FILE
        self.TEST_FILE = TEST_FILE
        self.NUM_USERS = NUM_USERS
        self.NUM_ITEMS = NUM_ITEMS
        self.TRAIN_SIZE = TRAIN_SIZE
        self.TEST_SIZE = TEST_SIZE

    def load_data(self,cPos):
        # load train matrix and test matrix
        self.train_sparsematrix, self.train_matrix = LoadData.load_train_matrix(self.TRAIN_FILE,self.NUM_USERS,self.NUM_ITEMS,cPos)
        self.train_tuple = LoadData.load_tuple(self.TRAIN_FILE,self.TRAIN_SIZE)
        self.test_matrix = LoadData.load_test_matrix(self.TEST_FILE,self.NUM_USERS,self.NUM_ITEMS)
        self.test_tuple = LoadData.load_tuple(self.TEST_FILE,self.TEST_SIZE)

    def model_train(self,NUM_FACTORS,NUM_ITERATIONS,REG_PARAMETERS):
        # train the model
        imf = ImplicitMF(self.train_sparsematrix,NUM_FACTORS,NUM_ITERATIONS,REG_PARAMETERS)
        self.userFactors,self.itemFactors = imf.train_model()

    def model_evaluate(self):
        # evaluate the model

        ## preprocess: find test_users and candidate_items
        test_users = np.unique(self.test_tuple[:,0])                     # all users in the test set
        allItems_train = np.unique(self.train_tuple[:,1])                # all items in the train set
        allItems_test = np.unique(self.test_tuple[:,1])                  # all items in the test set
        candidate_items = np.union1d(allItems_train,allItems_test)  # all items in the train and test set

        recEval = Evaluate(self.userFactors,self.itemFactors,self.train_matrix,self.test_matrix,test_users,candidate_items)
        ret = recEval.CalcMetrics()
        print 'AUC =',ret[0],'Prec@5 =',ret[1],'Prec@10 =',ret[2], 'MAP =', ret[3], 'Rec@5 =', ret[4], 'Rec@10 =', ret[5], 'NDCG =', ret[6], 'MRR =', ret[7]
        return ret

if __name__ == '__main__':
    '''
    Parameter setting
    '''
    TRAIN_FILE = './data/ml100k/train_data.txt'
    TEST_FILE = './data/ml100k/test_data.txt'
    NUM_USERS = 943
    NUM_ITEMS = 1682
    TRAIN_SIZE = 90570
    TEST_SIZE = 9430
    cPos = 2

    NUM_FACTORS = 10
    NUM_ITERATIONS = 20
    REG_PARAMETERS = 0.01

    # initialize the ItemRecommendation class
    item_rec = ItemRecommendation(TRAIN_FILE,TEST_FILE,NUM_USERS,NUM_ITEMS,TRAIN_SIZE,TEST_SIZE)
    # load data
    item_rec.load_data(cPos)
    item_rec.model_train(NUM_FACTORS,NUM_ITERATIONS,REG_PARAMETERS)
    metrics = item_rec.model_evaluate()