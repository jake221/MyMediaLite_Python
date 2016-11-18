__author__ = 'jake221'

import time
import numpy as np
import scipy.sparse as sparse

def load_train_matrix(file_name,num_users,num_items,cPos):
    '''
    :return: user-item rating martix (sparse)
    '''
    t0 = time.time()
    counts = np.zeros((num_users,num_items))
    total = 0.0     # to store the number of nonzero entry
    num_zeros = num_users * num_items
    for i, line in enumerate(open(file_name,'r')):
        user, item = line.strip().split('\t')
        user = int(user)
        item = int(item)
        count = 1.0
        if user > num_users:
            continue
        if item > num_items:
            continue
        if count != 0:
            counts[user-1,item-1] = count
            total += count
            num_zeros -= 1
        if i % 100000 == 0:
            print "loaded %i counts..." % i
    counts *=  cPos
    sparse_counts = sparse.csr_matrix(counts)       # transformed the matrix into a sparse matrix
    t1 = time.time()
    print 'Finished loading train matrix in %f seconds' % (t1 - t0)
    return sparse_counts, counts

def load_test_matrix(file_name,num_users,num_items):
    '''
    :return: user-item rating martix (sparse)
    '''
    t0 = time.time()
    counts = np.zeros((num_users,num_items))
    total = 0.0     # to store the number of nonzero entry
    num_zeros = num_users * num_items
    for i, line in enumerate(open(file_name,'r')):
        user, item = line.strip().split('\t')
        user = int(user)
        item = int(item)
        count = 1.0
        if user > num_users:
            continue
        if item > num_items:
            continue
        if count != 0:
            counts[user-1,item-1] = count
            total += count
            num_zeros -= 1
        if i % 100000 == 0:
            print "loaded %i counts..." % i
    t1 = time.time()
    print 'Finished loading test matrix in %f seconds' % (t1 - t0)
    return counts

def load_tuple(file_name,data_size):
    '''
    :return:
    '''
    rating_tuple = np.zeros((data_size,2))
    for i,line in enumerate(open(file_name,'r')):
        user, item = line.strip().split('\t')
        rating_tuple[i-1,0] = int(user)-1
        rating_tuple[i-1,1] = int(item)-1
    return rating_tuple