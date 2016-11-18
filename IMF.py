# coding: UTF-8
__author__ = 'jake221'

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time

class ImplicitMF():

    def __init__(self, counts, num_factors, num_iterations,reg_param):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param

    def train_model(self):
        #创建user_vectors和item_vectors，他们的元素~N(0,1)的正态分布
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))
        '''要生成很大的数字序列的时候，用xrange会比range性能优很多，
        因为不需要一上来就开辟一块很大的内存空间，这两个基本上都是在循环的时候用'''
        for i in xrange(self.num_iterations):
            t0 = time.time()
            print 'Solving for user vectors...'
            self.user_vectors = self.iteration(True, sparse.csr_matrix(self.item_vectors))
            print 'Solving for item vectors...'
            self.item_vectors = self.iteration(False, sparse.csr_matrix(self.user_vectors))
            t1 = time.time()
            print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)
        return self.user_vectors,self.item_vectors

    def iteration(self, user, fixed_vecs):
        #相当于C的三木运算符，if user=True num_solve = num_users,反之为num_items
        num_solve = self.num_users if user else self.num_items  # 用户（或者产品）个数：待求解的向量的规模
        num_fixed = fixed_vecs.shape[0]                         # 产品（或者用户）个数：固定的向量的规模
        YTY = fixed_vecs.T.dot(fixed_vecs)      # Y^T * Y
        eye = sparse.eye(num_fixed)             # 用于计算后面的Y^T * C^u * p(u)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)  # lambda * I
        solve_vecs = np.zeros((num_solve, self.num_factors))        # 结果存储器
        t = time.time()
        for i in xrange(num_solve):
            if user:
                counts_i = self.counts[i].toarray()     # Return a dense ndarray representation of this matrix：将第i个用户的评分向量由稀疏矩阵转变为向量
            else:
                #如果要求item_vec,counts_i为counts中的第i列的转置
                counts_i = self.counts[:, i].T.toarray()
            ''' 原论文中c_ui=1+alpha*r_ui,但是在计算Y’CuY时为了降低时间复杂度,利用了
                Y'CuY=Y'Y+Y'(Cu-I)Y,由于Cu是对角矩阵,其元素为c_ui，即1+alpha*r_ui。
                所以Cu-I也就是对角元素为alpha*r_ui的对角矩阵'''
            CuI = sparse.diags(counts_i, [0])           # 将r_ui放到对角线上，若r_ui != 0,则r_ii != 0
            pu = counts_i.copy()                        # 复制pu
            #np.where(pu != 0)返回pu中元素不为0的索引，然后将这些元素赋值为1,不知道这里为什么要赋值为1?:将pu中的非零项置为1；因为pu是由counts_i得到的而counts_i本质是alpha * r_ui
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)     # spsolve(A,B)求解Ax = B
            solve_vecs[i] = xu
            if i % 1000 == 0:
                print 'Solved %i vecs in %d seconds' % (i, time.time() - t)
                t = time.time()
        return solve_vecs