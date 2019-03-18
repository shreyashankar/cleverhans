from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import autograph
from scipy.sparse.linalg import eigs, LinearOperator

from cleverhans.experimental.certification import utils

TOL = 1E-2
TEST_DIM = 50
np.random.seed(0)

def advanced_lanczos_test(b):
  # b = b.astype(np.float64)
  k_vals = [100, 200, 300, 400, 500]
  filenames = ['diverging.npy', 'regular.npy']
  min_eigs = [-3.5875054e-05, 0.00739005]

  for idx, filename in enumerate(filenames):
    # if idx == 0:
    #   continue
    filename = './matrices/' + filename
    matrix = np.load(filename).astype(np.float32)
    n = matrix.shape[0]

    def _vector_prod_fn(x):
      return tf.matmul(matrix, x)
    def _np_vector_prod_fn(x):
      return np.matmul(matrix, x)

    # Create lanczos graph nodes
    min_eigen_fn = autograph.to_graph(utils.tf_lanczos_eigval)
    # min_eigen_fn = utils.lzs_two

    # Compare against scipy
    # linear_operator = LinearOperator((n, n), matvec=_np_vector_prod_fn)
    # min_eig_scipy, _ = eigs(linear_operator, k=1, which='SR', tol=TOL)
    min_eig_scipy = min_eigs[idx]
    print("Min eig scipy: " + str(min_eig_scipy))

    print('k\t\tlzs time\t\teigh time\t\tmin eig')

    for k in k_vals:
      # Use lanczos method
      with tf.Session() as sess:
        start = time.time()
        alpha_hat, beta_hat, Q_hat = min_eigen_fn(_vector_prod_fn, n, k, b)
        # Finalize graph to make sure no new nodes are added
        tf.get_default_graph().finalize()
        curr_alpha_hat, curr_beta_hat, Q_hat = sess.run([alpha_hat, beta_hat, Q_hat])
        # print(curr_alpha_hat)

        lzs_time = time.time() - start
        start = time.time()
        min_eig_lzs, _, _, _ = utils.eigen_tridiagonal(curr_alpha_hat, curr_beta_hat, Q_hat, maximum=False)
        eigh_time = time.time() - start
        # print(min_eig_lzs)
        print('%d\t\t%g\t\t%g\t\t%-10.5g' %(
            k, lzs_time, eigh_time, min_eig_lzs))
        # np.testing.assert_almost_equal(min_eig_lzs, min_eig_scipy, decimal=2)
      tf.reset_default_graph()

def advanced_lanczos_test_np(b):
  # b = b.astype(np.float64)
  k_vals = [100, 200, 300, 400, 500]
  filenames = ['diverging.npy', 'regular.npy']
  min_eigs = [-3.5875054e-05, 0.00739005]

  for idx, filename in enumerate(filenames):
    # if idx == 1:
    #   continue
    filename = './matrices/' + filename
    matrix = np.load(filename).astype(np.float32)
    n = matrix.shape[0]

    def _np_vector_prod_fn(x):
      return np.matmul(matrix, x)

    # Compare against scipy
    # linear_operator = LinearOperator((n, n), matvec=_np_vector_prod_fn)
    # min_eig_scipy, _ = eigs(linear_operator, k=1, which='SR', tol=TOL)
    min_eig_scipy = min_eigs[idx]
    print("Min eig scipy: " + str(min_eig_scipy))

    print('k\t\tlzs numpy time\t\teigh time\t\teig')

    for k in k_vals:
      # Use lanczos method
      start = time.time()
      curr_alpha_hat, curr_beta_hat, Q_hat = utils.lanczos_eigvec(_np_vector_prod_fn, b, k)
      # print(curr_alpha_hat)

      # Finalize graph to make sure no new nodes are added
      lzs_time = time.time() - start
      start = time.time()
      min_eig_lzs, _, _, _ = utils.eigen_tridiagonal(curr_alpha_hat, curr_beta_hat[1:], Q_hat, maximum=False)
      eigh_time = time.time() - start
      print('%d\t\t%g\t\t%g\t\t%-10.5g' %(
          k, lzs_time, eigh_time, min_eig_lzs))
      # np.testing.assert_almost_equal(min_eig_lzs, min_eig_scipy, decimal=2)

b = np.random.randn(2745, 1).astype(np.float32)
# advanced_lanczos_test_np(b)
advanced_lanczos_test(b)