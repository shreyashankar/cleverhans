from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from tensorflow.contrib import autograph
from scipy.sparse.linalg import eigs, LinearOperator

from cleverhans.experimental.certification import utils

TOL = 1E-5

def test_lanczos():
    # indices = [100, 200, 250, 300, 400, 500]
    indices = [50, 150, 350, 340, 500, 550]
    k_vals=[10, 20, 30, 40, 50, 70, 100, 200, 300, 400]

    def _vector_prod_fn(x):
      return tf.matmul(matrix, x)
    def _np_vector_prod_fn(x):
      return np.matmul(matrix, x)

    for i in indices:
      filename = './matrices/iter_' + str(i) + '.npy'#+ '_diverging.npy'
      print(filename)
      matrix = np.load(filename).astype(np.float32)
      n = matrix.shape[0]

      # Create lanczos graph nodes
      min_eigen_fn = autograph.to_graph(utils.lanczos_decomp)

      # Compare against scipy
      linear_operator = LinearOperator((n, n), matvec=_np_vector_prod_fn)
      min_eig_scipy, _ = eigs(matrix, k=1, which='SR', tol=TOL)
      print("Min eig scipy: " + str(min_eig_scipy))

      print('k\t\tlzs time\t\teigh time\t\tl_min err')

      for k in k_vals:
        # Use lanczos method
        with tf.Session() as sess:
          alpha_hat, beta_hat, Q_hat = min_eigen_fn(_vector_prod_fn, 0, n, k)
          # Finalize graph to make sure no new nodes are added
          tf.get_default_graph().finalize()
          curr_alpha_hat, curr_beta_hat, curr_Q_hat = sess.run([alpha_hat, beta_hat, Q_hat])
          start = time.time()
          curr_alpha_hat, curr_beta_hat, curr_Q_hat = sess.run([alpha_hat, beta_hat, Q_hat])
          lzs_time = time.time() - start
          start = time.time()
          min_eig_lzs, max_vec, _, _ = utils.eigen_tridiagonal(curr_alpha_hat, curr_beta_hat, maximum=False)
          eigh_time = time.time() - start
          print('%d\t\t%g\t\t%g\t\t%-10.5g\t\t%.5g' %(
                k, lzs_time, eigh_time, np.abs(min_eig_lzs-min_eig_scipy)/np.abs(min_eig_scipy),
                np.abs(min_eig_lzs-min_eig_scipy)))
        tf.reset_default_graph()

test_lanczos()