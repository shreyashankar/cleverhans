"""Tests for cleverhans.experimental.certification.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import autograph
from scipy.sparse.linalg import eigs, LinearOperator

from cleverhans.experimental.certification import utils

TOL = 1E-5


class UtilsTest(tf.test.TestCase):

  def test_minimum_eigen_vector(self):
    matrix = np.array([[1.0, 2.0], [2.0, 5.0]], dtype=np.float32)
    initial_vec = np.array([[1.0], [-1.0]], dtype=np.float32)

    def _vector_prod_fn(x):
      return tf.matmul(matrix, x)

    min_eigen_fn = autograph.to_graph(utils.minimum_eigen_vector)
    x = tf.placeholder(tf.float32, shape=(2, 1))
    min_eig_vec = min_eigen_fn(x, 10, 0.1, _vector_prod_fn)
    with self.test_session() as sess:
      v = sess.run(min_eig_vec, feed_dict={x: initial_vec})
      if v.flatten()[0] < 0:
        v = -v
    np.testing.assert_almost_equal(v, [[0.9239], [-0.3827]], decimal=4)

  def test_lanczos(self):
    indices = [100, 200, 250, 300, 400, 500]
    k_vals=[10, 20, 30, 40, 50, 70, 100, 200, 300, 400]

    def _vector_prod_fn(x):
      return tf.matmul(matrix, x)
    def _np_vector_prod_fn(x):
      return np.matmul(matrix, x)

    with tf.Session() as sess:
      for i in indices:
        filename = './matrices/iter_' + str(i) + '_diverging.npy'
        print(filename)
        matrix = np.load(filename).astype(np.float32)
        n = matrix.shape[0]

        # Create lanczos graph nodes
        min_eigen_fn = autograph.to_graph(utils.lzs_two)

        # Compare against scipy
        linear_operator = LinearOperator((n, n), matvec=_np_vector_prod_fn)
        min_eig_scipy, _ = eigs(matrix, k=1, which='SR', tol=TOL)
        print("Min eig scipy: " + str(min_eig_scipy))

        print('k\t\tl_min err')

        for k in k_vals:
          # Use lanczos method
          alpha_hat, beta_hat, Q_hat = min_eigen_fn(_vector_prod_fn, 0, n, k)
          curr_alpha_hat, curr_beta_hat, curr_Q_hat = sess.run([alpha_hat, beta_hat, Q_hat])
          min_eig_lzs, max_vec, _, _ = utils.eigen_tridiagonal(curr_alpha_hat, curr_beta_hat, maximum=False)
          print('%d\t\t%-10.5g\t\t%.5g' %(
                k, np.abs(min_eig_lzs-min_eig_scipy)/np.abs(min_eig_scipy),
                np.abs(min_eig_lzs-min_eig_scipy)))
  
  # def test_lanczos(self):
  #   indices = [100, 200, 250, 300, 400, 500]
  #   k_vals=[10, 20, 30, 40, 50, 70, 100, 200, 300, 400]

  #   with tf.Session() as sess:
  #     for i in indices:
  #       filename = './matrices/iter_' + str(i) + '_diverging.npy'
  #       print(filename)
  #       matrix = np.load(filename).astype(np.float32)
  #       n = matrix.shape[0]

  #       def _vector_prod_fn(x):
  #         return tf.matmul(matrix, x)
  #       def _np_vector_prod_fn(x):
  #         return np.matmul(matrix, x)

  #       # Create lanczos graph nodes
  #       min_eigen_fn = autograph.to_graph(utils.lanczos_decomp)
  #       alpha, beta, _ = min_eigen_fn(_vector_prod_fn, 0, n, 5)
  #       max_eig_ph = tf.placeholder(tf.float32, shape=[])

  #       # Compare against scipy
  #       linear_operator = LinearOperator((n, n), matvec=_np_vector_prod_fn)
  #       min_eig_scipy, _ = eigs(matrix, k=1, which='SR', tol=TOL)
  #       print("Min eig scipy: " + str(min_eig_scipy))

  #       print('k\t\tl_min err')

  #       for k in k_vals:
  #         # Use lanczos method
  #         alpha_hat, beta_hat, Q_hat = min_eigen_fn(_vector_prod_fn, max_eig_ph, n, k)
  #         curr_alpha, curr_beta = sess.run([alpha, beta])
  #         max_eig_1, _, _, _ = utils.eigen_tridiagonal(curr_alpha, curr_beta)
  #         curr_alpha_hat, curr_beta_hat, curr_Q_hat = sess.run([alpha_hat, beta_hat, Q_hat], feed_dict={max_eig_ph: max_eig_1})
  #         max_eig, max_vec, _, _ = utils.eigen_tridiagonal(curr_alpha_hat, curr_beta_hat)
  #         min_eig_lzs = max_eig + max_eig_1
  #         print('%d\t\t%-10.5g\t\t%.5g' %(
  #               k, np.abs(min_eig_lzs-min_eig_scipy)/np.abs(min_eig_scipy),
  #               np.abs(min_eig_lzs-min_eig_scipy)))


if __name__ == '__main__':
  tf.test.main()
