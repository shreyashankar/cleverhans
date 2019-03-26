"""Code with dual formulation for certification problem."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
from tensorflow.contrib import autograph
import numpy as np

from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse.linalg import lgmres
from scipy import optimize

from cleverhans.experimental.certification import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# Tolerance value for eigenvalue computation
TOL = 1E-5

# Bound on lowest value of certificate to check for numerical errors
LOWER_CERT_BOUND = -30.0
DEFAULT_LZS_PARAMS = {'min_iter': 5, 'max_iter': 50}


class DualFormulation(object):
  """DualFormulation is a class that creates the dual objective function
  and access to matrix vector products for the matrix that is constrained
  to be Positive semidefinite
  """

  def __init__(self, sess, dual_var, neural_net_param_object, test_input, true_class,
               adv_class, input_minval, input_maxval, epsilon,
               lzs_params=None):
    """Initializes dual formulation class.

    Args:
      sess: Tensorflow session
      dual_var: dictionary of dual variables containing a) lambda_pos
        b) lambda_neg, c) lambda_quad, d) lambda_lu
      neural_net_param_object: NeuralNetParam object created for the network
        under consideration
      test_input: clean example to certify around
      true_class: the class label of the test input
      adv_class: the label that the adversary tried to perturb input to
      input_minval: minimum value of valid input range
      input_maxval: maximum value of valid input range
      epsilon: Size of the perturbation (scaled for [0, 1] input)
      lzs_params: Parameters for Lanczos algorithm (dictionary) in the form:
        {
          'min_iter': 5
          'max_iter': 50
        }
    """
    self.sess = sess
    self.nn_params = neural_net_param_object
    self.test_input = tf.convert_to_tensor(test_input, dtype=tf.float32)
    self.true_class = true_class
    self.adv_class = adv_class
    self.input_minval = tf.convert_to_tensor(input_minval, dtype=tf.float32)
    self.input_maxval = tf.convert_to_tensor(input_maxval, dtype=tf.float32)
    self.epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
    self.lzs_params = lzs_params or DEFAULT_LZS_PARAMS.copy()
    self.final_linear = (self.nn_params.final_weights[adv_class, :]
                         - self.nn_params.final_weights[true_class, :])
    self.final_linear = tf.reshape(
        self.final_linear, shape=[tf.size(self.final_linear), 1])
    self.final_constant = (self.nn_params.final_bias[adv_class]
                           - self.nn_params.final_bias[true_class])

    # Computing lower and upper bounds
    # Note that lower and upper are of size nn_params.num_hidden_layers + 1
    self.lower = []
    self.upper = []

    # Also computing pre activation lower and upper bounds
    # to compute always-off and always-on units
    self.lower_relu = []
    self.upper_relu = []

    # Initializing at the input layer with \ell_\infty constraints
    self.lower_relu.append(
        tf.maximum(self.test_input - self.epsilon, self.input_minval))
    self.upper_relu.append(
        tf.minimum(self.test_input + self.epsilon, self.input_maxval))
    self.lower.append(self.lower_relu[0])
    self.upper.append(self.upper_relu[0])

    for i in range(0, self.nn_params.num_hidden_layers):
      lo_plus_up = self.nn_params.forward_pass(self.lower_relu[i] + self.upper_relu[i], i)
      lo_minus_up = self.nn_params.forward_pass(self.lower_relu[i] - self.upper_relu[i], i, is_abs=True)
      up_minus_lo = self.nn_params.forward_pass(self.upper_relu[i] - self.lower_relu[i], i, is_abs=True)
      current_lower = 0.5 * (lo_plus_up + lo_minus_up) + self.nn_params.biases[i]
      current_upper = 0.5 * (lo_plus_up + up_minus_lo) + self.nn_params.biases[i]
      self.lower.append(current_lower)
      self.upper.append(current_upper)
      self.lower_relu.append(tf.nn.relu(current_lower))
      self.upper_relu.append(tf.nn.relu(current_upper))

    # Run lower and upper because they don't change
    self.lower = self.sess.run(self.lower)
    self.upper = self.sess.run(self.upper)
    self.lower_relu = self.sess.run(self.lower_relu)
    self.upper_relu = self.sess.run(self.upper_relu)

    # Compute LP lower and upper bounds
    # self.lower_tilde = []
    # self.upper_tilde = []

    # for i in range(0, self.nn_params.num_hidden_layers):
    #   res = optimize.linprog(self.nn_params.forward_pass())

    # Using the preactivation lower and upper bounds
    # to compute the linear regions
    self.positive_indices = []
    self.negative_indices = []
    self.switch_indices = []

    for i in range(0, self.nn_params.num_hidden_layers + 1):
      # Positive index = 1 if the ReLU is always "on"
      self.positive_indices.append(np.asarray(self.lower[i] >= 0, dtype=np.float32))
      # Negative index = 1 if the ReLU is always off
      self.negative_indices.append(np.asarray(self.upper[i] <= 0, dtype=np.float32))
      # Switch index = 1 if the ReLU could be either on or off
      self.switch_indices.append(np.asarray(
          np.multiply(self.lower[i], self.upper[i]) < 0, dtype=np.float32))

    # Computing the optimization terms
    self.lambda_pos = [x for x in dual_var['lambda_pos']]
    self.lambda_neg = [x for x in dual_var['lambda_neg']]
    self.lambda_quad = [x for x in dual_var['lambda_quad']]
    self.lambda_lu = [x for x in dual_var['lambda_lu']]
    # self.lambda_lp = [x for x in dual_var['lambda_lp']]
    self.nu = dual_var['nu']
    self.min_eig_val_h = dual_var['min_eig_val_h'] if 'min_eig_val_h' in dual_var else None
    self.vector_g = None
    self.scalar_f = None
    self.matrix_h = None
    self.matrix_m = None
    self.matrix_m_dimension = 1 + np.sum(self.nn_params.sizes)

    # The primal vector in the SDP can be thought of as [layer_1, layer_2..]
    # In this concatenated version, dual_index[i] that marks the start
    # of layer_i
    # This is useful while computing implicit products with matrix H
    self.dual_index = [0]
    for i in range(self.nn_params.num_hidden_layers + 1):
      self.dual_index.append(self.dual_index[-1] + self.nn_params.sizes[i])

    # Construct objectives, matrices, and certificate
    self.set_differentiable_objective()
    if not self.nn_params.has_conv:
      self.get_full_psd_matrix()

    # Setup Lanczos functionality for compute certificate
    self.construct_lanczos_params()

  def construct_lanczos_params(self):
    """Computes matrices T and V using the Lanczos algorithm.

    Args:
      k: number of iterations and dimensionality of the tridiagonal matrix
    Returns:
      eig_vec: eigen vector corresponding to min eigenvalue
    """
    # Using autograph to automatically handle
    # the control flow of minimum_eigen_vector
    self.min_eigen_vec = autograph.to_graph(utils.tf_lanczos_smallest_eigval)

    # @autograph.do_not_convert()
    def _m_vector_prod_fn(x):
      return self.get_psd_product(x)

    # @autograph.do_not_convert()
    def _h_vector_prod_fn(x):
      return self.get_h_product(x)

    # Construct nodes for computing eigenvalue of M
    # Create random vector with Euclidean norm 1
    self.m_min_vec_estimate = np.zeros(shape=(self.matrix_m_dimension, 1), dtype=np.float64)
    self.m_min_vec_ph = tf.placeholder(shape=(self.matrix_m_dimension, 1), dtype=tf.float64, name="m_min_vec_ph")
    self.m_min_eig, self.m_min_vec = self.min_eigen_vec(_m_vector_prod_fn,
                                                      self.matrix_m_dimension,
                                                      self.m_min_vec_ph,
                                                      self.lzs_params['max_iter'],
                                                      dtype=tf.float64)
    self.h_min_vec_estimate = np.zeros(shape=(self.matrix_m_dimension - 1, 1), dtype=np.float64)
    self.h_min_vec_ph = tf.placeholder(shape=(self.matrix_m_dimension - 1, 1), dtype=tf.float64, name="h_min_vec_ph")
    self.h_min_eig, self.h_min_vec = self.min_eigen_vec(_h_vector_prod_fn,
                                                      self.matrix_m_dimension-1,
                                                      self.h_min_vec_ph,
                                                      self.lzs_params['max_iter'],
                                                      dtype=tf.float64)

  def set_differentiable_objective(self):
    """Function that constructs minimization objective from dual variables."""
    # Checking if graphs are already created
    if self.vector_g is not None:
      return

    # Computing the scalar term
    bias_sum = 0
    for i in range(0, self.nn_params.num_hidden_layers):
      bias_sum = bias_sum + tf.reduce_sum(
          tf.multiply(self.nn_params.biases[i], self.lambda_pos[i + 1]))
    lu_sum = 0
    for i in range(0, self.nn_params.num_hidden_layers + 1):
      lu_sum = lu_sum + tf.reduce_sum(
          tf.multiply(tf.multiply(self.lower[i], self.upper[i]),
                      self.lambda_lu[i]))

    self.scalar_f = -bias_sum - lu_sum + self.final_constant

    # Computing the vector term
    g_rows = []
    for i in range(0, self.nn_params.num_hidden_layers):
      if i > 0:
        current_row = (self.lambda_neg[i] + self.lambda_pos[i] -
                       self.nn_params.forward_pass(self.lambda_pos[i+1],
                                                   i, is_transpose=True) +
                       tf.multiply(self.lower[i]+self.upper[i],
                                   self.lambda_lu[i]) +
                       tf.multiply(self.lambda_quad[i],
                                   self.nn_params.biases[i-1]))
      else:
        current_row = (-self.nn_params.forward_pass(self.lambda_pos[i+1],
                                                    i, is_transpose=True)
                       + tf.multiply(self.lower[i]+self.upper[i],
                                     self.lambda_lu[i]))
      g_rows.append(current_row)

    # Term for final linear term
    g_rows.append((self.lambda_pos[self.nn_params.num_hidden_layers] +
                   self.lambda_neg[self.nn_params.num_hidden_layers] +
                   self.final_linear +
                   tf.multiply((self.lower[self.nn_params.num_hidden_layers]+
                                self.upper[self.nn_params.num_hidden_layers]),
                               self.lambda_lu[self.nn_params.num_hidden_layers])
                   + tf.multiply(
                       self.lambda_quad[self.nn_params.num_hidden_layers],
                       self.nn_params.biases[
                           self.nn_params.num_hidden_layers-1])))
    self.vector_g = tf.concat(g_rows, axis=0)
    self.unconstrained_objective = self.scalar_f + 0.5 * self.nu

  def get_h_product(self, vector):
    """Function that provides matrix product interface with PSD matrix.

    Args:
      vector: the vector to be multiplied with matrix H

    Returns:
      result_product: Matrix product of H and vector
    """
    # Computing the product of matrix_h with beta (input vector)
    # At first layer, h is simply diagonal
    beta = vector
    h_beta_rows = []
    for i in range(self.nn_params.num_hidden_layers):
      # Split beta of this block into [gamma, delta]
      gamma = beta[self.dual_index[i]:self.dual_index[i + 1]]
      delta = beta[self.dual_index[i + 1]:self.dual_index[i + 2]]
      # import pdb; pdb.set_trace()

      # Expanding the product with diagonal matrices
      if i == 0:
        h_beta_rows.append(
            tf.multiply(2 * self.lambda_lu[i], gamma) -
            self.nn_params.forward_pass(
                tf.multiply(self.lambda_quad[i + 1], delta),
                i,
                is_transpose=True))
      else:
        h_beta_rows[i] = (h_beta_rows[i] +
                          tf.multiply(self.lambda_quad[i] +
                                      self.lambda_lu[i], gamma) -
                          self.nn_params.forward_pass(
                              tf.multiply(self.lambda_quad[i+1], delta),
                              i, is_transpose=True))
        # print(h_beta_rows[i].get_shape())

      new_row = (
          tf.multiply(self.lambda_quad[i + 1] + self.lambda_lu[i + 1], delta) -
          tf.multiply(self.lambda_quad[i + 1],
                      self.nn_params.forward_pass(gamma, i)))
      # print(new_row.get_shape())
      h_beta_rows.append(new_row)

    # Last boundary case
    h_beta_rows[self.nn_params.num_hidden_layers] = (
        h_beta_rows[self.nn_params.num_hidden_layers] +
        tf.multiply((self.lambda_quad[self.nn_params.num_hidden_layers] +
                     self.lambda_lu[self.nn_params.num_hidden_layers]),
                    delta))

    h_beta = tf.concat(h_beta_rows, axis=0)
    return h_beta

  def get_psd_product(self, vector):
    """Function that provides matrix product interface with PSD matrix.

    Args:
      vector: the vector to be multiplied with matrix M

    Returns:
      result_product: Matrix product of M and vector
    """
    # For convenience, think of x as [\alpha, \beta]
    alpha = tf.reshape(vector[0], shape=[1, 1])
    beta = vector[1:]
    # Computing the product of matrix_h with beta part of vector
    # At first layer, h is simply diagonal
    h_beta = self.get_h_product(beta)

    # Constructing final result using vector_g
    result = tf.concat(
        [
            alpha * self.nu + tf.reduce_sum(tf.multiply(beta, self.vector_g)),
            tf.multiply(alpha, self.vector_g) + h_beta
        ],
        axis=0)
    return result

  def get_full_psd_matrix(self):
    """Function that returns the tf graph corresponding to the entire matrix M.

    Returns:
      matrix_h: unrolled version of tf matrix corresponding to H
      matrix_m: unrolled tf matrix corresponding to M
    """
    if self.matrix_m is not None:
      return self.matrix_h, self.matrix_m

    # Computing the matrix term
    h_columns = []
    for i in range(self.nn_params.num_hidden_layers + 1):
      current_col_elems = []
      for j in range(i):
        current_col_elems.append(
            tf.zeros([self.nn_params.sizes[j], self.nn_params.sizes[i]]))

    # For the first layer, there is no relu constraint
      if i == 0:
        current_col_elems.append(utils.diag(self.lambda_lu[i]))
      else:
        current_col_elems.append(
            utils.diag(self.lambda_lu[i] + self.lambda_quad[i]))
      if i < self.nn_params.num_hidden_layers:
        current_col_elems.append(tf.matmul(
            utils.diag(-1 * self.lambda_quad[i + 1]),
            self.nn_params.weights[i]))
      for j in range(i + 2, self.nn_params.num_hidden_layers + 1):
        current_col_elems.append(
            tf.zeros([self.nn_params.sizes[j], self.nn_params.sizes[i]]))
      current_column = tf.concat(current_col_elems, 0)
      h_columns.append(current_column)

    self.matrix_h = tf.concat(h_columns, 1)
    self.matrix_h = (self.matrix_h + tf.transpose(self.matrix_h))

    self.matrix_m = tf.concat(
        [
            tf.concat([self.nu, tf.transpose(self.vector_g)], axis=1),
            tf.concat([self.vector_g, self.matrix_h], axis=1)
        ],
        axis=0)
    return self.matrix_h, self.matrix_m

  def dump_M(self, iter, feed_dict):
    """Function to construct entire matrix and save to a file."""
    n = self.matrix_m_dimension
    M = np.zeros((n, n))
    input_vector_m = tf.placeholder(tf.float32, shape=(n, 1))
    output_vector_m = self.get_psd_product(input_vector_m)
    for i in range(n):
      input_vector = np.zeros((n, 1), dtype=np.float32)
      input_vector[i, 0] = 1.0
      feed_dict.update({input_vector_m: input_vector})
      M[:, i] = np.reshape(self.sess.run(output_vector_m, feed_dict=feed_dict), (n,))
    np.save('cleverhans/experimental/certification/matrices/iter_' + str(iter), M)

  def make_m_psd(self, original_nu, min_eig_val_h, feed_dict):
    """Run binary search to find a value for nu that makes M PSD
    Args:
      feed_dict: dictionary of updated lambda variables to feed into M
    Returns:
      new_nu: new value of nu
    """
    feed_dict.update({self.nu: original_nu, self.min_eig_val_h: min_eig_val_h})
    _, min_eig_val_m = self.get_lanczos_eig(compute_m=True, feed_dict=feed_dict)
    # min_eig_val_m = self.get_scipy_eig(feed_dict)
    print("min eig scipy: " + str(self.get_scipy_eig(feed_dict)))
    print("min eig lzs: " + str(min_eig_val_m))

    lower_nu = original_nu
    upper_nu = original_nu
    num_iter = 0

    # Find an upper bound on nu
    while min_eig_val_m - TOL < 0:
      if num_iter >= 5:
        break
      num_iter += 1
      upper_nu *= 1.3
      feed_dict.update({self.nu: upper_nu})
      _, min_eig_val_m = self.get_lanczos_eig(compute_m=True, feed_dict=feed_dict)
      # min_eig_val_m = self.get_scipy_eig(feed_dict)
      print(min_eig_val_m)

      # print("upper nu: " + str(upper_nu))
      # print("min eig: " + str(min_eig_val_m))
      # print("min eig scipy: " + str(self.get_scipy_eig(feed_dict)))
    final_nu = upper_nu

    # Perform binary search to find best value of nu
    while lower_nu <= upper_nu:
      if num_iter >= 10:
        break
      num_iter += 1
      mid_nu = (lower_nu + upper_nu) / 2
      feed_dict.update({self.nu: mid_nu})
      # print("mid_nu: " + str(mid_nu))
      _, min_eig_val_m = self.get_lanczos_eig(compute_m=True, feed_dict=feed_dict)
      # min_eig_val_m = self.get_scipy_eig(feed_dict)
      print(min_eig_val_m)
      if min_eig_val_m - TOL < 0:
        lower_nu = mid_nu
      else:
        upper_nu = mid_nu
    
    final_nu = upper_nu
    # _, min_eig_val_m = self.get_lanczos_eig(compute_m=True, feed_dict={self.nu: final_nu, self.min_eig_val_h: min_eig_val_h})
    # print("ori nu: " + str(original_nu))
    # print("final nu: " + str(final_nu))
    # print("min eig val m: " + str(min_eig_val_m))

    return original_nu, final_nu

  def get_scipy_eig(self, feed_dict):
    input_vector_m = tf.placeholder(tf.float32, shape=(self.matrix_m_dimension, 1))
    output_vector_m = self.get_psd_product(input_vector_m)

    def np_vector_prod_fn_m(np_vector):
      np_vector = np.reshape(np_vector, [-1, 1])
      feed_dict.update({input_vector_m:np_vector})
      # feed_dict = {input_vector_h:np_vector}
      output_np_vector = self.sess.run(output_vector_m, feed_dict=feed_dict)
      return output_np_vector
    linear_operator_m = LinearOperator((self.matrix_m_dimension,
                                        self.matrix_m_dimension ),
                                        matvec=np_vector_prod_fn_m)
    min_eig_val_m_scipy, _ = eigs(linear_operator_m,
                              k=1, which='SR', tol=TOL)
    return min_eig_val_m_scipy

  def get_lanczos_eig(self, compute_m=True, feed_dict={}):
    """Computes the min eigen value and corresponding vector of matrix M or H
    using the Lanczos algorithm.

    Args:
      compute_m: boolean to determine whether we should compute eig val/vec
        for M or for H. True for M; False for H.

    Returns:
      min_eig_vec: Corresponding eigen vector to min eig val
      eig_val: Minimum eigen value
    """
    start = time.time()
    if compute_m:
      min_eig, min_vec = self.sess.run([self.m_min_eig, self.m_min_vec], feed_dict=feed_dict)

    else:
      min_eig, min_vec = self.sess.run([self.h_min_eig, self.h_min_vec], feed_dict=feed_dict)

    return min_vec, min_eig
  
  def save_dual(self, folder):
    """Function to save the dual variables 
    Args:
      folder: The folder to save the dual variables 
      sess: current tensorflow session whose dual variables are to be saved 
    """ 
    if not tf.gfile.IsDirectory(folder):
      tf.gfile.MkDir(folder)
    [current_lambda_pos, current_lambda_neg, current_lambda_quad, 
    current_lambda_lu, current_nu] = self.sess.run([self.lambda_pos, 
      self.lambda_neg, self.lambda_quad, self.lambda_lu, self.nu])
    np.save(os.path.join(folder, 'lambda_pos'), current_lambda_pos)
    np.save(os.path.join(folder, 'lambda_neg'), current_lambda_neg)
    np.save(os.path.join(folder, 'lambda_lu'), current_lambda_lu)
    np.save(os.path.join(folder, 'lambda_quad'), current_lambda_quad)
    np.save(os.path.join(folder, 'nu'), current_nu)
    print('Saved the current dual variables in folder:', folder)

  def compute_certificate(self, current_step, nu, min_eig_val_h, feed_dict):
    """ Function to compute the certificate based either current value
    or dual variables loaded from dual folder """

    # Make matrix M PSD
    # input_vector_h = tf.placeholder(tf.float32, shape=(self.matrix_m_dimension-1, 1))
    # output_vector_h = self.get_h_product(input_vector_h)

    # def np_vector_prod_fn_h(np_vector):
    #   np_vector = np.reshape(np_vector, [-1, 1])
    #   feed_dict.update({self.nu: nu, self.min_eig_val_h: min_eig_val_h,input_vector_h:np_vector})
    #   # feed_dict = {input_vector_h:np_vector}
    #   output_np_vector = self.sess.run(output_vector_h, feed_dict=feed_dict)
    #   return output_np_vector
    # linear_operator_h = LinearOperator((self.matrix_m_dimension - 1,
    #                                     self.matrix_m_dimension - 1),
    #                                     matvec=np_vector_prod_fn_h)
    # min_eig_val_h_scipy, _ = eigs(linear_operator_h,
    #                           k=1, which='SR', tol=TOL)
    # print("h min eig: " + str(min_eig_val_h_scipy))
    _, second_term = self.make_m_psd(nu, min_eig_val_h, feed_dict)
    tf.logging.info("nu after modifying: " + str(second_term))
    feed_dict.update({self.nu: second_term, self.min_eig_val_h: min_eig_val_h})

    # Add 0.05 to final nu to account for numerical instability
    computed_certificate = self.sess.run(self.unconstrained_objective, feed_dict=feed_dict)

    tf.logging.info('Inner step: %d, current value of certificate: %f',
                      current_step, computed_certificate)

    # Sometimes due to either overflow or instability in inverses,
    # the returned certificate is large and negative -- keeping a check
    if LOWER_CERT_BOUND < computed_certificate < 0:
      _, min_eig_val_m = self.get_lanczos_eig(feed_dict=feed_dict)
      tf.logging.info("min eig val from lanczos: " + str(min_eig_val_m))
      input_vector_m = tf.placeholder(tf.float32, shape=(self.matrix_m_dimension, 1))
      output_vector_m = self.get_psd_product(input_vector_m)

      def np_vector_prod_fn_m(np_vector):
        np_vector = np.reshape(np_vector, [-1, 1])
        feed_dict.update({input_vector_m:np_vector})
        output_np_vector = self.sess.run(output_vector_m, feed_dict=feed_dict)
        return output_np_vector
      linear_operator_m = LinearOperator((self.matrix_m_dimension,
                                          self.matrix_m_dimension),
                                         matvec=np_vector_prod_fn_m)
      # Performing shift invert scipy operation when eig val estimate is available
      min_eig_val_m_scipy, _ = eigs(linear_operator_m,
                              k=1, which='SR', tol=TOL)
      
      print("min eig val m from scipy: " + str(min_eig_val_m_scipy))

      if min_eig_val_m - TOL > 0:
        tf.logging.info('Found certificate of robustness!')
        return True

    return False
