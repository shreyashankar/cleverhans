"""File containing some simple helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy.linalg import eigh_tridiagonal
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator, aslinearoperator

def diag(diag_elements):
  """Function to create tensorflow diagonal matrix with input diagonal entries.

  Args:
    diag_elements: tensor with diagonal elements

  Returns:
    tf matrix with diagonal entries as diag_elements
  """
  return tf.diag(tf.reshape(diag_elements, [-1]))


def initialize_dual(neural_net_params_object, init_dual_file=None,
                    random_init_variance=0.01, init_nu=200.0):
  """Function to initialize the dual variables of the class.

  Args:
    neural_net_params_object: Object with the neural net weights, biases
      and types
    init_dual_file: Path to file containing dual variables, if the path
      is empty, perform random initialization
      Expects numpy dictionary with
      lambda_pos_0, lambda_pos_1, ..
      lambda_neg_0, lambda_neg_1, ..
      lambda_quad_0, lambda_quad_1, ..
      lambda_lu_0, lambda_lu_1, ..
      random_init_variance: variance for random initialization
    init_nu: Value to initialize nu variable with

  Returns:
    dual_var: dual variables initialized appropriately.
  """
  lambda_pos = []
  lambda_neg = []
  lambda_quad = []
  lambda_lu = []

  if init_dual_file is None:
    for i in range(0, neural_net_params_object.num_hidden_layers + 1):
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_pos.append(tf.get_variable('lambda_pos_' + str(i),
                                        initializer=initializer,
                                        dtype=tf.float32))
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_neg.append(tf.get_variable('lambda_neg_' + str(i),
                                        initializer=initializer,
                                        dtype=tf.float32))
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_quad.append(tf.get_variable('lambda_quad_' + str(i),
                                         initializer=initializer,
                                         dtype=tf.float32))
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_lu.append(tf.get_variable('lambda_lu_' + str(i),
                                       initializer=initializer,
                                       dtype=tf.float32))
    nu = tf.get_variable('nu', initializer=init_nu)
  else:
    # Loading from file
    dual_var_init_val = np.load(init_dual_file).item()
    for i in range(0, neural_net_params_object.num_hidden_layers + 1):
      lambda_pos.append(
          tf.get_variable('lambda_pos_' + str(i),
                          initializer=dual_var_init_val['lambda_pos'][i],
                          dtype=tf.float32))
      lambda_neg.append(
          tf.get_variable('lambda_neg_' + str(i),
                          initializer=dual_var_init_val['lambda_neg'][i],
                          dtype=tf.float32))
      lambda_quad.append(
          tf.get_variable('lambda_quad_' + str(i),
                          initializer=dual_var_init_val['lambda_quad'][i],
                          dtype=tf.float32))
      lambda_lu.append(
          tf.get_variable('lambda_lu_' + str(i),
                          initializer=dual_var_init_val['lambda_lu'][i],
                          dtype=tf.float32))
    nu = tf.get_variable('nu', initializer=1.0*dual_var_init_val['nu'])
  dual_var = {'lambda_pos': lambda_pos, 'lambda_neg': lambda_neg,
              'lambda_quad': lambda_quad, 'lambda_lu': lambda_lu, 'nu': nu}
  return dual_var

def lanczos_eigvec(vector_prod_fn, b, k, collapse_tol=1e-12):
  """Approximation of smallest eigvector and associated eigenvalue
  via the Lanczos methd.

  Input:
  A: A LinearOperator object with shape=(d,d) and method matvec giving
    matrix-vector products with underlying symmetric d by d linear operator
  b: np.array with shape=(d,) generating the Krylov subspace. Should be a
    spherically symmetric (e.g. Gaussian) random vector
  k: Number of Lanczos iterations to run.
    collapse_tol: Tolerance for declaring that the Lanczos iterations
    converged to an invariant subspace, i.e. collapse. If such collapse occurs,
    for random b, then the resulting approximation for the eigenvalue and
    eigenvector should be very accurate. However, accurate approximation can
    also occur without such collapse.

  Output:
  lambda_min: Scalar approximating smallest eigenvalue of A
  v_min: np.array of shape=(d,) and unit norm approximating eigenvector
    corresponding to lambda_min.
  """

  b = np.squeeze(b)
  d = b.shape[0]
  # Lanczos process to generate symmetric tridiagonal matrix with diagonal
  # alpha and off-diagonal beta
  alpha = np.zeros(k, np.float64)
  beta = np.zeros(k, np.float64)
  Q = np.zeros((d, k), np.float64)

  w = b / norm(b)
  w_prev = np.zeros(d, b.dtype)
  w_ = np.zeros(d, b.dtype)
  i = 0
  krylov_collapsed = False
  for i in range(k):
    if i > 0:
      beta[i] = norm(w)
      if beta[i] < collapse_tol:
        print("hahahaha")
        krylov_collapsed = True
        break
      w /= beta[i]
    Q[:, i] = w
    w_ = vector_prod_fn(w.reshape(d, 1)).reshape(d)  # to make more efficient, perform matvec in place
    alpha[i] = np.tensordot(w, w_, axes=1)
    w_ -= alpha[i] * w
    w_ -= beta[i] * w_prev
    w, w_prev, w_ = w_, w, w_prev

  # in case the Krylov subspace collapsed, truncate it
  if krylov_collapsed:
      k = i
      alpha = alpha[:k]
      beta = beta[:k]
      Q = Q[:k]

  # Find minimum eigenvalue and eigenvector of tridiagonal matrix
  return alpha, beta, Q

def lanczos_decomp(vector_prod_fn, n, k):
  """Function that performs the Lanczos algorithm on a matrix.

  Args:
    vector_prod_fn: function which returns product H*x, where H is a matrix for
      which we computing eigenvector.
    n: dimensionality of matrix H
    k: number of iterations and dimensionality of the tridiagonal matrix to
      return

  Returns:
    alpha: vector of diagonal elements of T
    beta: vector of off-diagonal elements of T
    Q: orthonormal basis matrix for the Krylov subspace
  """
  Q = tf.zeros([n, 1])
  v = tf.random_uniform([n, 1])
  v = v / tf.norm(v)
  Q = tf.concat([Q, v], axis=1)

  # diagonals of the tridiagonal matrix
  beta = tf.constant(0.0, dtype=tf.float32, shape=[1])
  alpha = tf.constant(0.0, dtype=tf.float32, shape=[1])

  for i in range(k):
    v = vector_prod_fn(tf.reshape(Q[:, i+1], [n, 1]))
    v = tf.reshape(v, [n,])
    curr_alpha = tf.reshape(tf.reduce_sum(v * Q[:, i+1]), [1,])
    alpha = tf.concat([alpha, curr_alpha], axis=0)
    v = v-beta[-1]*Q[:, i]-alpha[-1]*Q[:, i+1]
    curr_beta = tf.reshape(tf.norm(v), [1,])
    beta = tf.concat([beta, curr_beta], axis=0)
    curr_norm = tf.reshape(v/(beta[-1]+1e-8), [n, 1])
    Q = tf.concat([Q, curr_norm], axis=1)

  alpha = tf.slice(alpha, begin=[1], size=[-1])
  beta = tf.slice(beta, begin=[1], size=[k-1])
  Q = tf.slice(Q, begin=[0, 1], size=[-1, k])
  return alpha, beta, Q

def check_collapse_tol(curr_beta, collapse_tol):
  res = tf.cond(curr_beta < collapse_tol)
  return res

def body(vector_prod_fn,  alpha, beta, Q, w, w_, w_prev, i, k, check_collapse_tol):
  curr_beta = tf.norm(w)
  curr_beta_rs = tf.reshape(curr_beta, [1])
  beta = tf.concat([beta, curr_beta_rs], axis=0)

  w = w / (curr_beta + 1e-8)
  w_ = vector_prod_fn(w)
  curr_alpha = tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)
  curr_alpha_rs = tf.reshape(curr_alpha, [1])
  Q = tf.concat([Q, w], axis=1)
  alpha = tf.concat([alpha, curr_alpha_rs], axis=0)
  w_ -= tf.scalar_mul(curr_alpha, w)
  w_ -= tf.scalar_mul(curr_beta, w_prev)
  w, w_prev, w_ = w_, w, w_prev
  i += 1


def condition(vector_prod_fn, alpha, beta, Q, w, w_, w_prev, i, k, check_collapse_tol):
  return i < k and tf.norm(w) > check_collapse_tol

def lzs_tf(vector_prod_fn, n, k, b, check_collapse_tol=1e-12):
  beta = tf.constant(0.0, dtype=tf.float32, shape=[1])

  # Create initial guesses
  w = b / tf.norm(b)
  w_prev = tf.zeros([n, 1], dtype=tf.float32)
  w_ = tf.zeros([n, 1], dtype=tf.float32)

  # Iteration zero is different
  Q = w
  w_ = vector_prod_fn(w)
  curr_alpha = tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)
  alpha = tf.reshape(curr_alpha, [1])
  w_ -= tf.scalar_mul(curr_alpha, w)
  w, w_prev, w_ = w_, w, w_prev
  i = 0

  alpha, beta, Q = tf.while_loop(condition, body, [vector_prod_fn, alpha, beta, Q, w, w_, w_prev, i, k, check_collapse_tol])

def _lzs_two(vector_prod_fn, n, k, b, sess, collapse_tol=1e-12):
  beta = tf.constant(0.0, dtype=tf.float64, shape=[1])

  # Create initial guesses
  # v = tf.random_normal([n, 1])
  w = b / tf.norm(b)
  w_prev = tf.zeros([n, 1], dtype=tf.float64)
  w_ = tf.zeros([n, 1], dtype=tf.float64)

  # Iteration zero is different
  Q = w
  w_ = tf.cast(vector_prod_fn(tf.cast(w, tf.float32)), tf.float64)
  npw_ = sess.run(w_)
  print(npw_)
  npw = sess.run(w)
  # print(np.tensordot(np.squeeze(npw_), np.squeeze(npw), axes=1))
  curr_alpha = tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)
  # print(sess.run(curr_alpha))
  # curr_alpha = tf.matmul(w_, w, transpose_a=True)
  alpha = tf.reshape(curr_alpha, [1])
  w_ -= tf.scalar_mul(curr_alpha, w)
  w, w_prev, w_ = w_, w, w_prev

  curr_beta = tf.norm(w)
  curr_beta_rs = tf.reshape(curr_beta, [1])
  beta = tf.concat([beta, curr_beta_rs], axis=0)

  w = w / (curr_beta)
  Q = tf.concat([Q, w], axis=1)
  w_ = tf.cast(vector_prod_fn(tf.cast(w, tf.float32)), tf.float64)
  npw_ = sess.run(w_)
  npw = sess.run(w)
  # print(np.tensordot(np.squeeze(npw_), np.squeeze(npw), axes=1))
  # print(sess.run(tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)))
  curr_alpha = tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)
  curr_alpha_rs = tf.reshape(curr_alpha, [1])

  alpha = tf.concat([alpha, curr_alpha_rs], axis=0)
  w_ -= tf.scalar_mul(curr_alpha, w)
  w_ -= tf.scalar_mul(curr_beta, w_prev)
  w, w_prev, w_ = w_, w, w_prev

  curr_beta = tf.norm(w)
  curr_beta_rs = tf.reshape(curr_beta, [1])
  beta = tf.concat([beta, curr_beta_rs], axis=0)

  w = w / (curr_beta)
  Q = tf.concat([Q, w], axis=1)
  w_ = tf.cast(vector_prod_fn(tf.cast(w, tf.float32)), tf.float64)
  npw_ = sess.run(w_)
  npw = sess.run(w)
  # print(np.tensordot(np.squeeze(npw_), np.squeeze(npw), axes=1))
  # print(sess.run(tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)))
  curr_alpha = tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)
  curr_alpha_rs = tf.reshape(curr_alpha, [1])

  alpha = tf.concat([alpha, curr_alpha_rs], axis=0)
  w_ -= tf.scalar_mul(curr_alpha, w)
  w_ -= tf.scalar_mul(curr_beta, w_prev)
  w, w_prev, w_ = w_, w, w_prev

  curr_beta = tf.norm(w)
  curr_beta_rs = tf.reshape(curr_beta, [1])
  beta = tf.concat([beta, curr_beta_rs], axis=0)

  w = w / (curr_beta)
  Q = tf.concat([Q, w], axis=1)
  w_ = tf.cast(vector_prod_fn(tf.cast(w, tf.float32)), tf.float64)
  curr_alpha = tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)
  curr_alpha_rs = tf.reshape(curr_alpha, [1])

  alpha = tf.concat([alpha, curr_alpha_rs], axis=0)
  w_ -= tf.scalar_mul(curr_alpha, w)
  w_ -= tf.scalar_mul(curr_beta, w_prev)
  w, w_prev, w_ = w_, w, w_prev

  # beta = tf.slice(beta, begin=[1], size=[-1])
  return alpha, beta, Q

def lzs_three(vector_prod_fn, n, k, b, collapse_tol=1e-12):
  beta = tf.TensorArray(tf.float64, size=1, dynamic_size=True)
  beta = beta.write(0, tf.constant([0.0], dtype=tf.float64))

  w = b / tf.norm(b)
  w_prev = tf.zeros([n, 1], dtype=tf.float64)
  w_ = tf.zeros([n, 1], dtype=tf.float64)

  Q = tf.TensorArray(w.dtype, size=1, dynamic_size=True)
  Q = Q.write(0, w)
  w_ = tf.cast(vector_prod_fn(tf.cast(w, tf.float32)), tf.float64)
  curr_alpha = tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)
  alpha = tf.TensorArray(tf.float64, size=1, dynamic_size=True)
  alpha = alpha.write(0, tf.reshape(curr_alpha, [1]))
  w_ -= tf.scalar_mul(curr_alpha, w)

  w, w_prev, w_ = w_, w, w_prev

  for i in tf.range(1, k):
    curr_beta = tf.norm(w)
    curr_beta_rs = tf.reshape(curr_beta, [1])
    w = w / curr_beta
    w_ = tf.cast(vector_prod_fn(tf.cast(w, tf.float32)), tf.float64)
    curr_alpha = tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)
    curr_alpha_rs = tf.reshape(curr_alpha, [1])

    if curr_beta < collapse_tol:
      break
    
    # Update alpha, beta, Q
    beta = beta.write(i, curr_beta_rs)
    Q = Q.write(i, w)
    alpha = alpha.write(i, curr_alpha_rs)

    w_ -= tf.scalar_mul(curr_alpha, w)
    w_ -= tf.scalar_mul(curr_beta, w_prev)
    w, w_prev, w_ = w_, w, w_prev
  
  alpha = tf.squeeze(alpha.stack())
  beta = tf.squeeze(beta.stack())
  Q = tf.transpose(tf.squeeze(Q.stack()))

  return alpha, beta, Q

def tf_lanczos_eigval(vector_prod_fn, n, k, collapse_tol=1e-12, dtype=tf.float32):
  alpha = tf.TensorArray(dtype, size=1, dynamic_size=True, element_shape=())  # diagonal elements
  beta = tf.TensorArray(dtype, size=0, dynamic_size=True, element_shape=())   # off diagonal elements
  
  b = tf.random_normal(shape=[n,1], dtype=dtype)
  w = b / tf.norm(b)
  
  # iteration 0:
  w_ = vector_prod_fn(w)
  cur_alpha = tf.reduce_sum(w_ * w)
  alpha = alpha.write(0, cur_alpha)
  w_ = w_ - tf.scalar_mul(cur_alpha, w)
  w_prev = w
  w = w_
  
  # subsequent iterations:
  for i in tf.range(1, k):
    cur_beta = tf.norm(w)
    if cur_beta < collapse_tol:
      # return early if Krylov subspace collapsed
      break

    w = w / cur_beta

    w_ = vector_prod_fn(w)
    cur_alpha = tf.reduce_sum(w_ * w)
    
    alpha = alpha.write(i, cur_alpha)
    beta = beta.write(i-1, cur_beta)
    i += 1
    w_ = w_ - tf.scalar_mul(cur_alpha, w) - tf.scalar_mul(cur_beta, w_prev)
    w_prev = w
    w = w_
    
  return alpha.stack(), beta.stack()


def lzs_two(vector_prod_fn, n, k, b, collapse_tol=1e-12):
  curr_beta = tf.norm(b)
  curr_beta_rs = tf.reshape(curr_beta, [1])
  beta = tf.constant(0.0, dtype=tf.float64, shape=[1])

  # Create initial guesses
  w = b / tf.norm(b)
  w_prev = tf.zeros([n, 1], dtype=tf.float64)
  w_ = tf.zeros([n, 1], dtype=tf.float64)

  # Iteration zero is different
  Q = w
  w_ = tf.cast(vector_prod_fn(tf.cast(w, tf.float32)), tf.float64)
  curr_alpha = tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)
  curr_alpha_rs = tf.reshape(curr_alpha, [1])
  alpha = curr_alpha_rs
  w_ -= tf.scalar_mul(curr_alpha, w)
  w, w_prev, w_ = w_, w, w_prev

  # i = 1
  should_stop = tf.norm(w) < collapse_tol
  for i in range(1, k):
    # in case the Krylov subspace collapsed, truncate it
    if should_stop:
      continue
    should_stop = tf.norm(w) < collapse_tol

    curr_beta = tf.norm(w)
    curr_beta_rs = tf.reshape(curr_beta, [1])
    beta = tf.concat([beta, curr_beta_rs], axis=0)

    w = w / (curr_beta)
    Q = tf.concat([Q, w], axis=1)
    w_ = tf.cast(vector_prod_fn(tf.cast(w, tf.float32)), tf.float64)
    curr_alpha = tf.tensordot(tf.squeeze(w_), tf.squeeze(w), 1)
    curr_alpha_rs = tf.reshape(curr_alpha, [1])

    alpha = tf.concat([alpha, curr_alpha_rs], axis=0)
    w_ -= tf.scalar_mul(curr_alpha, w)
    w_ -= tf.scalar_mul(curr_beta, w_prev)
    w, w_prev, w_ = w_, w, w_prev

  # beta = tf.slice(beta, begin=[1], size=[-1])
  return alpha, beta, Q

def eigen_tridiagonal(alpha, beta, Q, maximum=True):
  """Computes eigenvalues of a tridiagonal matrix.

  Args:
    alpha: vector of diagonal elements
    beta: vector of off-diagonal elements
    max: whether to compute the max or min magnitude eigenvalue
  Returns:
    eig: eigenvalue corresponding to max or min magnitude eigenvalue
    eig_vector: eigenvalue corresponding to eig
    eig_vectors: all eigenvectors
    eig_values: all eigenvalues
  """
  eig_values, eig_vectors = eigh_tridiagonal(alpha, beta, select='i', select_range=[0, 0])
  if maximum:
    ind_eig = np.argmax(np.abs(eig_values))
  else:
    # ind_eig = np.argmin(eig_values)
    ind_eig = 0
  eig = eig_values[ind_eig]
  eig_vector = np.matmul(Q, eig_vectors[:, ind_eig])
  eig_vector /= np.linalg.norm(eig_vector)
  return eig, eig_vector, eig_vectors, eig_values

def eig_one_step(current_vector, learning_rate, vector_prod_fn):
  """Function that performs one step of gd (variant) for min eigen value.

  Args:
    current_vector: current estimate of the eigen vector with minimum eigen
      value.
    learning_rate: learning rate.
    vector_prod_fn: function which returns product H*x, where H is a matrix for
      which we computing eigenvector.

  Returns:
    updated vector after one step
  """
  grad = 2*vector_prod_fn(current_vector)
  # Current objective = (1/2)*v^T (2*M*v); v = current_vector
  # grad = 2*M*v
  current_objective = tf.reshape(tf.matmul(tf.transpose(current_vector),
                                           grad) / 2., shape=())

  # Project the gradient into the tangent space of the constraint region.
  # This way we do not waste time taking steps that try to change the
  # norm of current_vector
  grad = grad - current_vector*tf.matmul(tf.transpose(current_vector), grad)
  grad_norm = tf.norm(grad)
  grad_norm_sq = tf.square(grad_norm)

  # Computing normalized gradient of unit norm
  norm_grad = grad / grad_norm

  # Computing directional second derivative (dsd)
  # dsd = 2*g^T M g, where g is normalized gradient
  directional_second_derivative = (
      tf.reshape(2*tf.matmul(tf.transpose(norm_grad),
                             vector_prod_fn(norm_grad)),
                 shape=()))

  # Computing grad^\top M grad [useful to compute step size later]
  # Just a rescaling of the directional_second_derivative (which uses
  # normalized gradient
  grad_m_grad = directional_second_derivative*grad_norm_sq / 2

  # Directional_second_derivative/2 = objective when vector is norm_grad
  # If this is smaller than current objective, simply return that
  if directional_second_derivative / 2. < current_objective:
    return norm_grad

  # If curvature is positive, jump to the bottom of the bowl
  if directional_second_derivative > 0.:
    step = -1. * grad_norm / directional_second_derivative
  else:
    # If the gradient is very small, do not move
    if grad_norm_sq <= 1e-16:
      step = 0.0
    else:
      # Make a heuristic guess of the step size
      step = -2. * tf.reduce_sum(current_vector*grad) / grad_norm_sq
      # Computing gain using the gradient and second derivative
      gain = -(2 * tf.reduce_sum(current_vector*grad) +
               (step*step) * grad_m_grad)

      # Fall back to pre-determined learning rate if no gain
      if gain < 0.:
        step = -learning_rate * grad_norm
  current_vector = current_vector + step * norm_grad
  return tf.nn.l2_normalize(current_vector)


def minimum_eigen_vector(x, num_steps, learning_rate, vector_prod_fn):
  """Computes eigenvector which corresponds to minimum eigenvalue.

  Args:
    x: initial value of eigenvector.
    num_steps: number of optimization steps.
    learning_rate: learning rate.
    vector_prod_fn: function which takes x and returns product H*x.

  Returns:
    approximate value of eigenvector.

  This function finds approximate value of eigenvector of matrix H which
  corresponds to smallest (by absolute value) eigenvalue of H.
  It works by solving optimization problem x^{T}*H*x -> min.
  """
  x = tf.nn.l2_normalize(x)
  for _ in range(num_steps):
    x = eig_one_step(x, learning_rate, vector_prod_fn)
  return x
