"""File containing some simple helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time


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

def python_lanczos(vector_prod_fn, scalar, n, k, logging=False):
  """Python version of lanczos for testing purposes"""
  b = np.random.uniform(0,1,n)
  Q = np.zeros([n, k+2], dtype = np.float32)
  Q[:, 1] = b/np.linalg.norm(b)

  # diagonals of the tridiagonal matrix
  beta = [0]
  alpha = []
  tim = 0

  for i in range(k):
    start = time.time()
    v = vector_prod_fn(Q[:,i+1].reshape((n, 1))) - scalar * Q[:,i+1].reshape((n, 1))
    end = time.time()
    tim += end - start
    v = v.reshape((n,))
    alpha.append(np.sum(Q[:,i+1] * v))
    v = v-beta[-1]*Q[:,i]-alpha[-1]*Q[:,i+1]
    beta.append(np.linalg.norm(v))
    Q[:,i+2] = v/(beta[-1]+1e-8)
  
  if logging:
    print("vector prod time: " + str(tim))
  return alpha, beta, Q

def lanczos_decomp(vector_prod_fn, scalar, n, k):
  Q = tf.zeros([n, 1])
  v = tf.random_uniform([n, 1])
  v = v / tf.norm(v)
  Q = tf.concat([Q, v], axis=1)

  # diagonals of the tridiagonal matrix
  beta = tf.reshape(tf.constant(0, dtype=tf.float32), [1,])
  alpha = tf.reshape(tf.constant(0, dtype=tf.float32), [1,])

  for i in range(k):
    v = vector_prod_fn(tf.reshape(Q[:, i+1], [n, 1])) - tf.scalar_mul(scalar, tf.reshape(Q[:, i+1], [n, 1]))
    v = tf.reshape(v, [n,])
    curr_alpha = tf.reshape(tf.reduce_sum(tf.multiply(v, Q[:, i+1])), [1,])
    alpha = tf.concat([alpha, curr_alpha], axis=0)
    v = v-beta[-1]*Q[:,i]-alpha[-1]*Q[:,i+1]
    curr_beta = tf.reshape(tf.norm(v), [1,])
    beta = tf.concat([beta, curr_beta], axis=0)
    curr_norm = tf.reshape(v/(beta[-1]+1e-8), [n,1])
    Q = tf.concat([Q, curr_norm], axis=1)
  
  alpha = tf.slice(alpha, begin=[1], size=[-1])
  return alpha, beta, Q

# def lanczos_decomp(vector_prod_fn, scalar, n, k):
#   """Function that performs the Lanczos algorithm on a matrix.

#   Args:
#     vector_prod_fn: function which returns product H*x, where H is a matrix for
#       which we computing eigenvector.
#     scalar: quantity to scale the product returned by vector_prod_fn by
#     n: dimensionality of matrix H
#     k: number of iterations and dimensionality of the tridiagonal matrix to
#       return
  
#   Returns:
#     d: vector of diagonal elements of T
#     e: vector of off-diagonal elements of T
#     V: orthonormal basis matrix for the Krylov subspace
#   """
#   # Choose random initial vector of dimentionality n
#   v = tf.nn.l2_normalize(tf.random_uniform([n, 1]))

#   # Compute first w
#   w = vector_prod_fn(v) - scalar * v
#   alpha = tf.matmul(tf.transpose(w), v)
#   w -= alpha * v

#   # Set T and V
#   d = alpha
#   e = alpha
#   V = v

#   # Compute rest of basis vectors
#   for i in range(1, k):
#     prev_v = v
#     beta = tf.norm(w)
#     if beta != 0:
#       v = w / beta
#     else:
#       v = tf.nn.l2_normalize(tf.random_uniform([n, 1]))
#     w = vector_prod_fn(v) - scalar * v
#     alpha = tf.matmul(tf.transpose(w), v)
#     w = w - alpha * v - beta * prev_v
#     beta = tf.reshape(beta, [1, 1])

#     # Set T and V
#     d = tf.concat([d, alpha], axis=0)
#     V = tf.concat([V, v], axis=1)
#     e = tf.concat([e, beta], axis=0)
  
#   # Compute beta and set T
#   beta = tf.norm(w)
#   beta = tf.reshape(beta, [1, 1])
#   e = tf.concat([e, beta], axis=0)

#   # Squeeze dims
#   d = tf.squeeze(d)
#   e = tf.squeeze(e)

#   e = tf.slice(e, begin=[1], size=[-1])

#   return d, e, V

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
