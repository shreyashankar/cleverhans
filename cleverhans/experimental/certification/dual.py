

class Dual (object):

	def __init__(self, neural_network, dual_file, init_nu, test_input, true_class,
               adv_class, input_minval, input_maxval, epsilon):
    """Initializes dual formulation class.

    Args:
      neural_network: NeuralNetwork object created for
        the network under consideration
			dual_file: Path to file containing dual variables, if the path
				is empty, perform random initialization
				Expects numpy dictionary with
				lambda_pos_0, lambda_pos_1, ..
				lambda_neg_0, lambda_neg_1, ..
				lambda_quad_0, lambda_quad_1, ..
				lambda_lu_0, lambda_lu_1, ..
			init_nu: Value to initialize nu variable with
      test_input: clean example to certify around
      true_class: the class label of the test input
      adv_class: the label that the adversary tried to perturb input to
      input_minval: minimum value of valid input range
      input_maxval: maximum value of valid input range
      epsilon: Size of the perturbation (scaled for [0, 1] input)
    """

		dual_var = self.initialize_dual_var (neural_network, init_dual_file=dual_file, init_nu=init_nu)
    self.set_constraints (neural_network, dual_var, test_input, true_class, 
											 adv_class, input_minval, input_maxval, epsilon)

    # Computing the optimization terms
    self.lambda_pos = [x for x in dual_var['lambda_pos']]
    self.lambda_neg = [x for x in dual_var['lambda_neg']]
    self.lambda_quad = [x for x in dual_var['lambda_quad']]
    self.lambda_lu = [x for x in dual_var['lambda_lu']]
    self.nu = dual_var['nu']
    self.vector_g = None
    self.scalar_f = None
    self.matrix_h = None
    self.matrix_m = None

		# Compute the optimization terms for the projected dual
		projected_var = self.project_dual ()
		self.projected_lambda_pos = [x for x in projected_var['lambda_pos']]
    self.projected_lambda_neg = [x for x in projected_var['lambda_neg']]
    self.projected_lambda_quad = [x for x in projected_var['lambda_quad']]
    self.projected_lambda_lu = [x for x in projected_var['lambda_lu']]
    self.projected_nu = projected_var['nu']
    self.projected_vector_g = None
    self.projected_scalar_f = None
    self.projected_matrix_h = None
    self.projected_matrix_m = None

    # The primal vector in the SDP can be thought of as [layer_1, layer_2..]
    # In this concatenated version, dual_index[i] that marks the start
    # of layer_i
    # This is useful while computing implicit products with matrix H
    self.dual_index = [0]
    for i in range(self.neural_network.num_hidden_layers + 1):
      self.dual_index.append(self.dual_index[-1] +
                             self.neural_network.sizes[i])

	def initialize_dual_var (self, neural_network, init_dual_file=None,
                    random_init_variance=0.01, init_nu=200.0):
		"""Function to initialize the dual variables of the class.

		Args:
			neural_network: Object with the neural net weights, biases
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
			for i in range(0, neural_network.num_hidden_layers + 1):
				initializer = (np.random.uniform(0, random_init_variance, size=(
						neural_network.sizes[i], 1))).astype(np.float32)
				lambda_pos.append(tf.get_variable('lambda_pos_' + str(i),
																					initializer=initializer,
																					dtype=tf.float32))
				initializer = (np.random.uniform(0, random_init_variance, size=(
						neural_network.sizes[i], 1))).astype(np.float32)
				lambda_neg.append(tf.get_variable('lambda_neg_' + str(i),
																					initializer=initializer,
																					dtype=tf.float32))
				initializer = (np.random.uniform(0, random_init_variance, size=(
						neural_network.sizes[i], 1))).astype(np.float32)
				lambda_quad.append(tf.get_variable('lambda_quad_' + str(i),
																					initializer=initializer,
																					dtype=tf.float32))
				initializer = (np.random.uniform(0, random_init_variance, size=(
						neural_network.sizes[i], 1))).astype(np.float32)
				lambda_lu.append(tf.get_variable('lambda_lu_' + str(i),
																				initializer=initializer,
																				dtype=tf.float32))
			nu = tf.get_variable('nu', initializer=init_nu)
			nu = tf.reshape(nu, shape=(1, 1))
		else:
			# Loading from file
			dual_var_init_val = np.load(init_dual_file).item()
			for i in range(0, neural_network.num_hidden_layers + 1):
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
			nu = tf.reshape(nu, shape=(1, 1))
		dual_var = {'lambda_pos': lambda_pos, 'lambda_neg': lambda_neg,
								'lambda_quad': lambda_quad, 'lambda_lu': lambda_lu, 'nu': nu}
		return dual_var
	
	def set_constraints (self, neural_network, dual_var, test_input, true_class, 
											 adv_class, input_minval, input_maxval, epsilon):
		
		self.neural_network = neural_network
    self.test_input = tf.convert_to_tensor(test_input, dtype=tf.float32)
    self.true_class = true_class
    self.adv_class = adv_class
    self.input_minval = tf.convert_to_tensor(input_minval, dtype=tf.float32)
    self.input_maxval = tf.convert_to_tensor(input_maxval, dtype=tf.float32)
    self.epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
    self.final_linear = (self.neural_network.final_weights[adv_class, :]
                         - self.neural_network.final_weights[true_class, :])
    self.final_linear = tf.reshape(self.final_linear,
                                   shape=[tf.size(self.final_linear), 1])
    self.final_constant = (self.neural_network.final_bias[adv_class]
                           - self.neural_network.final_bias[true_class])

    # Computing lower and upper bounds
    # Note that lower and upper are of size neural_network.num_hidden_layers + 1
    self.lower = []
    self.upper = []

    # Initializing at the input layer with \ell_\infty constraints
    self.lower.append(
        tf.maximum(self.test_input - self.epsilon, self.input_minval))
    self.upper.append(
        tf.minimum(self.test_input + self.epsilon, self.input_maxval))
    for i in range(0, self.neural_network.num_hidden_layers):
      current_lower = tf.nn.relu(0.5*(
          self.neural_network.forward_pass(self.lower[i] + self.upper[i], i)
          + self.neural_network.forward_pass(self.lower[i] - self.upper[i], i,
                                        is_abs=True))
                                 + self.neural_network.biases[i])
      current_upper = tf.nn.relu(0.5*(
          self.neural_network.forward_pass(self.lower[i] + self.upper[i], i)
          + self.neural_network.forward_pass(self.upper[i] -self.lower[i], i,
                                        is_abs=True))
                                 + self.neural_network.biases[i])
      self.lower.append(current_lower)
      self.upper.append(current_upper)

	def set_differentiable_objective (self):
    """Function that constructs minimization objective from dual variables."""
    # Checking if graphs are already created
    if self.vector_g is not None:
      return

    # Computing the scalar term
    bias_sum = 0
    for i in range(0, self.nn_params.num_hidden_layers):
      bias_sum = bias_sum + tf.reduce_sum(
          tf.multiply(self.nn_params.biases[i], self.lambda_pos[i+1]))
    lu_sum = 0
    for i in range(0, self.nn_params.num_hidden_layers+1):
      lu_sum = lu_sum + tf.reduce_sum(
          tf.multiply(tf.multiply(self.lower[i], self.upper[i]),
                      self.lambda_lu[i]))

    self.scalar_f = - bias_sum - lu_sum + self.final_constant

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
    self.unconstrained_objective = self.scalar_f + 0.5*self.nu

	def compute_certificate(self):
    """Function to compute the certificate associated with feasible solution."""
    self.set_differentiable_objective()
    self.get_full_psd_matrix()
    # TODO: replace matrix_inverse with functin which uses matrix-vector product
    projected_certificate = (
        self.projected_scalar_f +
        0.5*tf.matmul(tf.matmul(tf.transpose(self.projected_vector_g),
                                tf.matrix_inverse(self.projected_matrix_h)),
                      self.projected_vector_g))
    return projected_certificate

	def project_dual (self):
    """Function that projects the input dual variables onto the feasible set.

    Returns:
      projected_dual: Feasible dual solution corresponding to current dual
      projected_certificate: Objective value of feasible dual
    """
    # TODO: consider whether we can use shallow copy of the lists without
    # using tf.identity
    projected_lambda_pos = [tf.identity(x) for x in self.lambda_pos]
    projected_lambda_neg = [tf.identity(x) for x in self.lambda_neg]
    projected_lambda_quad = [tf.identity(x) for x in self.lambda_quad]
    projected_lambda_lu = [tf.identity(x) for x in self.lambda_lu]
    projected_nu = tf.identity(self.nu)

    # TODO: get rid of the special case for one hidden layer
    # Different projection for 1 hidden layer
    if self.nn_params.num_hidden_layers == 1:
      # Creating equivalent PSD matrix for H by Schur complements
      diag_entries = 0.5*tf.divide(
          tf.square(self.lambda_quad[self.nn_params.num_hidden_layers]),
          (self.lambda_quad[self.nn_params.num_hidden_layers] +
           self.lambda_lu[self.nn_params.num_hidden_layers]))
      # If lambda_quad[i], lambda_lu[i] are 0, entry is NaN currently,
      # but we want to set that to 0
      diag_entries = tf.where(tf.is_nan(diag_entries),
                              tf.zeros_like(diag_entries), diag_entries)
      matrix = (
          tf.matmul(tf.matmul(tf.transpose(
              self.nn_params.weights[self.nn_params.num_hidden_layers-1]),
                              utils.diag(diag_entries)),
                    self.nn_params.weights[self.nn_params.num_hidden_layers-1]))
      new_matrix = utils.diag(
          2*self.lambda_lu[self.nn_params.num_hidden_layers - 1]) - matrix
      # Making symmetric
      new_matrix = 0.5*(new_matrix + tf.transpose(new_matrix))
      eig_vals = tf.self_adjoint_eigvals(new_matrix)
      min_eig = tf.reduce_min(eig_vals)
      # If min_eig is positive, already feasible, so don't add
      # Otherwise add to make PSD [1E-6 is for ensuring strictly PSD (useful
      # while inverting)
      projected_lambda_lu[0] = (projected_lambda_lu[0] +
                                0.5*tf.maximum(-min_eig, 0) + 1E-6)

    else:
      # Minimum eigen value of H
      # TODO: Write this in terms of matrix multiply
      # matrix H is a submatrix of M, thus we just need to extend existing code
      # for computing matrix-vector product (see get_psd_product function).
      # Then use the same trick to compute smallest eigenvalue.
      eig_vals = tf.self_adjoint_eigvals(self.matrix_h)
      min_eig = tf.reduce_min(eig_vals)

      for i in range(self.nn_params.num_hidden_layers+1):
        # Since lambda_lu appears only in diagonal terms, can subtract to
        # make PSD and feasible
        projected_lambda_lu[i] = (projected_lambda_lu[i] +
                                  0.5*tf.maximum(-min_eig, 0) + 1E-6)
        # Adjusting lambda_neg wherever possible so that lambda_neg + lambda_lu
        # remains close to unchanged
        # projected_lambda_neg[i] = tf.maximum(0.0, projected_lambda_neg[i] +
        #                                     (0.5*min_eig - 1E-6)*
        #                                     (self.lower[i] + self.upper[i]))

    projected_dual_var = {'lambda_pos': projected_lambda_pos,
                          'lambda_neg': projected_lambda_neg,
                          'lambda_lu': projected_lambda_lu,
                          'lambda_quad': projected_lambda_quad,
                          'nu': projected_nu}
		return project_dual_var 
	
	def get_full_psd_matrix(self):
    """Function that retuns the tf graph corresponding to the entire matrix M.


    Returns:
      matrix_h: unrolled version of tf matrix corresponding to H
      matrix_m: unrolled tf matrix corresponding to M
    """
    # Computing the matrix term
    h_columns = []
    for i in range(self.nn_params.num_hidden_layers + 1):
      current_col_elems = []
      for j in range(i):
        current_col_elems.append(tf.zeros([self.nn_params.sizes[j],
                                           self.nn_params.sizes[i]]))

    # For the first layer, there is no relu constraint
      if i == 0:
        current_col_elems.append(utils.diag(self.lambda_lu[i]))
      else:
        current_col_elems.append(utils.diag(self.lambda_lu[i] +
                                            self.lambda_quad[i]))
      if i < self.nn_params.num_hidden_layers:
        current_col_elems.append((
            (tf.matmul(utils.diag(-1*self.lambda_quad[i+1]),
                       self.nn_params.weights[i]))))
      for j in range(i + 2, self.nn_params.num_hidden_layers + 1):
        current_col_elems.append(tf.zeros([self.nn_params.sizes[j],
                                           self.nn_params.sizes[i]]))
      current_column = tf.concat(current_col_elems, 0)
      h_columns.append(current_column)

    self.matrix_h = tf.concat(h_columns, 1)
    self.set_differentiable_objective()
    self.matrix_h = (self.matrix_h + tf.transpose(self.matrix_h))

    self.matrix_m = tf.concat([tf.concat([self.nu, tf.transpose(self.vector_g)],
                                         axis=1),
                               tf.concat([self.vector_g, self.matrix_h],
                                         axis=1)], axis=0)
    return self.matrix_h, self.matrix_m