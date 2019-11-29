"""
Social LSTM model implementation using Tensorflow
Social LSTM Paper: http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf

Author : Anirudh Vemula
Date: 10th October 2016
"""

import numpy as np
import tensorflow as tf
from tensorflow.nn import rnn_cell


# The Vanilla LSTM model
class Model:

    def __init__(self, args, infer=False):
        """
        Initialisation function for the class Model.
        Params:
        args: Contains arguments required for the Model creation
        """

        # If sampling new trajectories, then infer mode
        if infer:
            # Infer one position at a time
            args.batch_size = 1
            args.obs_length = 1
            args.pred_length = 1

        # Store the arguments
        self.args = args

        # placeholders for the input data and the target data
        # A sequence contains an ordered set of consecutive frames
        # Each frame can contain a maximum of 'args.maxNumPeds' number of peds
        # For each ped we have their (pedID, x, y) positions as input
        self.input_data = tf.placeholder(tf.float32, [args.obs_length, args.maxNumPeds, 3], name="input_data")
        # target data would be the same format as input_data except with one time-step ahead
        self.target_data = tf.placeholder(tf.float32, [args.obs_length, args.maxNumPeds, 3], name="target_data")
        # Learning rate
        self.lr = tf.placeholder(tf.float32, shape=None, name="learning_rate")
        self.training_epoch = tf.placeholder(tf.float32, shape=None, name="training_epoch")
        # keep prob
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        cells = []
        for _ in range(args.num_layers):
            # Initialize a BasicLSTMCell recurrent unit
            # args.rnn_size contains the dimension of the hidden state of the LSTM
            # cell = rnn_cell.BasicLSTMCell(args.rnn_size, name='basic_lstm_cell', state_is_tuple=False)

            # Construct the basicLSTMCell recurrent unit with a dimension given by args.rnn_size
            if args.model == "lstm":
                with tf.name_scope("LSTM_cell"):
                    cell = rnn_cell.LSTMCell(args.rnn_size, state_is_tuple=False)

            elif args.model == "gru":
                with tf.name_scope("GRU_cell"):
                    cell = rnn_cell.GRUCell(args.rnn_size, state_is_tuple=False)

            if not infer and args.keep_prob < 1:
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            cells.append(cell)

        # Multi-layer RNN construction, if more than one layer
        # cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)

        # Store the recurrent unit
        self.cell = cell

        # Output size is the set of parameters (mu, sigma, corr)
        self.output_size = 5  # 2 mu, 2 sigma and 1 corr

        with tf.name_scope("learning_rate"):
            self.lr = self.lr * (self.args.decay_rate ** self.training_epoch)

        self.define_embedding_and_output_layers(args)

        # Define LSTM states for each pedestrian
        with tf.variable_scope("LSTM_states"):
            self.LSTM_states = tf.zeros([args.maxNumPeds, cell.state_size], name="LSTM_states")
            self.initial_states = tf.split(self.LSTM_states, args.maxNumPeds, 0)
            # https://stackoverflow.com/a/41384913/2049763

        # Define hidden output states for each pedestrian
        with tf.variable_scope("Hidden_states"):
            self.output_states = tf.split(tf.zeros([args.maxNumPeds, cell.output_size]), args.maxNumPeds, 0)

        # List of tensors each of shape args.maxNumPeds x 3 corresponding to each frame in the sequence
        with tf.name_scope("frame_data_tensors"):
            frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.input_data, args.obs_length, 0)]

        with tf.name_scope("frame_target_data_tensors"):
            frame_target_data = [tf.squeeze(target_, [0]) for target_ in
                                 tf.split(self.target_data, args.obs_length, 0)]

        # Cost
        with tf.name_scope("Cost_related_stuff"):
            self.cost = tf.constant(0.0, name="cost")
            self.counter = tf.constant(0.0, name="counter")
            self.increment = tf.constant(1.0, name="increment")

        # Containers to store output distribution parameters
        with tf.name_scope("Distribution_parameters_stuff"):
            self.initial_output = tf.split(tf.zeros([args.maxNumPeds, self.output_size]), args.maxNumPeds, 0)

        # Tensor to represent non-existent ped
        with tf.name_scope("Non_existent_ped_stuff"):
            nonexistent_ped = tf.constant(0.0, name="zero_ped")

        self.final_result = []
        # Iterate over each frame in the sequence
        for seq, frame in enumerate(frame_data):
            # print("Frame number", seq)
            final_result_ped = []
            current_frame_data = frame  # MNP x 3 tensor
            for ped in range(args.maxNumPeds):
                # pedID of the current pedestrian
                pedID = current_frame_data[ped, 0]
                # print("Pedestrian Number", ped)

                with tf.name_scope("extract_input_ped"):
                    # Extract x and y positions of the current ped
                    self.spatial_input = tf.slice(current_frame_data, [ped, 1], [1, 2])  # Tensor of shape (1,2)

                with tf.name_scope("embeddings_operations"):
                    # Embed the spatial input
                    embedded_spatial_input = tf.nn.relu(
                        tf.nn.xw_plus_b(self.spatial_input, self.embedding_w, self.embedding_b))

                # One step of LSTM
                with tf.variable_scope("LSTM") as scope:
                    if seq > 0 or ped > 0:
                        scope.reuse_variables()
                    self.output_states[ped], self.initial_states[ped] = cell(embedded_spatial_input,
                                                                             self.initial_states[ped])

                # Apply the linear layer. Output would be a tensor of shape 1 x output_size
                with tf.name_scope("output_linear_layer"):
                    self.initial_output[ped] = tf.nn.xw_plus_b(self.output_states[ped], self.output_w, self.output_b)

                with tf.name_scope("extract_target_ped"):
                    # Extract x and y coordinates of the target data
                    # x_data and y_data would be tensors of shape 1 x 1
                    [x_data, y_data] = tf.split(tf.slice(frame_target_data[seq], [ped, 1], [1, 2]), 2, 1)
                    target_pedID = frame_target_data[seq][ped, 0]

                with tf.name_scope("get_coef"):
                    # Extract coef from output of the linear output layer
                    [o_mux, o_muy, o_sx, o_sy, o_corr] = self.get_coef(self.initial_output[ped])
                    final_result_ped.append([o_mux, o_muy, o_sx, o_sy, o_corr])

                # Calculate loss for the current ped
                with tf.name_scope("calculate_loss"):
                    lossfunc = self.get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

                # If it is a non-existent ped, it should not contribute to cost
                # If the ped doesn't exist in the next frame, he/she should not contribute to cost as well
                with tf.name_scope("increment_cost"):
                    self.cost = tf.where(
                        tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)),
                        self.cost, tf.add(self.cost, lossfunc))

                    self.counter = tf.where(
                        tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)),
                        self.counter, tf.add(self.counter, self.increment))

            self.final_result.append(tf.stack(final_result_ped))
        # Compute the cost
        with tf.name_scope("mean_cost"):
            # Mean of the cost
            self.cost = tf.div(self.cost, self.counter)

        # Get trainable_variables
        tvars = tf.trainable_variables()

        # L2 loss
        l2 = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars)
        self.cost = self.cost + l2

        # Get the final LSTM states
        self.final_states = tf.concat(self.initial_states, 0)
        # Get the final distribution parameters
        self.final_output = self.initial_output

        # initialize the optimizer with the given learning rate
        if args.optimizer == "RMSprop":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif args.optimizer == "AdamOpt":
            # NOTE: Using RMSprop as suggested by Social LSTM instead of Adam as Graves(2013) does
            optimizer = tf.train.AdamOptimizer(self.lr)

        # How to apply gradient clipping in TensorFlow? https://stackoverflow.com/a/43486487/2049763
        #         # https://stackoverflow.com/a/40540396/2049763
        # TODO: (resolve) We are clipping the gradients as is usually done in LSTM
        # implementations. Social LSTM paper doesn't mention about this at all
        # Calculate gradients of the cost w.r.t all the trainable variables
        self.gradients = tf.gradients(self.cost, tvars)
        # self.gradients = optimizer.compute_gradients(self.cost, var_list=tvars)
        # Clip the gradients if they are larger than the value given in args
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        # Train operator
        self.train_op = optimizer.apply_gradients(zip(self.clipped_gradients, tvars))
        # self.train_op = optimizer.apply_gradients(self.clipped_gradients, var_list=tvars)

        # Merge all summaries
        # merged_summary_op = tf.summary.merge_all()

    def define_embedding_and_output_layers(self, args):
        # Define variables for the spatial coordinates embedding layer
        with tf.variable_scope("coordinate_embedding"):
            self.embedding_w = tf.get_variable("embedding_w", [2, args.embedding_size],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1),
                                               trainable=True)
            self.embedding_b = tf.get_variable("embedding_b", [args.embedding_size],
                                               initializer=tf.zeros_initializer(),
                                               trainable=True)

        # Define variables for the output linear layer
        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w", [args.rnn_size, self.output_size],
                                            initializer=tf.truncated_normal_initializer(stddev=0.01), trainable=True)
            self.output_b = tf.get_variable("output_b", [self.output_size], initializer=tf.zeros_initializer(),
                                            trainable=True)

    @staticmethod
    def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
        """
        Function that implements the PDF of a 2D normal distribution
        params:
        x : input x points
        y : input y points
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        """
        # eq 3 in the paper
        # and eq 24 & 25 in Graves (2013)
        # Calculate (x - mux) and (y-muy)
        normx = tf.subtract(x, mux)
        normy = tf.subtract(y, muy)

        # Calculate sx*sy
        sxsy = tf.multiply(sx, sy)

        # Calculate the exponential factor
        z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2 * tf.div(
            tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
        negRho = 1 - tf.square(rho)
        # Numerator
        result = tf.exp(tf.div(-z, 2 * negRho))

        # Normalization constant
        denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))

        # Final PDF calculation
        result = tf.div(result, denom)
        return result

    def get_lossfunc(self, z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
        """
        Function to calculate given a 2D distribution over x and y, and target data
        of observed x and y points
        params:
        z_mux : mean of the distribution in x
        z_muy : mean of the distribution in y
        z_sx : std dev of the distribution in x
        z_sy : std dev of the distribution in y
        z_rho : Correlation factor of the distribution
        x_data : target x points
        y_data : target y points
        """
        # step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

        # Calculate the PDF of the data w.r.t to the distribution
        result0 = self.tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)

        # For numerical stability purposes
        epsilon = 1e-20

        # Apply the log operation
        result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability

        # Sum up all log probabilities for each data point
        return tf.reduce_sum(result1)

    @staticmethod
    def get_coef(output):
        # eq 20 -> 22 of Graves (2013)

        z = output
        # Split the output into 5 parts corresponding to means, std devs and corr
        z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, 1)

        # The output must be exponentiated for the std devs
        z_sx = tf.exp(z_sx)
        z_sy = tf.exp(z_sy)
        # Tanh applied to keep it in the range [-1, 1]
        z_corr = tf.tanh(z_corr)

        return [z_mux, z_muy, z_sx, z_sy, z_corr]

    @staticmethod
    def sample_gaussian_2d(mux, muy, sx, sy, rho):
        """
        Function to sample a point from a given 2D normal distribution
        params:
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        """
        # Extract mean
        mean = [mux, muy]
        # Extract covariance matrix
        cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
        # Sample a point from the multivariate normal distribution
        x = np.random.multivariate_normal(mean, cov, 1)

        # Modification of SIMONE not to use a random number to decide the future position of the pedestrian:
        return mux, muy  # was return x[0][0], x[0][1]

    def sample(self, sess, obs_traj, true_traj, pred_length=10):
        """
        Function that computes the trajectory predicted based on observed trajectory

        params: obs_traj : a sequence of frames (of length obs_length) of shape is (obs_length x maxNumPeds x 3)
        true_traj : numpy matrix with the points of the true trajectory of shape (obs_length+pred_length) x
                    maxNumPeds x 3
        """
        states = sess.run(self.LSTM_states)
        # print "Fitting"
        # For each frame in the sequence
        for index, frame in enumerate(obs_traj[:-1]):
            data = np.reshape(frame, (1, self.args.maxNumPeds, 3))
            target_data = np.reshape(obs_traj[index + 1], (1, self.args.maxNumPeds, 3))

            feed = {self.input_data: data, self.LSTM_states: states,
                    self.target_data: target_data, self.keep_prob: 1.}

            [states, cost] = sess.run([self.final_states, self.cost], feed)
            # print cost

        ret = obs_traj

        last_frame = obs_traj[-1]

        prev_data = np.reshape(last_frame, (1, self.args.maxNumPeds, 3))

        prev_target_data = np.reshape(true_traj[obs_traj.shape[0]], (1, self.args.maxNumPeds, 3))
        # Prediction
        for t in range(pred_length):
            # print "**** NEW PREDICTION TIME STEP", t, "****"
            feed = {self.input_data: prev_data, self.LSTM_states: states,
                    self.target_data: prev_target_data, self.keep_prob: 1.}
            [output, states, cost] = sess.run([self.final_output, self.final_states, self.cost], feed)
            # print "Cost", cost Output is a list of lists where the inner lists contain matrices of shape 1x5. The
            # outer list contains only one element (since seq_length=1) and the inner list contains maxNumPeds
            # elements output = output[0]
            newpos = np.zeros((1, self.args.maxNumPeds, 3))
            for pedindex, pedoutput in enumerate(output):
                [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(pedoutput[0], 5, 0)
                mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])

                next_x, next_y = self.sample_gaussian_2d(mux, muy, sx, sy, corr)

                newpos[0, pedindex, :] = [prev_data[0, pedindex, 0], next_x, next_y]
            ret = np.vstack((ret, newpos))
            prev_data = newpos

            if t != pred_length - 1:
                prev_target_data = np.reshape(true_traj[obs_traj.shape[0] + t + 1], (1, self.args.maxNumPeds, 3))

        # The returned ret is of shape (obs_length+pred_length) x maxNumPeds x 3
        return ret

    @staticmethod
    def get_mean_error(predicted_traj, true_traj, observed_length, maxNumPeds):
        """
        Function that computes the mean euclidean distance error between the
        predicted and the true trajectory
        params:
        predicted_traj : numpy matrix with the points of the predicted trajectory
        true_traj : numpy matrix with the points of the true trajectory
        observed_length : The length of trajectory observed
        """
        # The data structure to store all errors
        error = np.zeros(len(true_traj) - observed_length)
        # For each point in the predicted part of the trajectory
        for i in range(observed_length, len(true_traj)):
            # The predicted position. This will be a maxNumPeds x 3 matrix
            pred_pos = predicted_traj[i, :]
            # The true position. This will be a maxNumPeds x 3 matrix
            true_pos = true_traj[i, :]
            timestep_error = 0
            counter = 0
            for j in range(maxNumPeds):
                if check_true_pedestrian(true_pos[j, :], pred_pos[j, :]):
                    continue

                # print(true_pos[j], pred_pos[j])
                timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                counter += 1

            if counter != 0:
                error[i - observed_length] = timestep_error / counter

            # The euclidean distance is the error
            # error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

        # Return the mean error
        return np.mean(error)

    @staticmethod
    def training_gaussian_2d(mux, muy, sx, sy, rho):
        """
        Function to sample a point from a given 2D normal distribution
        params:
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        """
        # Extract mean
        mean = [mux, muy]
        # Extract covariance matrix
        cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
        # Sample a point from the multivariate normal distribution
        x = np.random.multivariate_normal(mean, cov, 1)

        # print("training_gaussian_2d: ", mean, x[0])
        return x[0][0], x[0][1]

    def training_mean_error(self, x_batch, y_batch, output):
        # find mean error
        true_traj = np.concatenate((x_batch, y_batch[-self.args.pred_length:]), axis=0)
        # complete_traj is an array of shape ( obs_length + pred_length ) x maxNumPeds x 3
        complete_traj = x_batch
        prev_data = np.reshape(x_batch[-1], (1, self.args.maxNumPeds, 3))
        for frame_index, frame in enumerate(output[-self.args.pred_length:]):
            newpos = np.zeros((1, self.args.maxNumPeds, 3))
            for ped_index, ped_output in enumerate(frame):
                mux, muy, sx, sy, corr = ped_output[:]
                mux, muy, sx, sy, corr = mux[0][0], muy[0][0], sx[0][0], sy[0][0], corr[0][0]
                next_x, next_y = self.training_gaussian_2d(mux, muy, sx, sy, corr)

                newpos[0, ped_index, :] = [prev_data[0, ped_index, 0], next_x, next_y]
            complete_traj = np.vstack((complete_traj, newpos))
            prev_data = newpos
        # complete_traj is an array of shape (obs_length+pred_length) x maxNumPeds x 3
        return self.get_mean_error(complete_traj, true_traj, self.args.obs_length, self.args.maxNumPeds)


def check_true_pedestrian(true_pos, pred_pos):
    # print("check_true_pedestrian: {} {}".format(len(true_pos), len(pred_pos)))
    if true_pos[0] == 0:
        # Non-existent ped
        return True
    elif pred_pos[0] == 0:
        # Ped comes in the prediction time. Not seen in observed part
        return True
    else:
        if true_pos[1] > 1 or true_pos[1] < -1:
            return True
        elif true_pos[2] > 1 or true_pos[2] < -1:
            return True
    return False
