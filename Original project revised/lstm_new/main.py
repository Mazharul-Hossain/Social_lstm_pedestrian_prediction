# conda activate tensorflow_gpuenv
# cd "D:\UofMemphis\Coding\notebooks\Social_lstm_pedestrian_prediction\Original project revised\lstm_new"
# D:
# to train just run:
# $ python main.py
# to test just run:
# $ python main.py -test or with options -viz --obs_length=36 --pred_length=6 --test_dataset=0
#
# tensorboard --logdir="D:\UofMemphis\Coding\notebooks\Social_lstm_pedestrian_prediction\Original project revised"
#
# [Introduction to Recurrent Networks in TensorFlow]
# (https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/)
# [TensorBoard Tutorial](https://www.datacamp.com/community/tutorials/tensorboard-tutorial)
# https://www.easy-tensorflow.com/tf-tutorials/basics/introduction-to-tensorboard
# https://www.oreilly.com/ideas/introduction-to-lstms-with-tensorflow

import argparse
import os
import pickle
import sys
import time
import traceback

import numpy as np
import tensorflow as tf

import sample
from model import Model
from social_utils import SocialDataLoader as DataLoader


def main(args):
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Type of recurrent unit parameter
    # Model currently not used. Only LSTM implemented
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')

    # Length of sequence to be considered parameter
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=9,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=1,
                        help='Predicted length of the trajectory must be less or equal to obs_length')
    parser.add_argument('--maxNumPeds', type=int, default=70,
                        help='Maximum number of pedestrian')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # Gradient value at which it should be clipped
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.004,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.98,
                        help='decay rate for rmsprop')
    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter; same as rnn_size, here rnn_size/2
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Embedding dimension for the spatial coordinates')

    # The dataset index to be left out in training; The test dataset
    parser.add_argument('--test_dataset', type=int, default=3,
                        help='Dataset to be tested on')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.05,
                        help='L2 regularization parameter')
    # optimizer for training
    parser.add_argument('--optimizer', type=str, default="RMSprop",
                        help='Training optimizer parameter RMSprop or AdamOpt')

    # flag to run testing
    parser.add_argument('-test', action='store_true',
                        help='Select testing vs training')
    # flag to run graphs
    parser.add_argument('-viz', action='store_true',
                        help='Visualize testing result')
    parser.add_argument('-viz_only', action='store_true',
                        help='Visualize testing result')
    args = parser.parse_args()

    args.train_logs = os.path.join('..', 'train_logs', 'lstm_new_02')  # + datetime.now().strftime("%Y%m%d_%H%M%S")
    # import shutil
    # if os.path.isdir(os.path.join('..', 'train_logs')):
    #     shutil.rmtree(os.path.join('..', 'train_logs'))

    # for dataset in range(5):
    #     try:
    #         args.test_dataset = dataset
    #
    #         # https://stackoverflow.com/a/47087740/2049763
    #         tf.reset_default_graph()
    #         train(args)
    #
    #         tf.reset_default_graph()
    #         sample.sample_and_visualize(args)
    #     except:
    #         traceback.print_exception(*sys.exc_info())

    # Training phase
    if not args.test:
        try:
            tf.reset_default_graph()
            train(args)
        except Exception as e:
            print('This code NOT be running because of', e)
            traceback.print_exception(*sys.exc_info())

    # testing phase
    if args.test:
        try:
            print("Testing is starting !")
            tf.reset_default_graph()
            sample.sample_and_visualize(args)
        except Exception as e:
            print('This code NOT be running because of', e)
            traceback.print_exception(*sys.exc_info())


def train(args):
    datasets = list(range(5))
    # Remove the leaveDataset from data_sets
    datasets.remove(args.test_dataset)

    # Create the data loader object. This object would preprocess the data in terms of
    # batches each of size args.batch_size, of length args.seq_length
    data_loader = DataLoader(args.batch_size, args.obs_length, args.obs_length,
                             maxNumPeds=args.maxNumPeds, datasets=datasets, forcePreProcess=True)

    # https://stackoverflow.com/a/41146954/2049763
    import pathlib
    # Log directory
    log_directory = os.path.join(args.train_logs, 'log', str(args.test_dataset))
    path = pathlib.Path(log_directory)
    path.mkdir(parents=True, exist_ok=True)
    # Logging files
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Save directory
    save_directory = os.path.join(args.train_logs, 'save', str(args.test_dataset))
    path = pathlib.Path(save_directory)
    path.mkdir(parents=True, exist_ok=True)
    # model directory
    model_directory = os.path.join(args.train_logs, 'model', str(args.test_dataset))
    path = pathlib.Path(model_directory)
    path.mkdir(parents=True, exist_ok=True)

    # Save the arguments int the config file
    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    checkpoint_path = os.path.join(model_directory, 'model.ckpt')

    # Create a Vanilla LSTM model with the arguments
    model = Model(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Showing which device is allocated (in case of multiple GPUs)
    config = tf.ConfigProto(log_device_placement=True)
    # Allocating 70% of memory in each GPU with 0.5
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # Initialize a TensorFlow session
    with tf.Session() as sess:
        # Summaries need to be displayed
        # Whenever you need to record the loss, feed the mean loss to this placeholder
        tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
        # Create a scalar summary object for the loss so it can be displayed
        tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)
        # Whenever you need to record the loss, feed the mean loss to this placeholder
        tf_embedding_w_ph = tf.placeholder(tf.float32, shape=None, name='embedding_w_summary')
        tf_embedding_w_summary = tf.summary.scalar('embedding_w', tf_embedding_w_ph)
        # Whenever you need to record the loss, feed the mean loss to this placeholder
        tf_output_w_ph = tf.placeholder(tf.float32, shape=None, name='output_w_summary')
        tf_output_w_summary = tf.summary.scalar('output_w', tf_output_w_ph)
        # Whenever you need to record the loss, feed the mean loss to this placeholder
        tf_lr_ph = tf.placeholder(tf.float32, shape=None, name='learning_rate_summary')
        tf_lr_ph_summary = tf.summary.scalar('learning_rate', tf_lr_ph)

        # Whenever you need to record the loss, feed the mean loss to this placeholder
        # tf_val_loss_ph = tf.placeholder(tf.float32, shape=None, name='val_loss_summary')
        # tf_val_loss_summary = tf.summary.scalar('val_loss', tf_val_loss_ph)
        # Whenever you need to record the loss, feed the mean loss to this placeholder
        tf_val_error_ph = tf.placeholder(tf.float32, shape=None, name='val_error_summary')
        tf_val_error_summary = tf.summary.scalar('val_error', tf_val_error_ph)

        # https://stackoverflow.com/a/40148954/2049763
        train_writer = tf.summary.FileWriter(model_directory, sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(model_directory, 'eval'))

        # Initialize all the variables in the graph
        sess.run(tf.global_variables_initializer())
        # Add all the variables to the list of variables to be saved
        saver = tf.train.Saver(tf.global_variables())

        best_val_loss = 100
        best_epoch = 0
        print("**** Training is starting !")
        # For each epoch
        for epoch in range(args.num_epochs):
            # Assign the learning rate (decayed acc. to the epoch number)
            # learning_rate = args.learning_rate * (args.decay_rate ** epoch)
            # Reset the pointers in the data loader object
            data_loader.reset_batch_pointer()

            loss_per_epoch = []
            total_steps = args.num_epochs * data_loader.num_batches
            # For each batch in this epoch
            for batch in range(data_loader.num_batches):
                # Tic
                start = time.time()
                # Get the source and target data of the current batch
                # x has the source data, y has the target data
                x, y, d = data_loader.next_batch(randomUpdate=True)

                # variable to store the loss for this batch
                loss_per_batch = []
                gradients = {}

                # For each sequence in the batch
                for sequence in range(data_loader.batch_size):
                    # x_batch, y_batch and d_batch contains the source, target and dataset index data for
                    # seq_length long consecutive frames in the dataset
                    # x_batch, y_batch would be numpy arrays of size seq_length x maxNumPeds x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is extracted
                    x_batch, y_batch = x[sequence], y[sequence]

                    # Feed the source, target data
                    feed = {model.input_data: x_batch, model.target_data: y_batch,
                            model.keep_prob: args.keep_prob, model.lr: args.learning_rate,
                            model.training_epoch: epoch}

                    # train_loss, gradient, _, lr = sess.run(
                    #     [model.cost, model.clipped_gradients, model.train_op, model.final_lr], feed)
                    train_loss, gradient, lr = sess.run(
                        [model.cost, model.clipped_gradients, model.final_lr], feed)

                    vars = tf.trainable_variables()
                    # vars_vals = sess.run(vars)
                    for var, val in zip(vars, gradient):
                        # print(var.name, np.shape(val))
                        if var.name in gradients:
                            gradients[var.name].append(val)
                        else:
                            gradients[var.name] = []

                    if not np.isnan(train_loss):
                        loss_per_batch.append(train_loss)
                        # break
                    else:
                        print("epoch#{} batch#{} sequence#{} train_loss is NaN".format(epoch + 1, batch + 1, sequence))

                avg_loss_per_batch = np.mean(loss_per_batch)
                loss_per_epoch.append(avg_loss_per_batch)

                my_global_step = (epoch + 1) * data_loader.num_batches + batch
                # Print epoch, batch, loss and time taken
                print(
                    "## {}/{} (Epoch: {}/{}), (Batch: {}/{}) train_loss = {:.3f}, time/batch = {:.3f}".format(
                        my_global_step, total_steps, epoch + 1, args.num_epochs, batch + 1, data_loader.num_batches,
                        avg_loss_per_batch, time.time() - start))
                # Save the model if the current epoch and batch number match the frequency
                if my_global_step % args.save_every == 0 and (my_global_step > 0):
                    saver.save(sess, checkpoint_path, global_step=my_global_step)
                    print("model saved to {}".format(checkpoint_path))
                # break

                # ######## **** Mini batch optimization starts **** ########
                # vars_vals = sess.run(vars)
                feed, grad_ph_key = {model.lr: args.learning_rate,
                                     model.training_epoch: epoch}, 0
                for var, val_ in gradients.items():
                    val = np.mean(val_, axis=0)
                    feed[model.grad_placeholders[grad_ph_key]] = val
                    grad_ph_key += 1

                    # print(var, np.shape(val_), np.shape(val))
                    val = np.sum(np.absolute(val))
                    # print("var: {}, value: {}".format(var, val))
                    if 'embedding_w' in var:
                        embedding_w_summary = val
                    elif 'output_w' in var:
                        output_w_summary = val

                _ = sess.run([model.train_op_2], feed)
                # ######## **** Mini batch optimization ends **** ########
            # ######## **** Training batch iteration ends **** ########
            avg_loss_per_epoch = np.mean(loss_per_epoch)
            print('# (Epoch {}/{}), Learning rate = {} Training Loss = {:.3f}'.format(epoch + 1, args.num_epochs, lr,
                                                                                      avg_loss_per_epoch))
            log_file_curve.write(str(epoch) + ',' + str(avg_loss_per_epoch) + ',')

            # Execute the summaries defined above
            training_loss_summary, embedding_w_summary, output_w_summary, lr_ph_summary = sess.run(
                [tf_loss_summary, tf_embedding_w_summary, tf_output_w_summary, tf_lr_ph_summary],
                feed_dict={tf_loss_ph: avg_loss_per_epoch,
                           tf_embedding_w_ph: embedding_w_summary,
                           tf_output_w_ph: output_w_summary,
                           tf_lr_ph: lr})
            training_summaries = tf.summary.merge(
                [training_loss_summary, embedding_w_summary, output_w_summary, lr_ph_summary])
            training_summaries_tensor = sess.run(training_summaries)

            # ######## **** Validation starts **** ########
            data_loader.reset_batch_pointer(valid=True)
            val_loss_per_epoch, val_error_per_epoch = [], []

            for batch in range(data_loader.num_batches):

                # Get the source, target and dataset data for the next batch x, y are input and target data which are
                # lists containing numpy arrays of size seq_length x maxNumPeds x 3 d is the list of dataset indices
                # from which each batch is generated (used to differentiate between datasets)
                x, y, d = data_loader.next_batch(valid=True, randomUpdate=True)

                # variable to store the loss for this batch
                val_loss_per_batch, val_error_per_batch = [], []

                # For each sequence in the batch
                for sequence in range(data_loader.batch_size):
                    # x_batch, y_batch and d_batch contains the source, target and dataset index data for
                    # seq_length long consecutive frames in the dataset
                    # x_batch, y_batch would be numpy arrays of size seq_length x maxNumPeds x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is extracted
                    x_batch, y_batch = x[sequence], y[sequence]

                    # Feed the source, target data
                    feed = {model.input_data: x_batch, model.target_data: y_batch, model.keep_prob: 1.}
                    output, val_loss = sess.run([model.final_result, model.cost], feed)
                    val_loss_per_batch.append(val_loss)
                    val_error_per_batch.append(model.training_mean_error(x_batch, y_batch, output))

                val_loss_per_epoch.append(np.mean(val_loss_per_batch))
                val_error_per_epoch.append(np.mean(val_error_per_batch))

                # break
            avg_val_loss_per_epoch = np.mean(val_loss_per_epoch)
            avg_val_error_per_epoch = np.mean(val_error_per_epoch)

            # Update best validation loss until now
            if avg_val_loss_per_epoch < best_val_loss:
                best_val_loss = avg_val_loss_per_epoch
                best_epoch = epoch

            print('# (Epoch {}/{}), Validation loss = {:.3f}, error = {:.3f}'.format(epoch + 1, args.num_epochs,
                                                                                     avg_val_loss_per_epoch,
                                                                                     avg_val_error_per_epoch))
            log_file_curve.write(str(avg_val_loss_per_epoch) + '\n')

            # Execute the summaries defined above
            val_loss_summary, val_error_summary = sess.run([tf_loss_summary, tf_val_error_summary],
                                                           feed_dict={tf_loss_ph: avg_val_loss_per_epoch,
                                                                      tf_val_error_ph: avg_val_error_per_epoch})

            # Merge all summaries together
            performance_summaries = tf.summary.merge([val_loss_summary, val_error_summary])
            # https://stackoverflow.com/a/51784126/2049763
            performance_summaries_tensor = sess.run(performance_summaries)
            # Write the obtained summaries to the file, so it can be displayed in the TensorBoard
            train_writer.add_summary(training_summaries_tensor, epoch)
            val_writer.add_summary(performance_summaries_tensor, epoch)

        print('Best epoch', best_epoch, 'Best validation loss', best_val_loss)
        log_file.write(str(best_epoch) + ',' + str(best_val_loss))

        my_global_step += 1
        saver.save(sess, checkpoint_path, global_step=my_global_step)
        print("model saved to {}".format(checkpoint_path))

        # CLose logging files
        log_file.close()
        log_file_curve.close()
        train_writer.close()
        val_writer.close()


if __name__ == '__main__':
    tf.app.run(main)
