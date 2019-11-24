import os
import pathlib
import pickle

import numpy as np
import tensorflow as tf

import social_visualize
from model import Model
from social_utils import SocialDataLoader


def get_mean_error(predicted_traj, true_traj, observed_length, maxNumPeds):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    '''
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
            if true_pos[j, 0] == 0:
                # Non-existent ped
                continue
            elif pred_pos[j, 0] == 0:
                # Ped comes in the prediction time. Not seen in observed part
                continue
            else:
                if true_pos[j, 1] > 1 or true_pos[j, 1] < 0:
                    continue
                elif true_pos[j, 2] > 1 or true_pos[j, 2] < 0:
                    continue

                timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                counter += 1

        if counter != 0:
            error[i - observed_length] = timestep_error / counter

        # The euclidean distance is the error
        # error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)


def sample_and_visualize(args):
    save_location = os.path.join(args.train_logs, 'save', str(args.test_dataset))
    results_pkl = os.path.join(save_location, 'results.pkl')

    plot_location = os.path.join(args.train_logs, 'plot', str(args.test_dataset))
    path = pathlib.Path(plot_location)
    path.mkdir(parents=True, exist_ok=True)

    if not args.viz_only:
        sample(args)

    if args.viz or args.viz_only:
        # creating visualization
        social_visualize.visualize(results_pkl, plot_location)


def sample(args):
    # Define the path for the config file for saved args
    save_location = os.path.join(args.train_logs, 'save', str(args.test_dataset))

    with open(os.path.join(save_location, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    results_pkl = os.path.join(save_location, 'results.pkl')

    # https://stackoverflow.com/a/47087740/2049763
    tf.reset_default_graph()

    # Create a SocialModel object with the saved_args and infer set to true
    model = Model(saved_args, True)
    # Initialize a TensorFlow session
    config = tf.ConfigProto(log_device_placement=True)  # Showing which device is allocated (in case of multiple GPUs)
    config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Allocating 40% of memory in each GPU
    sess = tf.InteractiveSession(config=config)

    # Initialize a saver
    saver = tf.train.Saver()
    # Get the checkpoint state for the model
    model_directory = os.path.join(args.train_logs, 'model', str(args.test_dataset))
    ckpt = tf.train.get_checkpoint_state(model_directory)

    # Restore the model at the checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Dataset to get data from
    dataset = [args.test_dataset]

    # Create a SocialDataLoader object with batch_size 1 and seq_length equal to observed_length + pred_length
    data_loader = SocialDataLoader(1, args.obs_length, args.pred_length, saved_args.maxNumPeds, dataset,
                                   forcePreProcess=True, infer=True)

    # Reset all pointers of the data_loader
    data_loader.reset_batch_pointer()

    results = []

    # Variable to maintain total error
    total_error = 0

    # For each batch
    for b in range(data_loader.num_batches):
        # Get the source, target and dataset data for the next batch
        x, y, d = data_loader.next_batch(randomUpdate=False)

        # Batch size is 1
        x_batch, y_batch = x[0], y[0]

        true_traj = np.concatenate((x_batch, y_batch[-args.pred_length:]), axis=0)
        # complete_traj is an array of shape ( obs_length + pred_length ) x maxNumPeds x 3
        complete_traj = model.sample(sess, x_batch, true_traj, args.pred_length)

        # ipdb.set_trace()
        # complete_traj is an array of shape (obs_length+pred_length) x maxNumPeds x 3
        total_error += get_mean_error(complete_traj, true_traj, args.obs_length, saved_args.maxNumPeds)

        print("Processed trajectory number : ", b, "out of ", data_loader.num_batches, " trajectories")
        results.append((true_traj, complete_traj, args.obs_length))

    print("Saving results")
    with open(results_pkl, 'wb') as f:
        pickle.dump(results, f)

    # Print the mean error across all the batches
    print("Total mean error of the model is {}".format(total_error / data_loader.num_batches))
