import os
import pathlib
import pickle

import numpy as np
import tensorflow as tf

import social_visualize
from model import Model
from social_utils import SocialDataLoader


def sample_and_visualize(args):
    # Define the path for the config file for saved args
    save_location = os.path.join(args.train_logs, 'save', str(args.test_dataset))
    results_pkl = os.path.join(save_location, 'results.pkl')

    # Get the checkpoint state for the model
    model_directory = os.path.join(args.train_logs, 'model', str(args.test_dataset))

    plot_location = os.path.join(args.train_logs, 'plot', str(args.test_dataset))
    path = pathlib.Path(plot_location)
    path.mkdir(parents=True, exist_ok=True)

    if not args.viz_only:
        sample(args, save_location, model_directory)

    if args.viz or args.viz_only:
        # creating visualization
        social_visualize.visualize(results_pkl, plot_location)


def sample(args, save_location, model_directory):
    results_pkl = os.path.join(save_location, 'results.pkl')
    with open(os.path.join(save_location, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # Create a SocialModel object with the saved_args and infer set to true
    model = Model(saved_args, True)
    # Initialize a TensorFlow session
    config = tf.ConfigProto(log_device_placement=True)  # Showing which device is allocated (in case of multiple GPUs)
    config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Allocating 40% of memory in each GPU
    sess = tf.InteractiveSession(config=config)

    ckpt = tf.train.get_checkpoint_state(model_directory)

    # Initialize a saver
    saver = tf.train.Saver()
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
    final_displacement_error = []

    # For each batch
    for b in range(data_loader.num_batches):
        # Get the source, target and dataset data for the next batch
        x, y, d = data_loader.next_batch(randomUpdate=False)

        # Batch size is 1
        x_batch, y_batch = x[0], y[0]

        true_traj = np.concatenate((x_batch, y_batch[-args.pred_length:]), axis=0)
        # complete_traj is an array of shape ( obs_length + pred_length ) x maxNumPeds x 3
        complete_traj = model.sample(sess, x_batch, true_traj, args.pred_length)
        total_error += model.get_mean_error(complete_traj, true_traj, args.obs_length, saved_args.maxNumPeds)

        final_error = model.get_final_displacement_error(complete_traj, true_traj, saved_args.maxNumPeds)
        if final_error is not None:
            final_displacement_error.append(final_error)

        print("Processed trajectory number : ", b, "out of ", data_loader.num_batches, " trajectories")
        results.append((true_traj, complete_traj, args.obs_length))

    print("Saving results")
    with open(results_pkl, 'wb') as f:
        pickle.dump(results, f)

    # Print the mean error across all the batches
    print("Total mean error of the model is {:.3f}".format(total_error / data_loader.num_batches))
    print("Total final error of the model is {:.3f}".format(np.mean(final_displacement_error)))
