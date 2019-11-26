'''
Script to help visualize the results of the trained model

Author : Anirudh Vemula
Date : 10th November 2016
'''
import os
import pathlib
import pickle

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn

import model


# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c


def plot_trajectories(true_trajs, pred_trajs, obs_length, name, save_location):
    """
    Function that plots the true trajectories and the
    trajectories predicted by the model alongside
    params:
    true_trajs : numpy matrix with points of the true trajectories
    pred_trajs : numpy matrix with points of the predicted trajectories
    Both parameters are of shape traj_length x maxNumPeds x 3
    obs_length : Length of observed trajectory
    name: Name of the plot
    """
    traj_length, maxNumPeds, _ = true_trajs.shape

    # Initialize figure
    # https://towardsdatascience.com/matplotlib-seaborn-basics-2bd7b66dbee2
    # conda install seaborn
    seaborn.set(palette="pastel")
    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.get_position()
    traj_data = {}

    # For each frame/each point in all trajectories
    def animate(i):
        print("#### animation is called: {} \t ####".format(i))
        if i == 0:
            traj_data.clear()

        ax.clear()
        seaborn.set(rc={'axes.facecolor': 'lightgrey', 'figure.facecolor': 'lightgrey', 'figure.edgecolor': 'black',
                        'axes.grid': True})
        ax.set_title(name, fontsize=12)
        ax.set_xlim((0.2, 0.8))
        ax.set_ylim((0.3, 0.7))

        pred_pos = pred_trajs[i, :]
        true_pos = true_trajs[i, :]

        # For each pedestrian
        for j in range(maxNumPeds):
            if model.check_true_pedestrian(true_pos[j, :], pred_pos[j, :]):
                continue

            if (j not in traj_data) and i < obs_length:
                traj_data[j] = [[], []]

            if j in traj_data:
                traj_data[j][0].append(true_pos[j, 1:3])
                if i >= obs_length - 1:
                    traj_data[j][1].append(pred_pos[j, 1:3])
                    # print(i, j, "\n", traj_data[j][0], "\n", traj_data[j][1])

        for j in traj_data:
            # c = np.random.rand(3, 1)
            true_traj_ped = traj_data[j][0]  # List of [x,y] elements
            pred_traj_ped = traj_data[j][1]

            # if len(pred_traj_ped) > 0:
            #     print(i, j, true_traj_ped[obs_length - 1:], pred_traj_ped)

            adding_factor = 1
            true_x = [(p[0] + adding_factor) / 2 for p in true_traj_ped]
            true_y = [(p[1] + adding_factor) / 2 for p in true_traj_ped]
            pred_x = [(p[0] + adding_factor) / 2 for p in pred_traj_ped]
            pred_y = [(p[1] + adding_factor) / 2 for p in pred_traj_ped]

            if len(pred_traj_ped) > 0:
                print("**** frame#{} ped#{} : ({}, {}) ****\ntrue_x:{}\npred_x:{}\ntrue_y:{}\npred_y:{}"
                      .format(i, j, len(true_x), len(pred_x),
                              ['%.3f' % elem for elem in true_x[-len(pred_x):]],
                              ['%.3f' % elem for elem in pred_x],
                              ['%.3f' % elem for elem in true_y[-len(pred_y):]],
                              ['%.3f' % elem for elem in pred_y]))

            if i >= obs_length and len(pred_traj_ped) <= 1:
                continue

            # color='g', linestyle='solid'
            color_value = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            ax.plot(true_x, true_y, color=color_value[j % len(color_value)], linestyle='dashed', marker='o',
                    label="true_ped#{}".format(j))
            ax.plot(pred_x, pred_y, linestyle='dotted', marker='x', label="pred_ped#{}".format(j))
        ax.set_xlabel('time_step {}'.format(i), fontsize=10)
        # Shrink current axis by 20%
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

        # Put a legend to the right of the current axis
        # https://stackoverflow.com/a/4701285/2049763
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return (ax,)

    # call the animator
    # print(traj_length)
    anim = animation.FuncAnimation(fig, animate, frames=traj_length, interval=500, repeat=True)
    plt.show()

    # save the animation as mp4 video file
    # writer = animation.PillowWriter(fps=2)  # imagemagick
    # writer = animation.writers['ffmpeg'](fps=2, metadata=dict(artist='Me'), bitrate=1800)
    # anim.save(os.path.join(save_location, 'plot_' + name + '.gif'), writer=writer)
    # http://docs.wand-py.org/en/0.4.1/guide/install.html
    # $ sudo apt-get install libmagickwand-dev libmagickcore5-extra
    # pip install Wand
    # anim.save(os.path.join(save_location, 'plot_' + name + '.gif'), writer='imagemagick')


def visualize(results_pkl, save_location=None):
    if save_location is None:
        save_location = 'plot/'

    f = open(results_pkl, 'rb')
    # f = open('save/3/social_results.pkl', 'rb')
    # was f = open('save/social_results.pkl', 'rb')
    results = pickle.load(f)

    zeros = '000'
    for i in range(len(results)):
        digits = len(str(abs(int(i))))
        name = 'sequence_' + zeros[:-digits] + str(i)
        print("\n{} is starting".format(name))
        plot_trajectories(results[i][0], results[i][1], results[i][2], name, save_location)


# https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data
# http://www.vision.ee.ethz.ch/datasets/index.en.html
def video_trajectories(results_pkl, video_input_path, video_output_path=None):
    """
    Function that plots the true trajectories and the
    trajectories predicted by the model alongside
    params:
    results_pkl : numpy matrix with points of the true trajectories
    video_input_path : numpy matrix with points of the predicted trajectories
    video_output_path : Length of observed trajectory
    """

    f = open(results_pkl, 'rb')
    results = pickle.load(f)

    vid = cv2.VideoCapture(video_input_path)
    if not vid.isOpened():
        raise IOError("Couldn't open the video")

    video_FourCC = cv2.VideoWriter_fourcc(*"MJPG")
    video_fps = 2.5

    width, height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_size = (width, height)
    font = cv2.FONT_HERSHEY_PLAIN

    if video_output_path is None:
        video_output_path = 'video'

    path = pathlib.Path(video_output_path)
    path.mkdir(parents=True, exist_ok=True)
    video_output_path = os.path.join(video_output_path, os.path.basename(video_input_path))

    print("Saving video to {} \nWith video_FourCC: {}, video_fps: {}, video_size: {}".format(
        video_output_path, video_FourCC, video_fps, video_size))
    out = cv2.VideoWriter(video_output_path, video_FourCC, video_fps, video_size, True)

    mili_second_step = 0
    for result in results:
        true_trajs, pred_trajs, obs_length = result[0], result[1], result[2]

        traj_length, maxNumPeds, _ = true_trajs.shape
        # For each frame/each point in all trajectories
        for i in range(traj_length):
            return_value, frame = vid.read()
            if return_value != True:
                break

            pred_pos = pred_trajs[i, :]
            true_pos = true_trajs[i, :]

            # For each pedestrian
            for j in range(maxNumPeds):
                if model.check_true_pedestrian(true_pos[j, :], pred_pos[j, :]):
                    continue
                true_traj_ped = true_pos[j, :3]
                true_x = int(-1 * true_traj_ped[1] * width + width / 2)
                true_y = int(-1 * true_traj_ped[2] * height + height / 2)

                cv2.putText(frame, str(true_traj_ped[0]), (true_x, true_y), font, 1, (200, 255, 155), 2,
                            cv2.LINE_AA)
                cv2.circle(frame, (true_x, true_y), 15, (200, 255, 155), 3)

                if i >= obs_length:
                    pred_traj_ped = pred_pos[j, :3]

                    pred_x = int(pred_traj_ped[1] * width + width / 2)
                    pred_y = int(pred_traj_ped[2] * height + height / 2)

                    cv2.circle(frame, (pred_x, pred_y), 15, (0, 0, 255), 3)
            out.write(frame)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", frame)

            mili_second_step += 0.4
            vid.set(cv2.CAP_PROP_POS_MSEC, mili_second_step * 100)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # do a bit of cleanup
    print("[INFO] Cleaning up...")
    vid.release()
    out.release()
    cv2.destroyAllWindows()


# results_pkl = "..\\train_logs\\lstm_new\\save\\0\\results.pkl"
# video_input_path = "..\\data\\ucy\\zara\\zara01\\crowds_zara01.avi"
# video_output_path = "..\\train_logs\\lstm_new\\video\\0\\"
# video_trajectories(results_pkl, video_input_path, video_output_path)
# save_location = "..\\train_logs\\lstm_new\\plot\\0\\"

results_pkl = "..\\train_logs\\social_lstm\\save\\3\\social_results.pkl"
save_location = "..\\train_logs\\social_lstm\\plot\\3\\"
visualize(results_pkl, save_location)
