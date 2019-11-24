'''
Script to help visualize the results of the trained model

Author : Anirudh Vemula
Date : 10th November 2016
'''
import os
import pathlib
import pickle

import cv2
import matplotlib.pyplot as plt


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
    plt.figure()

    # Load the background
    # im = plt.imread('plot/plot.png')
    # implot = plt.imshow(im)
    # width = im.shape[0]
    # height = im.shape[1]
    # width = 1
    # height = 1

    traj_data = {}
    # For each frame/each point in all trajectories
    for i in range(traj_length):
        pred_pos = pred_trajs[i, :]
        true_pos = true_trajs[i, :]

        # For each pedestrian
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Not a ped
                continue
            elif pred_pos[j, 0] == 0:
                # Not a ped
                continue
            else:
                # If he is a ped
                if true_pos[j, 1] > 1 or true_pos[j, 1] < 0:
                    continue
                elif true_pos[j, 2] > 1 or true_pos[j, 2] < 0:
                    continue

                if (j not in traj_data) and i < obs_length:
                    traj_data[j] = [[], []]

                if j in traj_data:
                    traj_data[j][0].append(true_pos[j, 1:3])
                    traj_data[j][1].append(pred_pos[j, 1:3])

    for j in traj_data:
        # c = np.random.rand(3, 1)
        true_traj_ped = traj_data[j][0]  # List of [x,y] elements
        pred_traj_ped = traj_data[j][1]

        true_x = [(p[0] + 1) / 2 for p in true_traj_ped]
        true_y = [(p[1] + 1) / 2 for p in true_traj_ped]
        pred_x = [(p[0] + 1) / 2 for p in pred_traj_ped]
        pred_y = [(p[1] + 1) / 2 for p in pred_traj_ped]

        # print(true_x, true_y, pred_x, pred_y)

        plt.plot(true_x, true_y, color='g', linestyle='solid', marker='o')
        plt.plot(pred_x, pred_y, color='b', linestyle='dashed', marker='x')

    # plt.ylim((0, 1))
    # plt.xlim((0, 1))
    plt.title(name)
    # plt.show()

    plt.savefig(os.path.join(save_location, 'plot_' + name + '.png'))
    plt.gcf().clear()
    plt.close()


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
                if true_pos[j, 0] == 0:
                    # Not a ped
                    continue
                elif pred_pos[j, 0] == 0:
                    # Not a ped
                    continue
                else:
                    # If he is a ped
                    if true_pos[j, 1] > 1 or true_pos[j, 1] < -1:
                        continue
                    elif true_pos[j, 2] > 1 or true_pos[j, 2] < -1:
                        continue
                    # print(true_pos[j, :])

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
        name = 'sequence' + zeros[:-digits] + str(i)
        plot_trajectories(results[i][0], results[i][1], results[i][2], name, save_location)

# results_pkl = "D:\\UofMemphis\\Coding\\notebooks\\Social_lstm_pedestrian_prediction\\Original project revised\\train_logs\\lstm\\save\\3\\results.pkl"
# video_input_path = "D:\\UofMemphis\\Coding\\notebooks\\Social_lstm_pedestrian_prediction\\Original project revised\\data\\eth\\hotel\\seq_hotel\\seq_hotel.avi"
# video_output_path = "D:\\UofMemphis\\Coding\\notebooks\\Social_lstm_pedestrian_prediction\\Original project revised\\train_logs\\lstm\\video\\3\\"
# video_trajectories(results_pkl, video_input_path, video_output_path)
