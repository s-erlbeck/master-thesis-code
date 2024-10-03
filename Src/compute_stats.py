import sys
import os
import openpyxl
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import torch
import numpy as np

import preprocess_pku_mmd as pp
import preprocess_metrabs as pm
import preprocess_ntu_rgbd as pn


def count_pkummd():
    """
    Runs through PKU-MMD data set and counts the number of frames, the number of multi-person frames, etc.
    """
    path_poses = "/globalwork/datasets/pkummd/PKU_Skeleton_Renew/"
    path_times = "/globalwork/datasets/pkummd/Train_Label_PKU_final/"

    num_people = 0
    num_single_person_frames = 0
    num_interaction_frames = 0
    num_frames = 0

    for root, _, files in os.walk(path_poses):
        for file in files:
            video = pp.parse_action_sequence(os.path.join(root, file))
            video = pp.disassemble_sequence(video, os.path.join(path_times, file))
            for action in video:
                for frame in action:
                    person = torch.sum(torch.abs(frame), dim=(1,2))
                    if person[0] > 0:
                        num_people += 1
                    if person[1] > 0:
                        num_people += 1
                    if person[0] > 0 and person[1] > 0:
                        num_interaction_frames += 1
                    elif person[0] > 0 or person[1] > 0:
                        num_single_person_frames += 1
                    num_frames += 1

    print("Number of poses: " + str(num_people))
    print("Number of interaction frames: " + str(num_interaction_frames))
    print("Number of single-person frames: " + str(num_single_person_frames))
    print("Number of frames in total: " + str(num_frames))


def cound_nturgbd():
    """
    Runs through NTU-RGB+D data set and counts the number of frames, the number of multi-person frames, etc.
    """
    path_poses = "/globalwork/datasets/NTU_RGBD/nturgb+d_skeletons/"

    num_people = 0
    num_single_person_frames = 0
    num_interaction_frames = 0
    num_frames = 0

    for root, _, files in os.walk(path_poses):
        for file in files:
            try:
                with open(os.path.join(root, file)) as handle:
                    # read number of frames
                    line = handle.readline()
                    length = int(line)
                    num_frames += length

                    for frame in range(length):
                        # read number of detections
                        line = handle.readline()
                        num_detections = int(line)
                        num_people += num_detections
                        if num_detections == 2:
                            num_interaction_frames += 1
                        elif num_detections == 1:
                            num_single_person_frames += 1

                        for detection in range(num_detections):
                            # skip a line containing some meta info
                            _ = handle.readline()

                            # read number of joints
                            line = handle.readline()
                            num_joints = int(line)

                            # skip all joints
                            for i in range(num_joints):
                                _ = handle.readline()
            except ValueError:
                pass
            finally:
                print(file)

    print("Number of poses: " + str(num_people))
    print("Number of interaction frames: " + str(num_interaction_frames))
    print("Number of single-person frames: " + str(num_single_person_frames))
    print("Number of frames in total: " + str(num_frames))


def analyse_pkummd_timestamps(whole_sequence):
    """
    Runs through PKU-MMD time stamps and measures the differences in framerate.
    For example, "1.3: 100" means that there were 100 sequences were a view's framerate was 1.3 times higher
        than the view with the lowest framerate.

    Args:
        whole_sequence: If true, measure the difference on whole video,
            else measure the difference on each isolated action.
    """
    path = "/home/embreaded/Downloads/PKUMMD/Train_Label_PKU_final"
    res = Counter()

    for i in range(2, 365):
        # some or all views are missing in these cases:
        if i == 60 or 247 <= i <= 250:
            continue

        # extract the good time stamps
        vec_l, vec_m, vec_r, _ = pm.extract_good_stamps(path, i=i)

        # actual routine
        if whole_sequence:
            # consider whole video
            time_l_1 = vec_l[-1] - vec_l[0]
            time_m_1 = vec_m[-1] - vec_m[0]
            time_r_1 = vec_r[-1] - vec_r[0]

            mini, midi, maxi = list(sorted([time_l_1, time_m_1, time_r_1]))
            res[round(midi / mini, 1)] += 1
            res[round(maxi / mini, 1)] += 1

        else:
            # consider each action individually
            for j in range(0, max(len(vec_l), len(vec_m), len(vec_r)), 2):
                time_l = vec_l[j + 1] - vec_l[j]
                time_m = vec_m[j + 1] - vec_m[j]
                time_r = vec_r[j + 1] - vec_r[j]

                mini, midi, maxi = list(sorted([time_l, time_m, time_r]))

                res[round(midi / mini, 1)] += 1
                res[round(maxi / mini, 1)] += 1

    # print result
    for key in sorted(res.keys()):
        print(str(key) + ": " + str(res[key]))


def count_zero_poses():
    """
    Runs through the preprocessed PKU-MMD data and counts how often poses are completely zero.
    """
    path_poses = "/work/erlbeck/datasets/pkummd_enhanced/combined"

    poses = 0

    for root, _, files in os.walk(path_poses):
        files_all = len(files)
        for file in files:
            seq = pm.pickle_load(os.path.join(root, file))
            zeros = 0
            for frame in seq:
                for pose in frame:
                    poses += 1
                    if torch.sum(pose) == 0:
                        zeros += 1

            if zeros > 0:
                print(zeros, "zero poses in file", file)

    print("Number of all poses:", poses)
    print("Number of all files:", files_all)


def get_pkummd_action_overview():
    """
    Computes an overview of which sequences show which actions.
    Stores a dictionary as a pickle. Keys are action labels (as strings)
        and values describe video number and sequence number, separated by "_".
    """
    path = "/globalwork/sarandi/data/pkummd/Train_Label_PKU_final/"
    wb = openpyxl.load_workbook(filename=os.path.join(path, "../Split/Actions.xlsx"))
    res = {}

    for i in range(2, 365):
        file_no = "{:04d}".format(i)  # 0 -> padding symbol, 4 -> padding length, d -> decimal

        # some time stamps are incomplete => skip them
        if i == 43 or i == 49 or i == 99 or i == 119 or i == 133 or i == 164 \
                or i == 167 or i == 198 or i == 201 or i == 217 or i == 234 \
                or i == 243 or i == 261 or i == 284 or i == 311 or i == 321 \
                or i == 60 or 247 <= i <= 250:
            continue

        with open(os.path.join(path, file_no + "-M.txt")) as file:
            for i, line in enumerate(file):
                # get action label and look up its meaning in the excel sheet
                label = int(line.split(sep=",")[0])
                label_as_str = wb["Sheet1"]["B" + str(label + 1)].value
                # add key-vallue pair to dict
                if label_as_str in res:
                    res[label_as_str].append(file_no + "_" + str(i))
                else:
                    res[label_as_str] = [file_no + "_" + str(i)]

    with open("/home/erlbeck/projects/master-thesis/pkummd_label_overview.pkl", "wb") as file:
        pickle.dump(res, file)


def get_distribution_of_dual_person_videos():
    """
    Computes which videos show two person and some related percentages.
    """
    path = "/globalwork/datasets/pkummd/Train_Label_PKU_final/"

    data_dual = 0
    data_all = 0

    for i in range(37):

        sub_dual = 0
        sub_all = 0

        for j in range(10):
            file_no = "{:04d}".format(i * 10 + j)  # 0 -> padding symbol, 4 -> padding length, d -> decimal

            try:
                labels = []
                with open(path + file_no + "-M.txt") as file:
                    for line in file:
                        labels.append(int(line.split(",")[0]))

                    if 18 in labels or 26 in labels or 27 in labels:
                        sub_dual += 1
                        print(file_no)
                    sub_all += 1
            except FileNotFoundError:
                continue

        data_dual += sub_dual
        data_all += sub_all

        proportion_dual = sub_dual / sub_all
        if proportion_dual != 0.2:
            print(str(i * 10) + "er: {:0.2f}".format(proportion_dual))

    print("Overall percentage of dual person sequences:", data_dual / data_all)
    print("Overall number of sequences:", data_all)


def count_bad_tracklets_pku():
    """
    Counts how often poses teleport (= pelvis moves 50 cm in one frame) in PKU-MMD.
    """
    path = "/globalwork/sarandi/data/pkummd-more/tracks_reid_65ed967/"

    for i in range (2, 365):
        if i == 60 or i == 191 or i == 192 or 247 <= i <= 250:
            continue

        video_num = "{:04d}".format(i)
        num_people = []

        with np.load(path + video_num + "-L.npz") as data:
            print(video_num + "-L:")
            num_people.append(data["poses"].shape[1])
            count_teleports_helper(data)
        with np.load(path + video_num + "-M.npz") as data:
            print(video_num + "-M:")
            num_people.append(data["poses"].shape[1])
            count_teleports_helper(data)
        with np.load(path + video_num + "-R.npz") as data:
            print(video_num + "-R:")
            num_people.append(data["poses"].shape[1])
            count_teleports_helper(data)

        if num_people != [1, 1, 1] and num_people != [2, 2, 2]:
            print("\nWARNING!!! NUMBER OF POSES INCONSISTENT!")
            print("#people", num_people)
        print("\n")


def count_teleports_helper(data):
    """
    Given a Metrabs view, it counts how often poses teleported.

    Args:
        data: The Metrabs detections from one of the Kinect cameras.
    """
    pelvis = pm.get_metrabs_joint_by_name("pelvi")

    for person in range(data["poses"].shape[1]):
        previous = data["poses"][0, person, pelvis, :]
        num_zero_poses = int(sum(abs(previous)) == 0)
        num_teleports = 0
        for frame in range(1, len(data["poses"])):
            current = data["poses"][frame, person, pelvis, :]
            if sum(abs(current)) == 0:
                # detected zero pose
                num_zero_poses += 1
            elif sum(abs(previous)) != 0:
                if np.linalg.norm(previous - current) > 500:
                    print("teleport in:", frame)
                    num_teleports += 1
            previous = current

        print("#zeros:", num_zero_poses)
        print("#teleports:", num_teleports)


def review_fastdtw_synch():
    """
    Plots the x-coordinate of the right hand of person one over time.
    Can be used to evaluate the quality of synchronisation.
    Resulting plots are stored on disk.
    """
    path = "/work/erlbeck/datasets/pkummd_enhanced/calibrated/"

    for i in range(2, 365):
        # some or all views are missing in these cases:
        if i == 60 or 247 <= i <= 250:
            continue

        video_number = "{:04d}".format(i)  # 0 -> padding symbol, 4 -> padding length, d -> decimal

        data = np.load(path + video_number + "-L.npy")
        hand_l = data[:, 0, pm.get_metrabs_joint_by_name("r_han"), 0]  # x coordinate (horizontal)
        data = np.load(path + video_number + "-M.npy")
        hand_m = data[:, 0, pm.get_metrabs_joint_by_name("r_han"), 0]  # x coordinate (horizontal)
        data = np.load(path + video_number + "-R.npy")
        hand_r = data[:, 0, pm.get_metrabs_joint_by_name("r_han"), 0]  # x coordinate (horizontal)

        # trim to 2 meters so that plot scale is still useful in presence of outliers
        hand_l[hand_l > 2] = 2
        hand_l[hand_l < -2] = -2
        hand_m[hand_m > 2] = 2
        hand_m[hand_m < -2] = -2
        hand_r[hand_r > 2] = 2
        hand_r[hand_r < -2] = -2

        plt.clf()
        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(hand_l)
        plt.plot(hand_m)
        plt.plot(hand_r)
        plt.savefig("../Results/PKU-MMD/synchronisation/synch visualisation/" + video_number + ".jpg")
        plt.close()


def count_bad_tracklets_ntu():
    """
    Counts how often poses teleport (= pelvis moves 50 cm in one frame) in NTU-RGB+D.
    Also counts zero poses and frozen poses.
    """
    path = "/work/erlbeck/datasets/nturgbd_enhanced/combined"

    num_zero = 0
    num_frozen = 0
    num_teleported = 0

    set_zero = set()
    set_frozen = set()
    set_teleported = set()

    for file in os.listdir(path):
        seq = pm.pickle_load(os.path.join(path, file))
        seq_mod = [y.view(-1, 72) for y in seq]

        num_zero += torch.sum(torch.all(seq_mod[0] == 0, dim=1)).item()

        for i in range(1, len(seq_mod)):
            for j in range(seq_mod[i].size(0)):
                is_zero = torch.all(seq_mod[i][j, :] == 0)
                if is_zero:
                    num_zero += 1
                    set_zero.add(file)
                elif not torch.all(seq_mod[i - 1][j, :] == 0):
                    is_frozen = torch.all(seq_mod[i][j, :] == seq_mod[i - 1][j, :])
                    diff = seq_mod[i][j, :].view(24, 3) - seq_mod[i - 1][j, :].view(24, 3)
                    dist = torch.linalg.norm(diff, dim=1)
                    is_teleported = dist[-1] > 0.5
                    if is_frozen:
                        num_frozen += 1
                        set_frozen.add(file)
                    if is_teleported:
                        num_teleported += 1
                        set_teleported.add(file)

    print("number of zero poses:", num_zero)
    print("number of frozen poses:", num_frozen)
    print("number of teleported poses:", num_teleported)

    print("size of set with zeros", len(set_zero))
    print("size of set with frozen", len(set_frozen))
    print("size of set with teleports", len(set_teleported))

    print("zero and frozen", len(set_zero.intersection(set_frozen)))
    print("zero and teleport", len(set_zero.intersection(set_teleported)))
    print("frozen and teleport", len(set_frozen.intersection(set_teleported)))

    print("all", len(set_zero.intersection(set_frozen).intersection(set_teleported)))

    counter = Counter()
    num_people_in_teleports = [pn.get_expected_num_people(el) for el in set_teleported]
    counter.update(num_people_in_teleports)
    print("\nnumber of motions with teleports by number of people\n", counter)


def count_teleports_after_preprocessing():
    """
    Counts the number of teleports after preprocessing.
    """
    path = "/work/erlbeck/datasets/nturgbd_enhanced/normalised/"
    # path = "/work/erlbeck/datasets/pkummd_enhanced/final/"

    num_tel = 0
    set_files = set()

    for file in os.listdir(path):
        x = torch.load(os.path.join(path, file))

        if x.size(-1) != 72:
            # skip time stamps etc. for PKU
            continue

        data = x.view(x.size(0), -1, 24, 3)[:, :, -1, :]

        for i in range(1, data.size(0)):
            dist = torch.linalg.norm(data[i, :, :] - data[i - 1, :, :], dim=1)
            if (dist > 0.5).any():
                set_files.add(file)
                num_tel += (dist > 0.5).sum().item()

    print("number of teleports", num_tel)
    print("number of affected files", len(set_files))
    print(set_files)


if __name__ == "__main__":
    sys.stdout = open("log_quality_ntu_before_final_filtering.txt", "w")
    sys.stderr = sys.stdout
    count_bad_tracklets_ntu()
