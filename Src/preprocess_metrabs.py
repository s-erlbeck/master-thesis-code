import pickle5
import os
import traceback
import sys  # for redirecting stdout to file
import itertools

import numpy as np
import torch

import util
import preprocess_pku_mmd as parse


metrabs_joint_names = {
    "l_hip": 0,
    "r_hip": 1,
    "belly": 2,
    "l_kne": 3,
    "r_kne": 4,
    "spine": 5,
    "l_ank": 6,
    "r_ank": 7,
    "thora": 8,  # thorax / chest
    "l_toe": 9,
    "r_toe": 10,
    "neck_": 11,
    "l_cla": 12,  # collarbone / clavicle
    "r_cla": 13,  # collarbone / clavicle
    "head_": 14,
    "l_sho": 15,
    "r_sho": 16,
    "l_elb": 17,
    "r_elb": 18,
    "l_wri": 19,
    "r_wri": 20,
    "l_han": 21,
    "r_han": 22,
    "pelvi": 23
}


metrabs_edges = (
    # spine
    ("pelvi", "belly"),
    ("belly", "spine"),
    ("spine", "thora"),
    ("thora", "neck_"),
    ("neck_", "head_"),
    # arm left
    ("neck_", "l_cla"),
    ("l_cla", "l_sho"),
    ("l_sho", "l_elb"),
    ("l_elb", "l_wri"),
    ("l_wri", "l_han"),
    # arm right
    ("neck_", "r_cla"),
    ("r_cla", "r_sho"),
    ("r_sho", "r_elb"),
    ("r_elb", "r_wri"),
    ("r_wri", "r_han"),
    # leg left
    ("pelvi", "l_hip"),
    ("l_hip", "l_kne"),
    ("l_kne", "l_ank"),
    ("l_ank", "l_toe"),
    # leg right
    ("pelvi", "r_hip"),
    ("r_hip", "r_kne"),
    ("r_kne", "r_ank"),
    ("r_ank", "r_toe"),
)


def get_metrabs_joint_by_name(name):
    return metrabs_joint_names[name]


def get_metrabs_edges():
    return tuple((metrabs_joint_names[edge[0]], metrabs_joint_names[edge[1]]) for edge in metrabs_edges)


def parse_action_sequence(path):
    """
    Parses raw detections from Metrabs.

    Args:
        path: The path to the detection pickle-file.

    Returns:
        A list of 2x24x3 torch tensors (2 persons, 24 joints, metric space) in meters.
    """
    res = []
    with open(path, "rb") as file:
        seq = pickle5.load(file)
        for frame in seq:
            new = torch.empty(len(frame), 24, 3)
            for i, tuple in enumerate(frame):
                new[i, :, :] = torch.tensor(tuple[1][:24, :]) / 1000
            # video is mirror inverted
            # => right-handed pose estimator will predict left-handed coordinates!!!
            # raw poses must therefore be reflected
            # has been taken care of in the tracklets!
            new[:, :, 0] *= -1
            res.append(new)

    return res


def fill_second(seq):
    """
    In the case that Metrabs should fail to detect a person, add zero tensors inplace.
    This is important for remove_false_positives to still recover.
    This is only required for raw detections, not for tracklets.

    Args:
        seq: The video sequence.
    """
    for frame in range(len(seq)):
        while len(seq[frame]) < 2:
            # this works because a tensor can also have size (0 x 24 x 3)
            seq[frame] = torch.cat([seq[frame], torch.zeros(1, 24, 3)], dim=0)


def determine_start(seq, path):
    """
    Perform a simplistic matching of the first frame to the ground truth inplace.
    This is necessary to extract the best detections from the raw Metrabs detections.
    It is only required for tracklets if it is a single person sequence and there is a false positive.

    Args:
        seq: The Metrabs pose sequence.
        path: The path to the Kinect file.
    """
    mean = torch.mean(seq[0], dim=1)
    ref = parse.parse_action_sequence(path)[0]
    ground_truth = torch.mean(ref, dim=1)

    matches, _, _ = util.hungarian_match(ground_truth, mean, dist_fn=lambda x, y: torch.norm(x-y))

    # sort the centers in the correct order
    matched_centers = [x[1] for x in matches]
    # extract the dimension where the centers lie in the original matrix mean
    indices = [np.where(mean == center)[0][0] for center in matched_centers]
    print(indices)
    # use these indices to select the correct poses
    temp = seq[0][indices]
    seq[0] = temp


def parse_tracklets(path):
    """
    Parses Metrabs tracklets.

    Args:
        path: The path to the numpy file.

    Returns:
        A list of 2x24x3 torch tensors (2 persons, 24 joints, metric space) in meters.
    """
    res = []
    seq = np.load(path)

    # preprocessing code expects two poses (which may be all zeros)
    if seq.shape[0] != 2:
        raise RuntimeError("Expected two tracklets in " + path)

    # extract the relevant 24 joints for each frame
    for i in range(seq.shape[1]):
        # no reflection required!
        frame = seq[:, i, :24, :] / 1000
        frame[frame != frame] = 0
        res.append(torch.tensor(frame, dtype=torch.float))

    return res


def count_average_num_person(path):
    """
    Count the average number of persons per frame in the Kinect ground truth.

    Args:
        path: The path to the Kinect file.

    Returns:
        The average number of persons per frame.
    """
    seq = parse.parse_action_sequence(path)
    num_person = 0

    for frame in seq:
        for person in frame:
            if torch.sum(torch.abs(person)) > 0.0:
                num_person += 1

    return num_person / len(seq)


def remove_false_positives(seq, path):
    """
    Performs frame-to-frame matching similar to check_continuity inplace.
    The difference is that it also removes detections which are likely false positives.
    This is determined based on similarity and whether the Kinect file shows 2 persons or 1 person.
    This function is required both for raw detections and tracklets:
    The reason is that the tracklets were produced by matching the Kinect sequence,
        yet sometimes the Kinect sequence also contain swaps of the poses.

    Args:
        seq: The pose sequence to process.
        path: The path to the Kinect file
    """
    avg = count_average_num_person(path)

    if 1.1 < avg < 1.7:
        # check these cases manually to see if threshold of 1.5 is correct
        print("Ground truth contains inconsistent number of persons per frame: ", avg)
        traceback.print_stack()
        print("\n")

    if avg < 1.5:
        # single-person => perform tracking of ONE person

        if torch.sum(seq[0][0, :, :]) != 0 and torch.sum(seq[0][1, :, :]) != 0:
            # two non-zero detections in the first frame of a single person sequence
            # therefore, add a zero-pose and perform hungarian matching to Kinect
            seq[0] = torch.cat([seq[0], torch.zeros(1, 24, 3)], dim=0)
            determine_start(seq, path)

        # now only one person should be non-zero
        if torch.sum(seq[0][0, :, :]) == 0:
            prev = [seq[0][1, :, :]]
        elif torch.sum(seq[0][1, :, :]) == 0:
            prev = [seq[0][0, :, :]]
        else:
            # if this should happen, the hungarian matching failed
            raise NotImplementedError("Still found false Metrabs positive in first frame.")

        # build first frame such that the real person has ID 0
        seq[0] = torch.stack([prev[0], torch.zeros_like(prev[0])], dim=0)
    else:
        # 2 persons => perform tracking for TWO people
        prev = seq[0]

    for frame in range(1, len(seq)):
        # if both persons are not visible, skip
        if torch.sum(torch.abs(seq[frame])) == 0:
            continue

        # compare current poses to previous poses via hungarian matching
        matches, _, _ = util.hungarian_match(prev, seq[frame],
                                             dist_fn=lambda x, y: torch.mean(torch.norm(x - y, dim=1)))

        # build correct order
        matches = torch.stack([x[1] for x in matches], dim=0)
        prev = matches

        # if single-person sequence, perform zero-padding
        if len(prev) == 1:
            matches = torch.cat([matches, torch.zeros_like(matches)], dim=0)

        # store resulting order
        seq[frame] = matches


def parse_tracklets_with_reid(path, video_number):
    """
    Parses the Metrabs tracks which were improved with re-identification.

    Args:
        path: The path to the track archives.
        video_number: The string of the current video sequence (4 digits with leading zeros).
    Returns:
        The left view, middle view and right view pose sequences.
    """
    path_l = os.path.join(path, video_number + "-L.npz")
    path_m = os.path.join(path, video_number + "-M.npz")
    path_r = os.path.join(path, video_number + "-R.npz")

    # two video files broken => replace respective poses with zeros
    if video_number == "0191" or video_number == "0192":
        data_l = np.zeros(shape=(6050, 1, 24, 3))
    else:
        with np.load(path_l, allow_pickle=False) as array:
            data_l = array["poses"]
    with np.load(path_m, allow_pickle=False) as array:
        data_m = array["poses"]
    # only 0363-R requires object pickling
    with np.load(path_r, allow_pickle=True) as array:
        data_r = array["poses"]

    if video_number == "0193":
        # left view shows two tracks of same person => choose first
        data_l = data_l[:, 0:1, :, :]
    elif video_number == "0363":
        # first 120 frames are empty => duplicate first known pose
        # those frames will be discarded during synch anyway
        for idx in range(120):
            data_r[idx] = data_r[120]

    # convert to float, convert to meters
    video_l = [torch.tensor(x[:, :24, :] / 1000, dtype=torch.float) for x in data_l]
    video_m = [torch.tensor(x[:, :24, :] / 1000, dtype=torch.float) for x in data_m]
    video_r = [torch.tensor(x[:, :24, :] / 1000, dtype=torch.float) for x in data_r]

    return video_l, video_m, video_r


def check_continuity_reid(seq):
    """
    Performs frame-to-frame matching inplace similar to check_continuity and remove_false_positives.
    The difference is that this is aimed towards tracks with re-identification.
    As such, tracks do not contain false positives, missing detections or frozen poses.
    This allows a simple matching against the very first frame.
    Note that tracking with re-id makes frame-to-frame matching almost obsolete.
    However, hugging lead to ID swaps in a few cases.

    Args:
        seq: The pose sequence which contains two interacting humans.
    """
    pelvis = get_metrabs_joint_by_name("pelvi")
    first = seq[0][:, pelvis, :]

    for frame in range(1, len(seq)):
        # compare current poses to FIRST FRAME via hungarian matching
        matches, _, _ = util.hungarian_match(first, seq[frame][:, pelvis, :],
                                             dist_fn=lambda x, y: torch.norm(x - y))
        # build correct order
        if (seq[frame][:, pelvis, :] != torch.stack([x[1] for x in matches], dim=0)).any():
            swapped = seq[frame][torch.LongTensor([1, 0])]
            seq[frame] = swapped


def extract_good_stamps(path, *, i=None, file_no=None):
    """
    This method reads the ground truth time stamps similar to disassemble_sequence.
    The difference is that it filters the worst time stamps (mainly missing stamps).
    It also sorts and filters action labels.
    At least one of the keyword arguments must be different from None.

    Args:
        path: The path to the time stamps.
        i: The video number between 2 and 364 (inclusive)
        file_no: The string of the video number.

    Returns:
        A list containing start and finishing times (chronological order) in the left view.
        A list containing start and finishing times (chronological order) in the middle view.
        A list containing start and finishing times (chronological order) in the right view.
        A list of the action labels consistent over all three views.
    """
    times_l = []
    times_m = []
    times_r = []

    actions_l = []
    actions_m = []
    actions_r = []

    if i is not None:
        file_no = "{:04d}".format(i)  # 0 -> padding symbol, 4 -> padding length, d -> decimal
    elif file_no is not None:
        i = int(file_no)
    else:
        raise TypeError("At least one keyword argument must be unequal to None!")

    # open the files
    with open(os.path.join(path, file_no + "-L.txt")) as file:
        for line in file:
            numbers = line.split(sep=",")
            actions_l.append(int(numbers[0]))
            times_l.extend([int(x) for x in numbers[1:3]])

    with open(os.path.join(path, file_no + "-M.txt")) as file:
        for line in file:
            numbers = line.split(sep=",")
            actions_m.append(int(numbers[0]))
            times_m.extend([int(x) for x in numbers[1:3]])

    with open(os.path.join(path, file_no + "-R.txt")) as file:
        for line in file:
            numbers = line.split(sep=",")
            actions_r.append(int(numbers[0]))
            times_r.extend([int(x) for x in numbers[1:3]])

    # some time stamps are incomplete => delete them in other views
    if i == 43:
        # wrong time stamp in 21 st action in left view resulting in negative action length
        del times_l[20 * 2: 21 * 2]
        del times_m[20 * 2: 21 * 2]
        del times_r[20 * 2: 21 * 2]
        del actions_l[20:21]
        del actions_m[20:21]
        del actions_r[20:21]
    elif i == 49:
        # 13 th action is missing in right view
        del times_l[12 * 2: 13 * 2]
        del times_m[12 * 2: 13 * 2]
        del actions_l[12:13]
        del actions_m[12:13]
    elif i == 99:
        # last two to three actions inconsistently
        del times_l[23 * 2:]
        del times_m[23 * 2:]
        del times_r[23 * 2:]
        del actions_l[23:]
        del actions_m[23:]
        del actions_r[23:]
    elif i == 119:
        # 2 nd action is missing in left view (also order messed up but will be sorted)
        del times_m[1 * 2: 2 * 2]
        del times_r[1 * 2: 2 * 2]
        del actions_m[1:2]
        del actions_r[1:2]
    elif i == 133:
        # 13 th action in mid view is only 2 frames long
        del times_l[12 * 2: 13 * 2]
        del times_m[12 * 2: 13 * 2]
        del times_r[12 * 2: 13 * 2]
        del actions_l[12:13]
        del actions_m[12:13]
        del actions_r[12:13]
    elif i == 164:
        # 16 th action missing in middle view
        del times_l[15 * 2: 16 * 2]
        del times_r[15 * 2: 16 * 2]
        del actions_l[15:16]
        del actions_r[15:16]
    elif i == 167:
        # 23 th action is missing in right view
        del times_l[22 * 2: 23 * 2]
        del times_m[22 * 2: 23 * 2]
        del actions_l[22:23]
        del actions_m[22:23]
    elif i == 198:
        # 14 th to 17 th action are mixed up
        del times_l[13 * 2: 15 * 2]
        del times_m[13 * 2: 17 * 2]
        del times_r[13 * 2: 15 * 2]
        del actions_l[13:15]
        del actions_m[13:17]
        del actions_r[13:15]
    elif i == 201:
        # 7 th action missing in left view
        del times_m[6 * 2: 7 * 2]
        del times_r[6 * 2: 7 * 2]
        del actions_m[6:7]
        del actions_r[6:7]
    elif i == 217:
        # 7 th action missing in right view
        del times_l[6 * 2: 7 * 2]
        del times_m[6 * 2: 7 * 2]
        del actions_l[6:7]
        del actions_m[6:7]
    elif i == 234:
        # 15 th action missing in left view
        del times_m[14 * 2: 15 * 2]
        del times_r[14 * 2: 15 * 2]
        del actions_m[14:15]
        del actions_r[14:15]
    elif i == 243:
        # 24 the action missing in left view
        del times_m[23 * 2: 24 * 2]
        del times_r[23 * 2: 24 * 2]
        del actions_m[23:24]
        del actions_r[23:24]
    elif i == 261:
        # 7 th action missing in left view
        del times_m[6 * 2: 7 * 2]
        del times_r[6 * 2: 7 * 2]
        del actions_m[6:7]
        del actions_r[6:7]
    elif i == 284:
        # 2 nd action missing in middle view
        del times_l[1 * 2: 2 * 2]
        del times_r[1 * 2: 2 * 2]
        del actions_l[1:2]
        del actions_r[1:2]
    elif i == 311:
        # 6 th action missing in right view
        del times_l[5 * 2: 6 * 2]
        del times_m[5 * 2: 6 * 2]
        del actions_l[5:6]
        del actions_m[5:6]
    elif i == 321:
        # 6 th action missing in right view
        del times_l[5 * 2: 6 * 2]
        del times_m[5 * 2: 6 * 2]
        del actions_l[5:6]
        del actions_m[5:6]

    times_l, actions_l = sort_actions_and_times(times_l, actions_l)
    times_m, actions_m = sort_actions_and_times(times_m, actions_m)
    times_r, actions_r = sort_actions_and_times(times_r, actions_r)

    majority_vote_actions(actions_l, actions_m, actions_r)  # makes sure that all three lists are the same

    return times_l, times_m, times_r, actions_m


def sort_actions_and_times(times, actions):
    """
    Sorts all the time stamps of when actions start and end in a single view.
    Applies same permutation to action labels.

    Args:
        times: The time stamps in chronological order.
        actions: The actions corresponding to the time stamps.
    Returns:
        The sorted list of time stamps and the permutated list of action labels.
    """
    if any([times[i] > times[i + 1] for i in range(len(times) - 1)]):
        # line swaps in the time label files occur quite often and can be fixed by sorting
        # since there is a risk of a typo mixing the correct order, we should tell the user

        # stack start and end time in pairs
        pairs = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
        # sort both actions and times via time stamps
        sorted_zip = sorted(zip(pairs, actions), key=lambda x: x[0][0])
        # undo zipping, i.e. obtain two lists again
        pairs, actions = map(list, zip(*sorted_zip))  # list is pointer to casting function
        # transform pairs into list of time stamps again
        times = list(itertools.chain(*pairs))   # asterisk unpacks into vararg,

        # print("Time stamps were not ordered.")
        # traceback.print_stack()
    return times, actions


def majority_vote_actions(action_l, action_m, action_r):
    """
    Iterates through all actions and removes inconsistencies by relying on the majority.
    Should be called after manual inconsistency solving in extract_good_stamps and after sorting.
    Changes the lists inplace.

    Args:
        action_l: The actions as labelled in the left view.
        action_m: The actions as labelled in the middle view.
        action_r: The actions as labelled in the right view.
    """
    for i, (x, y, z) in enumerate(zip(action_l, action_m, action_r)):
        if not (x == y == z):
            majority = int(np.median([x, y, z]))
            action_l[i] = majority
            action_m[i] = majority
            action_r[i] = majority


def disassemble_good_only(seq, times):
    """
    Works similar to disassemble_sequence, but after times have been filtered by extract_good_stamps.
    Also, idle actions are going to be synchronised as well when using this method.

    Args:
        seq: The video which should be cut into single actions.
        times: The times returned by extract_good_stamps

    Returns:
        A list of list of tensors, each inner list representing a single action.
    """
    res = []

    for j in range(len(times) - 1):
        if times[j + 1] == times[j] and times[j + 1] < len(seq):
            # in sequence 7, one idle phase is missing => sacrifice one action frame
            times[j + 1] += 1
        action = seq[times[j]:times[j + 1]]
        res.append(action)
    return res


def get_bone_vector_matrix():
    """
    Computes the matrices to transform metrabs joints to bone vectors and back to joints (the inverse matrix).
    The bone vector representation keeps the absolute position of the root joint in order to allow reconstruction.
    Using these matrices is faster than the function in util module but only works with metrabs.
    The operation also works with multiple poses, e.g. joints (2 x 72) @ transpose(to_bones) (72 x 72)

    Returns:
        The 72 x 72 torch tensor to transform joints to bones.
    """
    transform_bones = torch.zeros(72, 72)

    # pelvis point kept for reconstruction:
    transform_bones[0, get_metrabs_joint_by_name("pelvi") * 3 + 0] = 1
    transform_bones[1, get_metrabs_joint_by_name("pelvi") * 3 + 1] = 1
    transform_bones[2, get_metrabs_joint_by_name("pelvi") * 3 + 2] = 1

    # all the bones
    for i, edge in enumerate(get_metrabs_edges()):
        # i + 1 because the first three columns contain root joint
        transform_bones[((i + 1) * 3 + 0), edge[0] * 3 + 0] = -1
        transform_bones[((i + 1) * 3 + 1), edge[0] * 3 + 1] = -1
        transform_bones[((i + 1) * 3 + 2), edge[0] * 3 + 2] = -1
        transform_bones[((i + 1) * 3 + 0), edge[1] * 3 + 0] = 1
        transform_bones[((i + 1) * 3 + 1), edge[1] * 3 + 1] = 1
        transform_bones[((i + 1) * 3 + 2), edge[1] * 3 + 2] = 1

    return transform_bones


def get_root_relative_matrix(invertible=False):
    """
    Computes the matrices to transform metrabs joints to root-relative joints and back to absolute joints (the inverse).
    The operation also works with multiple poses, e.g. joints (2 x 72) @ transpose(to_relative) (72 x 72)

    Args:
        invertible: If true, the global position of pelvis is kept to allow reconstruction of movement through space.
    Returns:
        The 72 x 72 torch tensor to transform joints to root-relative coordinates.
    """
    transform_bones = torch.eye(72)
    pelvis = get_metrabs_joint_by_name("pelvi")

    for i in range(24):
        if invertible and i == pelvis:
            # if global position should be modelled, the pelvis location needs to be kept
            continue
        transform_bones[i * 3 + 0, pelvis * 3 + 0] -= 1
        transform_bones[i * 3 + 1, pelvis * 3 + 1] -= 1
        transform_bones[i * 3 + 2, pelvis * 3 + 2] -= 1

    return transform_bones


def numpy_save_sequence(seq, path):
    """
    Used for intermediate results of preprocessing.
    This function is for the whole sequence.

    Args:
        seq: The pose sequence.
        path: The string containing path and filename.
    """
    stored = np.empty((len(seq), 2, 24, 3))
    for i in range(len(seq)):
        stored[i, :, :, :] = seq[i].numpy()
    np.save(path, stored, allow_pickle=False)


def numpy_load_sequence(path):
    """
    Used for intermediate results of preprocessing.
    This function is for the whole sequence.

    Args:
        path: The string containing path and filename.

    Returns:
        The list of pose tensors.
    """
    res = []
    stored = np.load(path, allow_pickle=False)
    for frame in stored:
        res.append(torch.tensor(frame, dtype=torch.float))
    return res


def pickle_save(seq, path):
    """
    Used for intermediate results of preprocessing.
    This one is different because it is supposed to store a list of lists of arrays.

    Args:
        seq: The pose sequence.
        path: The string containing path and filename.
    """
    with open(path, "wb") as file:
        pickle5.dump(seq, file)


def pickle_load(path):
    """
    Used for intermediate results of preprocessing.
    This one is different because it is supposed to load a list of lists of arrays.

    Args:
        path: The string containing path and filename.

    Returns:
        The list of pose tensors.
    """
    with open(path, "rb") as file:
        return pickle5.load(file)


def preprocess_first(start=2, end=364):
    """
    Performs the first step of preprocessing.
    Namely, it ensures that pose 0 shows the same person in each frame and for each Kinect sensor.

    Args:
        start: The index to start at.
        end: The index to end with (inclusive).
    """
    torch.manual_seed(42)
    path_metrabs = "/globalwork/sarandi/data/pkummd-more/tracks_reid_65ed967/"
    path_store = "/work/erlbeck/datasets/pkummd_enhanced/tracklets"

    for i in range(start, end + 1):
        # some or all views are missing in these cases:
        if i == 60 or 247 <= i <= 250:
            continue

        video_number = "{:04d}".format(i)  # 0 -> padding symbol, 4 -> padding length, d -> decimal
        print(video_number)

        # parsing the metrabs tracks with re-identification tracking
        video_l, video_m, video_r = parse_tracklets_with_reid(path_metrabs, video_number)

        if video_l[0].size()[0] == video_m[0].size()[0] == video_r[0].size()[0] == 1:
            # add zero poses to single-person videos (compatibility reasons)
            video_l = [torch.cat([x, torch.zeros_like(x)], dim=0) for x in video_l]
            video_m = [torch.cat([x, torch.zeros_like(x)], dim=0) for x in video_m]
            video_r = [torch.cat([x, torch.zeros_like(x)], dim=0) for x in video_r]
        else:
            # two person found => frame-to-frame matching (due to rare id swaps)
            check_continuity_reid(video_l)
            check_continuity_reid(video_m)
            check_continuity_reid(video_r)

            # now match across different camera views
            edges = get_metrabs_edges()
            parse.identify_cross_view(video_m, video_l, edges, [])
            parse.identify_cross_view(video_m, video_r, edges, [])

        # store intermediate results
        numpy_save_sequence(video_l, os.path.join(path_store, video_number + "-L.npy"))
        numpy_save_sequence(video_m, os.path.join(path_store, video_number + "-M.npy"))
        numpy_save_sequence(video_r, os.path.join(path_store, video_number + "-R.npy"))


def preprocess_second(start=2, end=364):
    """
    Performs the second step of preprocessing.
    Namely, it disassembles the videos into actions, synchronises the different views and compute extrinsics.

    Args:
        start: The index to start at.
        end: The index to end with (inclusive).
    """
    torch.manual_seed(42)
    path_load = "/work/erlbeck/datasets/pkummd_enhanced/tracklets"
    path_store = "/work/erlbeck/datasets/pkummd_enhanced/calibrated"
    path_stamp_dir = "/globalwork/datasets/pkummd/Train_Label_PKU_final"

    for i in range(start, end + 1):
        # some or all views are missing in these cases:
        if i == 60 or 247 <= i <= 250:
            continue

        video_number = "{:04d}".format(i)  # 0 -> padding symbol, 4 -> padding length, d -> decimal
        print(video_number)

        # load
        video_l = numpy_load_sequence(os.path.join(path_load, video_number + "-L.npy"))
        video_m = numpy_load_sequence(os.path.join(path_load, video_number + "-M.npy"))
        video_r = numpy_load_sequence(os.path.join(path_load, video_number + "-R.npy"))

        # disassemble into actions
        time_l, time_m, time_r, actions = extract_good_stamps(path_stamp_dir, file_no=video_number)
        video_l = disassemble_good_only(video_l, time_l)
        video_m = disassemble_good_only(video_m, time_m)
        video_r = disassemble_good_only(video_r, time_r)

        # synchronise the sequences
        edges = get_metrabs_edges()
        for j in range(len(video_m)):
            seq_l = video_l[j]
            seq_m = video_m[j]
            seq_r = video_r[j]
            parse.synchronise_views(seq_l, seq_m, seq_r, edges, [])

        # compute calibration
        parse.calibrate(video_l, video_m, video_r)

        # compute new time stamps
        times = [0]
        cumulative = 0
        for action in video_m:
            cumulative += len(action)
            times.append(cumulative)

        # chain the actions together again
        video_l = list(itertools.chain.from_iterable(video_l))
        video_m = list(itertools.chain.from_iterable(video_m))
        video_r = list(itertools.chain.from_iterable(video_r))

        # store
        numpy_save_sequence(video_l, os.path.join(path_store, video_number + "-L.npy"))
        numpy_save_sequence(video_r, os.path.join(path_store, video_number + "-R.npy"))
        numpy_save_sequence(video_m, os.path.join(path_store, video_number + "-M.npy"))
        pickle_save(times, os.path.join(path_store, video_number + "-times.pkl"))
        pickle_save(actions, os.path.join(path_store, video_number + "-actions.pkl"))


def preprocess_third(start=2, end=364):
    """
    Performs the third step of preprocessing.
    Namely, it discards unrealistic poses and fuses the different views.

    Args:
        start: The index to start at.
        end: The index to end with (inclusive).
    """
    torch.manual_seed(42)
    path_load = "/work/erlbeck/datasets/pkummd_enhanced/calibrated"
    path_store = "/work/erlbeck/datasets/pkummd_enhanced/combined"

    for i in range(start, end + 1):
        # some or all views are missing in these cases:
        if i == 60 or 247 <= i <= 250:
            continue

        video_number = "{:04d}".format(i)  # 0 -> padding symbol, 4 -> padding length, d -> decimal
        print(video_number)

        # load
        video_l = numpy_load_sequence(os.path.join(path_load, video_number + "-L.npy"))
        video_m = numpy_load_sequence(os.path.join(path_load, video_number + "-M.npy"))
        video_r = numpy_load_sequence(os.path.join(path_load, video_number + "-R.npy"))
        times = pickle_load(os.path.join(path_load, video_number + "-times.pkl"))
        actions = pickle_load(os.path.join(path_load, video_number + "-actions.pkl"))

        edges = get_metrabs_edges()

        # remove implausible bone lengths
        parse.check_bone_length(video_l, "/globalwork/datasets/pkummd/mean_bone_length_metrabs.txt", edges, verbose=False)
        parse.check_bone_length(video_m, "/globalwork/datasets/pkummd/mean_bone_length_metrabs.txt", edges, verbose=False)
        parse.check_bone_length(video_r, "/globalwork/datasets/pkummd/mean_bone_length_metrabs.txt", edges, verbose=False)

        # fuse ground truths
        res = parse.fuse_ground_truth_geo_median(video_l, video_m, video_r, get_metrabs_joint_by_name("pelvi"), times)
        # use pickle because numpy_save_sequence expects 2 people:
        pickle_save(res, os.path.join(path_store, video_number + "-poses.pkl"))
        pickle_save(times, os.path.join(path_store, video_number + "-times.pkl"))
        pickle_save(actions, os.path.join(path_store, video_number + "-actions.pkl"))


def preprocess_fourth(start=2, end=364):
    """
    Performs the fourth step of preprocessing.
    Namely, it performs a final filtering after the fusion and stores them for the learning.
    Filtering affects zero poses, frozen poses and insane bone lengths.

    Args:
        start: The index to start at.
        end: The index to end with (inclusive).
    """
    torch.manual_seed(42)
    path_load = "/work/erlbeck/datasets/pkummd_enhanced/combined"
    path_store = "/work/erlbeck/datasets/pkummd_enhanced/final"
    edges = get_metrabs_edges()

    for i in range(start, end + 1):
        # some or all views are missing in these cases:
        if i == 60 or 247 <= i <= 250:
            continue

        video_number = "{:04d}".format(i)  # 0 -> padding symbol, 4 -> padding length, d -> decimal
        print(video_number)

        seq = pickle_load(os.path.join(path_load, video_number + "-poses.pkl"))
        times = pickle_load(os.path.join(path_load, video_number + "-times.pkl"))
        actions = pickle_load(os.path.join(path_load, video_number + "-actions.pkl"))

        # set implausible frames to 0
        previous = torch.clone(seq[0])
        for frame in seq[1:]:
            for j, person in enumerate(frame):
                # note that pseudo-track in single-person sequences was already removed
                if (person == previous[j]).all():
                    frame[j, :, :] = 0
                    print("Found frozen pose.")
                elif (torch.abs(person) > 10).any():
                    # sometimes, the correct pose is placed at a wrong location
                    # => hard to distinguish from id swaps etc.
                    frame[j, :, :] = 0
                    print("Found pose outside realistic range.")
                else:
                    # might be zero pose, but then comparison to previous is meaningless anyway
                    previous[j] = frame[j, :, :]
        # since frozen poses are set to zero, bone length check will ignore them
        parse.check_bone_length(seq, "/globalwork/datasets/pkummd/mean_bone_length_metrabs.txt", edges, verbose=False, two_poses=False)

        # collect all bad frame ids for deletion
        bad_frames_succ = 0
        bad_frames_all = 0
        discard_frames = []
        for j, frame in enumerate(seq):
            sums = torch.sum(torch.abs(frame), dim=(1, 2))
            if (frame == frame).all() and (sums > 0).all():
                if bad_frames_succ > 10:
                    # sometimes, > 10 consecutive frames are deleted
                    # however, analysis showed no relevant information in those cases
                    # => just delete them
                    print("bad frames in succession:", bad_frames_succ)
                bad_frames_succ = 0
            else:
                discard_frames.append(j)
                bad_frames_succ += 1
                bad_frames_all += 1
        if bad_frames_all > 20:
            print("bad frames in total:", bad_frames_all)
        # for readability
        print("")

        # delete all frames with NaNs or zero poses
        discard_frames.reverse()
        for j in discard_frames:
            del seq[j]
            # removing a frame means decrementing all later time stamps
            for k, stamp in enumerate(times):
                if stamp > j:
                    times[k] -= 1

        # store sequence
        tensor = torch.empty(len(seq), seq[0].size()[0], 24 * 3)
        for j, frame in enumerate(seq):
            tensor[j, :, :] = frame.view(-1, 24 * 3)
        torch.save(tensor, os.path.join(path_store, video_number + "-poses.pt"))
        torch.save(torch.tensor(times, dtype=torch.long), os.path.join(path_store, video_number + "-times.pt"))
        torch.save(torch.tensor(actions, dtype=torch.long), os.path.join(path_store, video_number + "-actions.pt"))


if __name__ == "__main__":
    # commented out so one does not accidentally overwrite files
    # sys.stdout = open("log_preprocess_fourth_stage.txt", "w")
    # sys.stderr = sys.stdout
    # preprocess_fourth()
    pass
