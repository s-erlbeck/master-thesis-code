import os

import numpy as np
import torch
from fastdtw import fastdtw

import util
import visualisation
import preprocess_metrabs as pm
import preprocess_pku_mmd as pp


def get_moved_distance(seq):
    """
    Computes view-invariant representation by taking the euclidean distance of each joints' movement between frames.
    Could be used as alternative to angle representation (but first test showed no advantage).
    Is probably faster than angle computation, but since synch using angles took so long before it worked,
        I will probably never touch this topic again.

    Args:
        seq: The pose sequence to compute the view-invariant features for.
    Returns:
        The view-invariant representation consisting of moved distances.
    """
    prev = seq[0]
    res = [torch.zeros(seq[0].size(0) * seq[0].size(1)).numpy()]

    for i in range(1, len(seq)):
        moved = seq[i] - prev
        res.append(torch.norm(moved, dim=2).flatten().numpy())
        prev = seq[i]

    return res


def get_rot():
    """
    Computes a rotation matrix for the synthetically created pose sequence.

    Returns:
        The rotation matrix.
    """
    R1 = torch.tensor([[np.cos(0.5), -np.sin(0.5), 0],
                       [np.sin(0.5), np.cos(0.5), 0],
                       [0, 0, 1]], dtype=torch.float32)
    R2 = torch.tensor([[np.cos(-0.2), 0, np.sin(-0.2)],
                       [0, 1, 0],
                       [-np.sin(-0.2), 0, np.cos(-0.2)]], dtype=torch.float32)
    return R1 @ R2


def helper_trim(pair_0, pair_1, path_pair, foreign, foreign_inv, edges, radius, distance):
    """
    This function differs to the one in preprocess_pku_mmd.py in the distance function
        used for fastdtw.
    """
    delete = pp.align_sequences(path_pair)
    for i in delete[0]:
        del pair_0[i]
    for i in delete[1]:
        del pair_1[i]
    inv_pair = util.get_view_invariant_representation(pair_0, edges, [])
    _, path_final = fastdtw(inv_pair, foreign_inv, radius=radius, dist=distance)
    delete = pp.align_sequences(path_final)
    for i in delete[0]:
        del pair_0[i]
        del pair_1[i]
    for i in delete[1]:
        del foreign[i]


def parse_seq_len_mode(seq_len_mode, time_l, time_m, time_r, video_l, video_m, video_r):
    """
    Computes the time frames / subsequences which are then synchronised.

    Args:
        seq_len_mode: Whether "short", "medium", "long" or "untrimmed" sequence length is used for synch.
        time_l: The time labels of the left view.
        time_m: The time labels of the middle view.
        time_r:  The time labels of the right view.
        video_l: The left view.
        video_m: The middle view.
        video_r: The right view.
    Returns:
        Three lists of time labels of left, middle and right view. Each list alternates between start and end index.
    """
    if seq_len_mode == "short":
        # get first, middle and last action, discard all inactivity to have short sequences

        # first, get index of the "most middle action"
        # the time stamps at position 0, 2, 4, ... are starting time stamps
        # while the time stamps at positions 1, 3, ... are ending time stamps
        # therefore we need to distinguish between even and odd number of actions (#actions = len(time_m) / 2)
        if (len(time_m) // 2) % 2 == 0:
            # number of actions is even
            mid = len(time_m) // 2
        else:
            # number of actions is odd
            mid = len(time_m) // 2 - 1

        mod_time_l = [time_l[0:2],
                      time_l[mid:(mid + 2)],
                      time_l[(len(time_l) - 2):]]
        mod_time_m = [time_m[0:2],
                      time_m[mid:(mid + 2)],
                      time_m[(len(time_m) - 2):]]
        mod_time_r = [time_r[0:2],
                      time_r[mid:(mid + 2)],
                      time_r[(len(time_r) - 2):]]
    elif seq_len_mode == "medium":
        # get first, middle and last action, but keep inactivity to have medium long sequences

        # in "short", mid decides whether we analyse action or inactivity
        # here, mid decides whether action comes before inactivity or vice versa
        # since we do not care for the order, we do not need if-statement
        mid = len(time_m) // 2 - 1

        mod_time_l = [[time_l[0], time_l[2]],
                      [time_l[mid], time_l[(mid + 2)]],
                      [time_l[-3], time_l[-1]]]
        mod_time_m = [[time_m[0], time_m[2]],
                      [time_m[mid], time_m[(mid + 2)]],
                      [time_m[-3], time_m[-1]]]
        mod_time_r = [[time_r[0], time_r[2]],
                      [time_r[mid], time_r[(mid + 2)]],
                      [time_r[-3], time_r[-1]]]
    elif seq_len_mode == "long":
        # get whole sequence (but trim the very first and last frames)
        mod_time_l = [[time_l[0], time_l[-1]]]
        mod_time_m = [[time_m[0], time_m[-1]]]
        mod_time_r = [[time_r[0], time_r[-1]]]
    elif seq_len_mode == "untrimmed":
        # keep the whole sequence
        # this may result in bad synch if one kinect was activated significantly earlier

        mod_time_l = [[0, len(video_l)]]
        mod_time_m = [[0, len(video_m)]]
        mod_time_r = [[0, len(video_r)]]
    else:
        raise ValueError("Unknown sequence length mode: " + str(seq_len_mode))

    return mod_time_l, mod_time_m, mod_time_r


def test_synch_two(seq_len, radius, distance, dist_threshold, video_number):
    """
    Using two views of the same action sequences.
    """
    file_number = "{:04d}".format(video_number)
    path_metrabs = "/globalwork/sarandi/data/pkummd-more/estimates_38814fa9/"
    path_kinect = "/globalwork/datasets/pkummd/PKU_Skeleton_Renew/"
    path_stamp_dir = "/globalwork/datasets/pkummd/Train_Label_PKU_final"
    file_m = file_number + "-M.npy"
    file_r = file_number + "-R.npy"
    video_m = pm.parse_tracklets(os.path.join(path_metrabs, file_m))
    video_r = pm.parse_tracklets(os.path.join(path_metrabs, file_r))
    pm.remove_false_positives(video_m, os.path.join(path_kinect, file_m[:-3] + "txt"))
    pm.remove_false_positives(video_r, os.path.join(path_kinect, file_r[:-3] + "txt"))
    edges = pm.get_metrabs_edges()
    pp.identify_cross_view(video_m, video_r, edges, [])
    time_l, time_m, time_r, _ = pm.extract_good_stamps(path_stamp_dir, file_no=file_number)

    # fill in dummy values for left sequence, which was not parsed
    _, mod_time_m, mod_time_r = parse_seq_len_mode(seq_len, time_m, time_m, time_r, video_m, video_m, video_r)

    for time_m, time_r in zip(mod_time_m, mod_time_r):
        seq_m = video_m[time_m[0]:time_m[1]]
        seq_r = video_r[time_r[0]:time_r[1]]

        inv_m = util.get_view_invariant_representation(seq_m, edges, [])
        inv_r = util.get_view_invariant_representation(seq_r, edges, [])

        score, path = fastdtw(inv_r, inv_m, radius=radius, dist=lambda x, y: distance(x, y, dist_threshold))

        print("The sequences are {:d} and {:d} frames long.".format(len(seq_m), len(seq_r)))
        print("The fastdtw error is {:.3f}.".format(score))
        avg = sum([len(seq_m), len(seq_r)]) / 2
        print("That is {:.3f} error per frame.".format(score / avg))
        print()

        delete = pp.align_sequences(path)
        for i in delete[0]:
            del seq_r[i]
        for i in delete[1]:
            del seq_m[i]
        seq_vis = [torch.stack([x[0, :, :], y[0, :, :]], dim=0) for x, y in zip(seq_r, seq_m)]
        stride = len(seq_vis) // 500 + 1
        visualisation.draw_sequence(seq_vis[::stride], "metrabs")


def test_synch_three(seq_len, radius, distance, dist_threshold, normalize, video_number):
    """
    Using three views of the same action sequences.
    """
    file_number = "{:04d}".format(video_number)
    path_metrabs = "/globalwork/sarandi/data/pkummd-more/estimates_38814fa9/"
    path_kinect = "/globalwork/datasets/pkummd/PKU_Skeleton_Renew/"
    path_stamp_dir = "/globalwork/datasets/pkummd/Train_Label_PKU_final"
    file_l = file_number + "-L.npy"
    file_m = file_number + "-M.npy"
    file_r = file_number + "-R.npy"
    video_l = pm.parse_tracklets(os.path.join(path_metrabs, file_l))
    video_m = pm.parse_tracklets(os.path.join(path_metrabs, file_m))
    video_r = pm.parse_tracklets(os.path.join(path_metrabs, file_r))
    pm.remove_false_positives(video_l, os.path.join(path_kinect, file_l[:-3] + "txt"))
    pm.remove_false_positives(video_m, os.path.join(path_kinect, file_m[:-3] + "txt"))
    pm.remove_false_positives(video_r, os.path.join(path_kinect, file_r[:-3] + "txt"))
    edges = pm.get_metrabs_edges()
    pp.identify_cross_view(video_m, video_l, edges, [])
    pp.identify_cross_view(video_m, video_r, edges, [])

    dist_fn = lambda x, y: distance(x, y, dist_threshold)

    time_l, time_m, time_r, _ = pm.extract_good_stamps(path_stamp_dir, file_no=file_number)
    mod_time_l, mod_time_m, mod_time_r = parse_seq_len_mode(seq_len, time_l, time_m, time_r, video_l, video_m, video_r)

    for time_l, time_m, time_r in zip(mod_time_l, mod_time_m, mod_time_r):
        seq_l = video_l[time_l[0]:time_l[1]]
        seq_m = video_m[time_m[0]:time_m[1]]
        seq_r = video_r[time_r[0]:time_r[1]]

        # customized synchronise_views function
        inv_l = util.get_view_invariant_representation(seq_l, edges, [])
        inv_m = util.get_view_invariant_representation(seq_m, edges, [])
        inv_r = util.get_view_invariant_representation(seq_r, edges, [])

        score_lm, path_lm = fastdtw(inv_l, inv_m, radius=radius, dist=dist_fn)
        score_mr, path_mr = fastdtw(inv_m, inv_r, radius=radius, dist=dist_fn)
        score_rl, path_rl = fastdtw(inv_r, inv_l, radius=radius, dist=dist_fn)

        if normalize:
            # divide by the path length
            # => short paths / solutions are not preferred over long paths
            score_lm /= len(path_lm)
            score_mr /= len(path_mr)
            score_rl /= len(path_rl)

        m = min(score_lm, score_mr, score_rl)
        if score_lm == m:
            helper_trim(seq_l, seq_m, path_lm, seq_r, inv_r, edges, radius, dist_fn)
        elif score_mr == m:
            helper_trim(seq_m, seq_r, path_mr, seq_l, inv_l, edges, radius, dist_fn)
        elif score_rl == m:
            helper_trim(seq_r, seq_l, path_rl, seq_m, inv_m, edges, radius, dist_fn)

        seq_vis = [torch.stack([x[0, ...], y[0, ...], z[0, ...]], dim=0) for x, y, z in zip(seq_l, seq_m, seq_r)]
        stride = len(seq_vis) // 500 + 1
        visualisation.draw_sequence(seq_vis[::stride], "metrabs")


def test_synch_synthetic(radius, repetitions):
    """
    Using synthetic sequences / views.
    """
    path_metrabs = "/globalwork/sarandi/data/pkummd-more/estimates_38814fa9/"
    path_kinect = "/globalwork/datasets/pkummd/PKU_Skeleton_Renew/"
    file = "0107-M.npy"
    seq = pm.parse_tracklets(os.path.join(path_metrabs, file))
    pm.remove_false_positives(seq, os.path.join(path_kinect, file[:-3] + "txt"))

    rot = get_rot()
    edges = pm.get_metrabs_edges()

    start = 100
    ends = [130, 250, 600, 1600] # sequences usually between 30 and 150
                                 # other lenghts are for testing

    for end in ends:
        series1 = seq[start:(end - 1)]
        series1.extend([seq[end - 1]] * repetitions)

        series2 = [seq[start]] * repetitions
        series2.extend(seq[(start + 1):end])

        for frame in series2:
            temp = frame @ rot + torch.tensor([1, 2, -2])
            frame = temp

        inv_1 = util.get_view_invariant_representation(series1, edges, [])
        inv_2 = util.get_view_invariant_representation(series2, edges, [])
        score, path = fastdtw(inv_1, inv_2, radius=radius, dist=lambda x, y: util.angular_dist_min(x, y, 1))

        print("The sequences are {:d} frames long (with {:d} unique frames).".format(len(series1), end - start))
        print("The fastdtw error is {:.3f}.".format(score))
        print("That is {:.3f} error per frame.".format(score / len(series1)))

        delete = pp.align_sequences(path)
        for i in delete[0]:
            del series1[i]
        for i in delete[1]:
            del series2[i]

        if score > 0:
            seq_vis = [torch.cat([x, y], dim=0) for x, y in zip(series1, series2)]
            visualisation.draw_sequence(seq_vis[:200], "metrabs")
        else:
            print("Skipping visualisation because error is 0...")
        print()


if __name__ == "__main__":
    # test_synch_synthetic(10, 50)
    # test_synch_two("long", radius=10, distance=util.angular_distance, dist_threshold=30, video_number=182)
    test_synch_three("short", radius=10, distance=util.angular_distance, dist_threshold=30, normalize=True, video_number=225)
