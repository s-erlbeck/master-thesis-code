import os
from functools import reduce
from collections import Counter

import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import util


def parse_action_sequence(path_txt):
    """
    Parses a single txt file for its pose sequence.

    Args:
        path_txt: The path to the txt file.
    Returns:
        A list of 2x25x3 torch tensors (2 persons, 25 joints, metric space).
    """
    seq = []
    with open(path_txt, "r") as file:
        for line in file:
            if line is not None:
                a = torch.tensor([float(x) for x in line.split()])
                b = a.reshape(2, 25, 3)
                # kinect: x-axis goes left & y-axis goes up
                # usual practice: x-axis goes left & y-axis goes down
                # => x and y have same meaning as in pixels
                b[:, :, :2] *= -1
                seq.append(b)
    return seq


def check_continuity(seq):
    """
    Compute identification within a camera view, i.e. such that pose 1 always belongs to pose 1.
    Does not work with sequence 107 because of a rare Kinect false positive.

    Args:
        seq: A sequence of poses (complete video).
    """
    # obtain poses from very first frame
    prev = seq[0]

    for frame in range(1, len(seq)):
        # if both persons are not visible, skip
        # reason: both "prev" and "current" should contain at least one pose for reliable matching
        if torch.sum(torch.abs(seq[frame])) == 0:
            continue

        # compare current poses to previous poses via hungarian matching
        matches, _, _ = util.hungarian_match(prev, seq[frame], dist_fn=lambda x, y: torch.mean(torch.norm(x - y, dim=1)))

        # build correct order
        seq[frame] = torch.stack([x[1] for x in matches], dim=0)
        prev = seq[frame]


def identify_cross_view(seq_m, seq_other, edges, unwanted):
    """
    Given two views, each consistent w.r.t. person order, this function computes cross-view identification.
    The other sequence will get changed if the person order is found to mismatch.

    Args:
        seq_m: The mid view video.
        seq_other: The other video where person order might change.
        edges: The kinematic tree of the poses.
        unwanted: The indices of the joints which are too noisy for angular features.
    """
    inv_reference = util.get_view_invariant_representation(seq_m, edges, unwanted)
    inv_normal = util.get_view_invariant_representation(seq_other, edges, unwanted)
    # create array where person 0 and person 1 are swapped
    inv_swapped = np.empty_like(inv_normal)
    half = inv_normal.shape[1] // 2
    inv_swapped[:, :half] = inv_normal[:, half:]
    inv_swapped[:, half:] = inv_normal[:, :half]

    dist_normal, _ = fastdtw(inv_reference, inv_normal, radius=10, dist=lambda x, y: util.angular_distance(x, y, 30))
    dist_swapped, _ = fastdtw(inv_reference, inv_swapped, radius=10, dist=lambda x, y: util.angular_distance(x, y, 30))

    # if swapped has lower score, swap persons in every frame
    if dist_swapped < dist_normal:
        swap = torch.LongTensor([1, 0])
        for frame in range(len(seq_other)):
            temp = seq_other[frame][swap]
            seq_other[frame] = temp
        print("Cross-View mismatch has been fixed.")


def disassemble_sequence(video, path_pose, path_stamp_dir=None):
    """
    Takes a parsed pose sequence and disassembles the video into a list of single action sequences.
    Computation is based on the ground truth action labels provided with PKUMMD.

    Args:
        video: The parsed pose sequence itself.
        path_pose: The path to the file containing the parsed pose video.
        path_stamp_dir: The path to the directory of the time stamps. If None, it is searched in
            "some_path/../Train_Label_PKU_final/0002-L.txt" where "some_path" is the directory of path_pose.
    Returns:
        A list of lists of (2 x 25 x 3) torch tensors.
    """
    path, file = os.path.split(path_pose)
    file = file[:(-3)] + "txt"
    if path_stamp_dir is None:
        path_labels = os.path.join(path, "../Train_Label_PKU_final", file)
    else:
        path_labels = os.path.join(path_stamp_dir, file)

    res = []
    with open(path_labels, "r") as file:
        for line in file:
            _, start, end, _ = line.split(sep=",")
            start = int(start)
            end = int(end)
            action = video[start:(end + 1)]
            res.append(action)
    return res


def synchronise_views(seq_l, seq_m, seq_r, edges, unwanted):
    """
    Given three views, this function computes how to synchronise these views.
    If the framerate of one Kinect is too high, those poses are deleted inplace.

    Args:
        seq_l: The left view.
        seq_m: The middle view.
        seq_r: The right view.
        edges: The kinematic tree of the poses.
        unwanted: The list of joints to ignore as features.
    """
    # combining pairwise alignment to global alignment: https://stackoverflow.com/questions/5813859
    inv_l = util.get_view_invariant_representation(seq_l, edges, unwanted)
    inv_m = util.get_view_invariant_representation(seq_m, edges, unwanted)
    inv_r = util.get_view_invariant_representation(seq_r, edges, unwanted)

    score_lm, path_lm = fastdtw(inv_l, inv_m, radius=10, dist=lambda x, y: util.angular_distance(x, y, 30))
    score_mr, path_mr = fastdtw(inv_m, inv_r, radius=10, dist=lambda x, y: util.angular_distance(x, y, 30))
    score_rl, path_rl = fastdtw(inv_r, inv_l, radius=10, dist=lambda x, y: util.angular_distance(x, y, 30))

    # normalize with path length so that long synchronisations are not penalized
    score_lm /= len(path_lm)
    score_mr /= len(path_mr)
    score_rl /= len(path_rl)

    m = min(score_lm, score_mr, score_rl)
    if score_lm == m:
        helper_trim(seq_l, seq_m, path_lm, seq_r, inv_r, edges, unwanted)
    elif score_mr == m:
        helper_trim(seq_m, seq_r, path_mr, seq_l, inv_l, edges, unwanted)
    elif score_rl == m:
        helper_trim(seq_r, seq_l, path_rl, seq_m, inv_m, edges, unwanted)


def helper_trim(pair_0, pair_1, path_pair, foreign, foreign_inv, edges, unwanted):
    """
    Helper function for synchronise_views.
    Synchronises pair_0 and pair_1, then computes synchronisation of both with foreign.
    Deletes redundant frames inplace.

    Args:
        pair_0: The sequence which corresponds to the first argument of fastdtw of the most similar pair.
        pair_1: The sequence which corresponds to the second argument of fastdtw of the most similar pair.
        path_pair: The resulting path of fastdtw on the most similar pair.
        foreign: The sequence which is not part of the most similar pair.
        foreign_inv: The invariant representation of the foreign sequence.
        edges: The kinematic tree for angle features.
        unwanted: The indices of the joints not considered for angle features.
    """
    delete = align_sequences(path_pair)
    for i in delete[0]:
        del pair_0[i]
    for i in delete[1]:
        del pair_1[i]
    inv_pair = util.get_view_invariant_representation(pair_0, edges, unwanted)
    _, path_final = fastdtw(inv_pair, foreign_inv, radius=10, dist=lambda x, y: util.angular_distance(x, y, 30))
    delete = align_sequences(path_final)
    for i in delete[0]:
        del pair_0[i]
        del pair_1[i]
    for i in delete[1]:
        del foreign[i]


def align_sequences(path):
    """
    Given a path returned by fastdtw, compute a maximal alignment between the sequences.
    If one frame matches several frames in the other sequence, only the frame in the middle is kept.
    Details are a bit complicated.

    Args:
        path: The path returned by fastdtw.

    Returns:
        Two lists of indices to delete from the sequences. The first list corresponds to the first argument of fastdtw.
        Indices are ordered descendingly so that deleting the first index does not invalidate other indices.
    """
    # unzip the path to two lists x and y
    x, y = map(list, zip(*path))

    # count how often each frame is repeated
    x_count = Counter(x)
    y_count = Counter(y)

    # init
    last_added = (-1, -1)
    matches = []

    # core routine
    # for explanation, think of a (m x n) grid graph, where (1, 1) is upper left corner
    # path is a path from (1, 1) to (m, n) using only the directions: RIGHT, DOWN, DIAGONAL (right and down)
    # word of advice: DO NOT try to simplify
    # I spent enough time failing at that
    for i in range(len(path) - 1):
        x, y = path[i]
        if x == path[i + 1][0]:
            # case: path goes DOWN
            if x_count[x] > 2:
                # if the down-going segment has length at least 3
                # then compute mid point and consider it as best interpolation
                if last_added[0] == x:
                    # segment already visited, therefore nothing to do
                    continue
                else:
                    # segment not visited yet, add mid point to matches
                    mid = (x_count[x] - 1) // 2
                    last_added = path[i + mid]
                    matches.append(last_added)
            else:
                # the down-going segment consists of only two vertices
                # therefore greedily add first vertex of that segment if possible
                # second vertex goes RIGHT or DIAGONAL and is handled in next iteration
                if last_added[1] != y:
                    last_added = (x, y)
                    matches.append(last_added)
        elif y == path[i + 1][1]:
            # case: path goes RIGHT
            if y_count[y] > 2:
                # if the right-going segment has length at least 3
                # then compute mid point and consider it as best interpolation
                if last_added[1] == y:
                    # segment already visited, therefore nothing to do
                    continue
                else:
                    # segment not visited yet, add mid point to matches
                    mid = (y_count[y] - 1) // 2
                    last_added = path[i + mid]
                    matches.append(last_added)
            else:
                # the right-going segment consists of only two vertices
                # therefore greedily add first vertex of that segment if possible
                # second vertex goes DOWN or DIAGONAL and is handled in next iteration
                if last_added[0] != x:
                    last_added = (x, y)
                    matches.append(last_added)
        else:
            # case: path goes DIAGONAL
            # therefore greedily add current vertex if possible
            if last_added[0] != x and last_added[1] != y:
                last_added = (x, y)
                matches.append(last_added)

    # check if last element of path can be added without conflict
    if last_added[0] != path[-1][0] and last_added[1] != path[-1][1]:
        last_added = path[-1]
        matches.append(last_added)

    # obtain lists of those x-frames and y-frames which are going to be kept
    matched_x, matched_y = zip(*matches)

    # store indices of all frames which should be deleted
    x_len, y_len = path[-1]
    delete = [[], []]
    for x in range(x_len + 1):
        if x not in matched_x:
            delete[0].insert(0, x)  # insert for descending order
    for y in range(y_len + 1):
        if y not in matched_y:
            delete[1].insert(0, y)  # insert for descending order
    return delete


def calibrate(seq_l, seq_m, seq_r):
    """
    Transforms left and right view to middle view coordinates via Procrustes.
    Expects sequence of actions.

    Args:
        seq_l: The left view, which is transformed inplace.
        seq_m: The middle view, which is the reference view.
        seq_r: The right view, which ist transformed inplace.
    """
    seq_l_pts, seq_m_pts, seq_r_pts = obtain_all_points(seq_l, seq_m, seq_r)

    R_l, t_l = util.ransac_rigid(seq_l_pts, seq_m_pts)
    R_r, t_r = util.ransac_rigid(seq_r_pts, seq_m_pts)

    # add appropriate dimensions for batch processing
    t_l = t_l.view(1, 1, 3)
    t_r = t_r.view(1, 1, 3)

    # apply rigid transform on all actions
    for i in range(len(seq_m)):
        # rotation via batched torch.matmul
        # due to the order of dimensions (2 x 25 x 3), the rotation matrix needs to be the second factor!
        # use: x^T * R^T = (Rx)^T
        seq_l[i][:] = map(lambda x: torch.matmul(x, R_l.T), seq_l[i])
        seq_r[i][:] = map(lambda x: torch.matmul(x, R_r.T), seq_r[i])

        # do not translate invalid poses (= zero matrix)
        for frame in range(len(seq_m[i])):
            for person in range(2):
                if torch.sum(seq_l[i][frame][person, :, :]) != 0:
                    seq_l[i][frame][person, :, :] = seq_l[i][frame][person, :, :] + t_l
                if torch.sum(seq_r[i][frame][person, :, :]) != 0:
                    seq_r[i][frame][person, :, :] = seq_r[i][frame][person, :, :] + t_r


def obtain_all_points(seq_l, seq_m, seq_r):
    """
    Transforms 3 ground truth sequences to 3 lists of 3D points for RANSAC Procrustes.
    If a person is not visible in all three views, these points are not used in RANSAC.

    Args:
        seq_l: The first view.
        seq_m: The second view.
        seq_r: The third view.
    Returns:
        3 Torch tensors of size (3 x N)
    """
    res_l = []
    res_m = []
    res_r = []

    for action_id in range(len(seq_m)):
        for frame_id in range(len(seq_m[action_id])):
            # if person 0 visible in any view, use these points for estimation
            if torch.sum(torch.abs(seq_l[action_id][frame_id][0, :, :])) > 0 \
                    or torch.sum(torch.abs(seq_m[action_id][frame_id][0, :, :])) > 0 \
                    or torch.sum(torch.abs(seq_r[action_id][frame_id][0, :, :])) > 0:
                res_l.extend(seq_l[action_id][frame_id][0, :, :])
                res_m.extend(seq_m[action_id][frame_id][0, :, :])
                res_r.extend(seq_r[action_id][frame_id][0, :, :])
            # if person 1 visible in any view, use these points for estimation
            if torch.sum(torch.abs(seq_l[action_id][frame_id][1, :, :])) > 0 \
                    or torch.sum(torch.abs(seq_m[action_id][frame_id][1, :, :])) > 0 \
                    or torch.sum(torch.abs(seq_r[action_id][frame_id][1, :, :])) > 0:
                res_l.extend(seq_l[action_id][frame_id][1, :, :])
                res_m.extend(seq_m[action_id][frame_id][1, :, :])
                res_r.extend(seq_r[action_id][frame_id][1, :, :])

    return torch.stack(res_l, dim=1), torch.stack(res_m, dim=1), torch.stack(res_r, dim=1)


def check_bone_length(seq, path_mean, edges, rel_difference=2, abs_difference=0.15, verbose=False, two_poses=True):
    """
    Takes  a synchronised sequence and checks for plausible bone lengths.
    Implausible poses are set to NaN (to differentiate from a person not being visible).
    Implausible means that the bone length differs by x cm and is y times larger / smaller.

    Args:
        seq: The synchronised sequence which is to be checked. The sequence is changed by this function.
        path_mean: The path to the mean bone length txt.
        edges: The kinematic tree used.
        rel_difference: The factor of how much larger / smaller the bones are allowed to be compared to mean.
            Default of 2 means that bones are discarded if twice or half as large as the average.
        abs_difference: The allowed absolute difference between bones and mean. Default is 15 cm.
        verbose: If verbose is True, all insane bones are printed to the console. Default is false.
        two_poses: If two_poses is True (which is default), the method expects to find two poses (one may be zero values only).
    """
    mean_bones = torch.tensor(np.loadtxt(path_mean))
    discarded = 0

    for frame in seq:
        bone_vectors = util.get_bone_vectors(frame, edges)
        bone_length = torch.norm(bone_vectors, dim=2)
        lower = bone_length < mean_bones / rel_difference
        upper = bone_length > mean_bones * rel_difference
        abs_dif = torch.abs(bone_length - mean_bones) > abs_difference
        insane = torch.logical_and(abs_dif, torch.logical_or(lower, upper))
        # set implausible poses to NaN
        # allows to differentiate from "person not visible" (represented by a zero-tensor)
        if sum(bone_length[0, :]) > 0 and insane[0, :].any():
            if verbose:
                insane_bone = torch.nonzero(insane[0, :], as_tuple=False)[0]
                length = bone_length[0, insane_bone].item()
                print(str(edges[insane_bone]) + " " + str(length))
            frame[0, :, :] = float("nan")
            discarded += 1
        if (two_poses or len(bone_length) > 1) and sum(bone_length[1, :]) > 0 and insane[1, :].any():
            if verbose:
                insane_bone = torch.nonzero(insane[1, :], as_tuple=False)[0]
                length = bone_length[1, insane_bone].item()
                print(str(edges[insane_bone]) + " " + str(length))
            frame[1, :, :] = float("nan")
            discarded += 1

    print(str(discarded) + " poses were discarded due to bone length.")


def fuse_ground_truth_geo_median(seq_l, seq_m, seq_r, pelvis_id, times=None):
    """
    Given three synchronised and calibrated views of a single action, fuse them into a coherent pose sequence.
    This version uses the geometric median implementation.

    Args:
        seq_l: The left view.
        seq_m: The middle view.
        seq_r: The right view.
        pelvis_id: The ID of the pelvis joint of the kinematic tree.
        times: If provided, time stamps will be decremented for deleted frames.
    Returns:
        A single list of (#person x 25 x 3) tensors.
    """
    res = [[], []]
    for frame in range(len(seq_m)):
        for person in range(2):
            candidates = []
            # collect detected poses (visible and with plausible bone lengths)
            if seq_l[frame][person, 0, 0] == seq_l[frame][person, 0, 0] and torch.sum(seq_l[frame][person, :, :]) != 0:
                candidates.append(seq_l[frame][person, :, :])
            if seq_m[frame][person, 0, 0] == seq_m[frame][person, 0, 0] and torch.sum(seq_m[frame][person, :, :]) != 0:
                candidates.append(seq_m[frame][person, :, :])
            if seq_r[frame][person, 0, 0] == seq_r[frame][person, 0, 0] and torch.sum(seq_r[frame][person, :, :]) != 0:
                candidates.append(seq_r[frame][person, :, :])
            # aggregate the detections
            if len(candidates) == 0:
                # nothing found for that particular person (may still show other person)
                res[person].append(None)
            elif len(candidates) == 1:
                # no choice
                res[person].append(candidates[0])
            else:
                fused = util.compute_median_pose([x.numpy() for x in candidates], pelvis_id)
                res[person].append(torch.tensor(fused))

    return post_process(res[0], res[1], times)


def fuse_ground_truth_velocity(seq_l, seq_m, seq_r):
    """
    Given three synchronised and calibrated views of a single action, fuse them into a coherent pose sequence.
    This version uses a constant velocity assumption of each joint to smooth temporarily.

    Args:
        seq_l: The left view.
        seq_m: The middle view.
        seq_r: The right view.
    Returns:
        A single list of (#person x 25 x 3) tensors.
    """
    res = [[], []]
    velocity = torch.zeros_like(seq_m[0])
    for frame in range(len(seq_m)):
        for person in range(2):
            candidates = []
            # collect detected poses (visible and with plausible bone lengths)
            # mid view first -> used in tie-breaker because mid view has best quality
            if seq_m[frame][person, 0, 0] == seq_m[frame][person, 0, 0] and torch.sum(seq_m[frame][person, :, :]) != 0:
                candidates.append(seq_m[frame][person, :, :])
            if seq_l[frame][person, 0, 0] == seq_l[frame][person, 0, 0] and torch.sum(seq_l[frame][person, :, :]) != 0:
                candidates.append(seq_l[frame][person, :, :])
            if seq_r[frame][person, 0, 0] == seq_r[frame][person, 0, 0] and torch.sum(seq_r[frame][person, :, :]) != 0:
                candidates.append(seq_r[frame][person, :, :])
            # aggregate the detections
            if len(candidates) == 0:
                # nothing found for that particular person (may still show other person)
                res[person].append(None)
            elif len(candidates) == 1:
                # no choice
                if len(res[person]) == 0:
                    res[person].append(candidates[0])
                else:
                    momentum = candidates[0] - res[person][-1]
                    res[person].append(res[person][-1] + (velocity[person] + momentum) / 2.0)
                    velocity[person] = res[person][-1] - res[person][-2]
            elif len(candidates) == 2:
                if len(res[person]) == 0:
                    # no previous frame => use mean
                    res[person].append(0.5 * (candidates[0] + candidates[1]))
                else:
                    # previous frame => trust the pose which is mor similar to previous pose
                    dist_0 = torch.mean(torch.norm(candidates[0] - res[person][-1], dim=1))
                    dist_1 = torch.mean(torch.norm(candidates[1] - res[person][-1], dim=1))
                    if dist_0 < dist_1:
                        momentum = candidates[0] - res[person][-1]
                        res[person].append(res[person][-1] + (velocity[person] + momentum) / 2.0)
                        velocity[person] = res[person][-1] - res[person][-2]
                    else:
                        momentum = candidates[1] - res[person][-1]
                        res[person].append(res[person][-1] + (velocity[person] + momentum) / 2.0)
                        velocity[person] = res[person][-1] - res[person][-2]
            elif len(candidates) == 3:
                # for 3 persons, use the mean of the more similar pair
                dist_01 = torch.mean(torch.norm(candidates[0] - candidates[1], dim=1))
                dist_02 = torch.mean(torch.norm(candidates[0] - candidates[2], dim=1))
                dist_12 = torch.mean(torch.norm(candidates[1] - candidates[2], dim=1))

                m = min(dist_01, dist_02, dist_12)
                if dist_01 == m:
                    temp = candidates[0] + candidates[1]
                elif dist_02 == m:
                    temp = candidates[0] + candidates[2]
                elif dist_12 == m:
                    temp = candidates[1] + candidates[2]

                if len(res[person]) == 0:
                    res[person].append(0.5 * temp)
                else:
                    momentum = temp / 2.0 -  res[person][-1]
                    res[person].append(res[person][-1] + (velocity[person] + momentum) / 2.0)
                    velocity[person] = res[person][-1] - res[person][-2]

            else:
                raise RuntimeError("This should not have occurred.")

    return post_process(res[0], res[1])


def post_process(person_0, person_1, times=None):
    """
    Helper function which skips frames with no sane pose present.
    Also shrinks the person-dimension if the video ony shows a single person.

    Args:
        person_0: The sequence of person 0 as computed by fuse_ground_truth internally.
        person_1: The sequence of person 1 as computed by fuse_ground_truth internally.
        times: If provided, time stamps will be decremented for deleted frames.
    Returns:
        A single list of (#person x 25 x 3) tensors.
    """
    # check if persons visible at some point
    person_0_is_none = reduce(lambda x, y: x and y is None, person_0, True)
    person_1_is_none = reduce(lambda x, y: x and y is None, person_1, True)

    if person_0_is_none and person_1_is_none:
        raise ValueError("Both sequences are empty.")
    elif person_0_is_none:
        # only person 1 is present
        res = []
        for x in person_1:
            if x is not None:
                res.append(x[None, :, :])
            elif times is not None:
                for i in range(len(times)):
                    if times[i] > len(res):
                        # times[i] is the first index (!) that belongs to the next subsequence
                        # we are currently discarding the frame with index len(res)
                        # if these two indices are the same, we delete the very first pose of the next subsequence
                        # => next subsequence starts at the same time but ends earlier
                        # => use > instead of ==
                        times[i] -= 1
        return res
    elif person_1_is_none:
        # only person 0 is present
        res = []
        for x in person_0:
            if x is not None:
                res.append(x[None, :, :])
            elif times is not None:
                for i in range(len(times)):
                    if times[i] > len(res):
                        times[i] -= 1
        return res
    else:
        res = []
        for frame in range(len(person_0)):
            if person_0[frame] is None and person_1[frame] is None:
                # skip frame without ground truth
                if times is not None:
                    for i in range(len(times)):
                        if times[i] > len(res):
                            times[i] -= 1
            elif person_0[frame] is None:
                res.append(torch.stack([torch.zeros_like(person_1[frame]), person_1[frame]], dim=0))
            elif person_1[frame] is None:
                res.append(torch.stack([person_0[frame], torch.zeros_like(person_0[frame])], dim=0))
            else:
                res.append(torch.stack([person_0[frame], person_1[frame]], dim=0))

    return res
