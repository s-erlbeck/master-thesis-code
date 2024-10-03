import os
import sys
import collections

import torch
import numpy as np
import scipy

import util
import preprocess_metrabs as pm
import preprocess_pku_mmd as pp


def parse_sequence(path):
    """
    Parses an original NTU RGB+D kinect file.
    Method not really needed anymore, but useful in case one needs to understand the file format.

    Args:
        path: The path to the kinect data file.
    Returns:
        A list of torch tensors.
    """
    res = []

    with open(path) as handle:
        # read number of frames
        line = handle.readline()
        length = int(line)

        for frame in range(length):
            poses = torch.zeros(2, 25, 3)

            # read number of detections
            line = handle.readline()
            num_detections = int(line)
            if num_detections > 2:
                print(path)
                print("In frame", frame, "there were", num_detections, "detections.")

            for detection in range(num_detections):
                # skip a line containing some meta info
                _ = handle.readline()

                # read number of joints
                line = handle.readline()
                num_joints = int(line)
                if num_joints != 25:
                    print(path)
                    print("In frame", frame, "there were", num_joints, "joints.")

                # parse all joints
                for i in range(num_joints):
                    joint = handle.readline().split(" ")
                    location = torch.FloatTensor([float(x) for x in joint[:3]])
                    if detection < 2:
                        poses[detection, i, :] = location
                        # other detections need to be stored as well!

            # append to sequence
            poses[:, :, :2] *= -1
            res.append(poses)

    return res


def get_expected_num_people(file_name):
    """
    Computes the expected number of people based on the action ID.

    Args:
        file_name: The file name of the sequence to be analysed.
    Returns:
        The number of people expected to be visible.
    """
    file_name = os.path.splitext(file_name)[0]
    if file_name[-4] != "A":
        raise ValueError("Filename has unexpected format:", file_name)

    label = int(file_name[-3:])
    if 50 <= label <= 60:
        # mutual action in NTU RGB+D 60
        return 2
    elif 106 <= label <= 120:
        # mutual action in NTU RGB+D 120
        return 2
    else:
        return 1


def parse_tracklets_with_reid(path, seq_name):
    """
    Parses the Metrabs detections of all three camera views of the same action sequence.
    Always expects tracklet of camera "001" as argument.

    Args:
        path: The path to base directory of Metrabs detections.
        seq_name: The name of the sequence.
    Returns:
        The left, middle and right view of the action sequence.
    """
    if seq_name[-17] != "1":
        raise ValueError("Expects camera 1 as argument.")
    path_l = os.path.join(path, seq_name)
    path_m = os.path.join(path, seq_name[:(-17)] + "2" + seq_name[(-16):])
    path_r = os.path.join(path, seq_name[:(-17)] + "3" + seq_name[(-16):])

    video_l = parse_single_view(path_l)
    video_m = parse_single_view(path_m)
    video_r = parse_single_view(path_r)

    return video_l, video_m, video_r


def parse_single_view(path):
    """
    Parses the Metrabs detections of a single view.

    Args:
        path: The path to the file.
    Returns:
        A list of torch tensors in meters.
    """
    with np.load(path, allow_pickle=True) as array:
        data = array["poses"]

    # if number of people varies, the arrays are stored as object arrays
    if not np.issubdtype(data.dtype, np.floating):
        # only known reason: one person is not always visible
        num_people = get_expected_num_people(path)

        if num_people != 2:
            # this case never occurs, so it is not handled
            raise NotImplementedError(path + " contains object arrays. Could not determine why.")

        # allocate new matrix
        data_new = np.empty((data.shape[0], num_people,
                             data[0].shape[1], data[0].shape[2]))  # data is list, data[0] pose matrix

        # get first frame where 2 people are visible
        first_idx = min([x for x in range(len(data)) if data[x].shape[0] == 2])
        prev = data[first_idx]  # no clone since changes are on reference "prev", not its data

        # pad last known pose if person goes missing
        for i in range(len(data)):
            if data[i].shape[0] == 2:
                prev = data[i]
                data_new[i, :, :, :] = data[i]
            else:
                # determine which pose is missing
                diff = prev[:, :24, :] - data[i][:, :24, :]
                dist = np.linalg.norm(diff, axis=2)
                error = np.sum(dist, axis=1)

                # now pad the other pose into the frame
                if error[0] < error[1]:
                    data_new[i, 0, :, :] = data[i][0, :, :]
                    data_new[i, 1, :, :] = prev[1, :, :]
                else:
                    data_new[i, 0, :, :] = prev[0, :, :]
                    data_new[i, 1, :, :] = data[i][0, :, :]

        data = data_new

    # convert to meters, cast to float
    video = [torch.tensor(x[:, :24, :] / 1000, dtype=torch.float) for x in data]

    return video


def compute_mean_bone_length():
    """
    Iterates over NTU-RGB+D Metrabs detections and computes average bone lengths.
    """
    path_source = "/globalwork/sarandi/data/NTU_RGBD-more/tracks_reid_65ed967/"
    path_res = "/work/erlbeck/datasets/nturgbd_enhanced/"
    edges = pm.get_metrabs_edges()
    util.get_mean_bone_length(path_source, edges, parse_single_view, path_res)


def align_feet(feet):
    """
    Computes the rotation to align ground plane to xz-plane.

    Args:
        feet: The Nx3 numpy array of feet locations.
    Returns:
        The rotation matrix to align them to the xz-plane
    """
    # trim to reasonable value at center
    feet[feet > 4] = 4
    feet[feet < -4] = -4
    feet_through_origin = feet - np.mean(feet, axis=0, keepdims=True)

    # project feet onto xz-plane
    target = np.array(feet_through_origin, copy=True)
    target[:, 1] = 0

    # scale targets to original magnitude
    target_norms = np.linalg.norm(target, axis=1, keepdims=True)
    feet_norms = np.linalg.norm(feet_through_origin, axis=1, keepdims=True)
    target = (target / target_norms) * feet_norms

    # compute alignment
    rot, _ = scipy.spatial.transform.Rotation.align_vectors(target, feet_through_origin)
    # set yaw and roll to 0
    corrected = scipy.spatial.transform.Rotation.from_euler("x", rot.as_euler("xyz")[0])
    return corrected.as_matrix()


def align_all(path):
    """
    Calls align_feet for each of the 32 set-up using multiple sequences.

    Returns:
        The 32 rotation matrices.
    """
    feet = collections.defaultdict(list)
    rotations = np.empty((32, 3, 3))

    # extract all feet from suitable actions
    for file in os.listdir(path):
        action = int(file[(-6):(-3)])
        setup = int(file[1:4])

        # these actions include walking => robust estimation
        if action != 60 and action != 116:
            continue

        # reshape the feet to Nx3
        data = torch.load(os.path.join(path, file)).view(-1, 24, 3)
        data = data[:, [pm.get_metrabs_joint_by_name("l_toe"), pm.get_metrabs_joint_by_name("r_toe")], :]
        data = data.view(-1, 3)

        # add points to the correct set up
        feet[setup].append(data)

    # estimate all ground plane alignments
    for i in range(1, 33):
        points = feet[i]

        # should not happen
        if len(points) == 0:
            raise NotImplementedError("No videos with specified action id found for setup " + str(i))

        # concat points and compute rotation
        accumulated = torch.cat(points, dim=0).numpy()
        rotations[i - 1, :, :] = align_feet(accumulated)

    return rotations


def preprocess_first():
    """
    Performs the first step of preprocessing.
    Namely, it ensures that pose 0 shows the same person in each frame and for each view.
    """
    torch.manual_seed(42)
    path_metrabs = "/globalwork/sarandi/data/NTU_RGBD-more/tracks_reid_65ed967/"
    path_store = "/work/erlbeck/datasets/nturgbd_enhanced/tracklets/"

    for _, _, files in os.walk(path_metrabs):
        for file in files:
            # deal with all three views during same iteration
            if file[-17] != "1":
                continue
            print(file)

            # parsing the metrabs tracks with re-identification tracking
            video_l, video_m, video_r = parse_tracklets_with_reid(path_metrabs, file)

            if video_l[0].size()[0] == video_m[0].size()[0] == video_r[0].size()[0] == 1:
                # add zero poses to single-person videos (compatibility reasons)
                video_l = [torch.cat([x, torch.zeros_like(x)], dim=0) for x in video_l]
                video_m = [torch.cat([x, torch.zeros_like(x)], dim=0) for x in video_m]
                video_r = [torch.cat([x, torch.zeros_like(x)], dim=0) for x in video_r]
            elif video_l[0].size()[0] == video_m[0].size()[0] == video_r[0].size()[0] == 2:
                # two person found => frame-to-frame matching (due to rare id swaps)
                pm.check_continuity_reid(video_l)
                pm.check_continuity_reid(video_m)
                pm.check_continuity_reid(video_r)
            else:
                # should never occur (does not occur either)
                print(video_l.size()[0])
                print(video_m.size()[0])
                print(video_r.size()[0])
                raise NotImplementedError("Missing detections in some views.")

                # now match across different camera views
                edges = pm.get_metrabs_edges()
                pp.identify_cross_view(video_m, video_l, edges, [])
                pp.identify_cross_view(video_m, video_r, edges, [])

            # store intermediate results
            pm.numpy_save_sequence(video_l, os.path.join(path_store, file[:(-4)] + ".npy"))
            pm.numpy_save_sequence(video_m, os.path.join(path_store, file[:(-17)] + "2" + file[(-16):(-4)] + ".npy"))
            pm.numpy_save_sequence(video_r, os.path.join(path_store, file[:(-17)] + "3" + file[(-16):(-4)] + ".npy"))


def preprocess_second():
    """
    Performs the second step of preprocessing.
    Namely, it synchronises the different views and compute extrinsics.
    """
    torch.manual_seed(42)
    path_load = "/work/erlbeck/datasets/nturgbd_enhanced/tracklets/"
    path_store = "/work/erlbeck/datasets/nturgbd_enhanced/calibrated/"

    for _, _, files in os.walk(path_load):
        for file in files:
            # deal with all three views during same iteration
            if file[-17] != "1":
                continue
            print(file)

            # load
            video_l = pm.numpy_load_sequence(os.path.join(path_load, file))
            video_m = pm.numpy_load_sequence(os.path.join(path_load, file[:(-17)] + "2" + file[(-16):]))
            video_r = pm.numpy_load_sequence(os.path.join(path_load, file[:(-17)] + "3" + file[(-16):]))

            # synchronise the sequences
            edges = pm.get_metrabs_edges()
            min_len = min(len(video_l), len(video_m), len(video_r))
            pp.synchronise_views(video_l, video_m, video_r, edges, [])
            if min_len != len(video_m):
                print("Discarded", len(video_m) - min_len, "frames during synch!")

            # compute calibration
            # method expects multiple actions, so we wrap the videos in lists
            pp.calibrate([video_l], [video_m], [video_r])

            # store
            pm.numpy_save_sequence(video_l, os.path.join(path_store, file))
            pm.numpy_save_sequence(video_r, os.path.join(path_store, file[:(-17)] + "2" + file[(-16):]))
            pm.numpy_save_sequence(video_m, os.path.join(path_store, file[:(-17)] + "3" + file[(-16):]))


def preprocess_third():
    """
    Performs the third step of preprocessing.
    Namely, it discards unrealistic poses and fuses the different views.
    """
    torch.manual_seed(42)
    path_load = "/work/erlbeck/datasets/nturgbd_enhanced/calibrated/"
    path_store = "/work/erlbeck/datasets/nturgbd_enhanced/combined/"
    path_mean = "/work/erlbeck/datasets/nturgbd_enhanced/mean_bone_length.txt"

    for _, _, files in os.walk(path_load):
        for file in files:
            # deal with all three views during same iteration
            if file[-17] != "1":
                continue
            print(file)

            # load
            video_l = pm.numpy_load_sequence(os.path.join(path_load, file))
            video_m = pm.numpy_load_sequence(os.path.join(path_load, file[:(-17)] + "2" + file[(-16):]))
            video_r = pm.numpy_load_sequence(os.path.join(path_load, file[:(-17)] + "3" + file[(-16):]))

            # remove implausible bone lengths
            edges = pm.get_metrabs_edges()
            pp.check_bone_length(video_l, path_mean, edges, verbose=False)
            pp.check_bone_length(video_m, path_mean, edges, verbose=False)
            pp.check_bone_length(video_r, path_mean, edges, verbose=False)

            # fuse ground truths
            res = pp.fuse_ground_truth_geo_median(video_l, video_m, video_r, pm.get_metrabs_joint_by_name("pelvi"))
            pm.pickle_save(res, os.path.join(path_store, file[:(-4)] + ".pkl"))


def preprocess_fourth():
    """
    Performs the fourth step of preprocessing.
    Namely, it performs a final filtering after the fusion and stores them for the learning.
    Filtering affects zero poses, frozen poses and insane bone lengths.
    """
    torch.manual_seed(42)
    path_load = "/work/erlbeck/datasets/nturgbd_enhanced/combined/"
    path_store = "/work/erlbeck/datasets/nturgbd_enhanced/final/"
    path_mean = "/work/erlbeck/datasets/nturgbd_enhanced/mean_bone_length.txt"
    edges = pm.get_metrabs_edges()

    for _, _, files in os.walk(path_load):
        for file in files:
            print(file)
            seq = pm.pickle_load(os.path.join(path_load, file))

            # set implausible frames to 0
            previous = torch.clone(seq[0])  # clone because otherwise parts are overwritten
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
            pp.check_bone_length(seq, path_mean, edges, verbose=False, two_poses=False)

            # collect all bad frame ids for deletion
            bad_frames_succ = 0
            bad_frames_all = 0
            discard_frames = []
            for j, frame in enumerate(seq):
                sums = torch.sum(torch.abs(frame), dim=(1, 2))
                if (frame == frame).all() and (sums > 0).all():
                    if bad_frames_succ > 10:
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

            if len(seq) == 0:
                continue

            # store sequence
            tensor = torch.empty(len(seq), seq[0].size()[0], 24 * 3)
            for j, frame in enumerate(seq):
                tensor[j, :, :] = frame.view(-1, 24 * 3)
            torch.save(tensor, os.path.join(path_store, file[:(-4)] + ".pt"))


def preprocess_fifth():
    """
    Performs the fifth step of preprocessing.
    Namely, it computes the 32 ground plane alignments and applies them.
    """
    torch.manual_seed(42)
    path_load = "/work/erlbeck/datasets/nturgbd_enhanced/final/"
    path_store = "/work/erlbeck/datasets/nturgbd_enhanced/normalised/"

    rotations = align_all(path_load)

    for file in os.listdir(path_load):
        setup = int(file[1:4])
        original = torch.load(os.path.join(path_load, file))
        original = original.view(len(original), -1, 24, 3)
        rotated = original @ rotations[setup - 1, :, :].T
        rotated = rotated.view(rotated.size(0), rotated.size(1), 72)
        torch.save(rotated, os.path.join(path_store, file))


if __name__ == "__main__":
    # sys.stdout = open("log_preprocess_fifth_stage.txt", "w")
    # sys.stderr = sys.stdout
    # preprocess_fifth()
    pass
