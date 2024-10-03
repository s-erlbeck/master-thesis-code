import visualisation
from preprocess_pku_mmd import *


def show_sequence():
    """
    Visualises a single sequence.
    """
    # uncomment depending on the system this code is running on
    # path_dir = "/home/embreaded/Downloads/PKUMMD/Pose_Sequences/"
    path_dir = "/globalwork/datasets/pkummd/PKU_Skeleton_Renew/"
    file = "0017-R.txt"
    seq = parse_action_sequence(os.path.join(path_dir, file))
    seq = disassemble_sequence(seq, os.path.join(path_dir, file))
    for x in seq:
        visualisation.draw_sequence(x)


def test_continuity():
    """
    Tests the check_continuity function which swaps the order of people if inconsistencies are found within a view.
    """
    path_dir = "/home/embreaded/Downloads/PKUMMD/Pose_Sequences/"
    # path_dir = "/globalwork/datasets/pkummd/PKU_Skeleton_Renew/"
    file = "0016-M.txt"
    seq = parse_action_sequence(os.path.join(path_dir, file))
    check_continuity(seq)
    visualisation.draw_sequence(seq)


def benchmark_matching_features(use_angles=True):
    """
    Visualizes the result of frame-to-frame matching between persons within a fixed view.
    One can either use angles or joint positions as features.
    """
    path_dir = "/home/embreaded/Downloads/PKUMMD/Pose_Sequences/"
    # path_dir = "/globalwork/datasets/pkummd/PKU_Skeleton_Renew/"
    file = "0016-M.txt"
    seq = parse_action_sequence(os.path.join(path_dir, file))

    if use_angles:
        edges = util.get_kinect_edges()
        unwanted = [util.get_kinect_joint_by_name("hand_r"),
                    util.get_kinect_joint_by_name("hand_l"),
                    util.get_kinect_joint_by_name("wrist_r"),
                    util.get_kinect_joint_by_name("wrist_l")]
        inv_seq = util.get_view_invariant_representation(seq, edges, unwanted)
        prev = [inv_seq[0][:21], inv_seq[0][21:]]
    else:
        prev = [seq[0][0, :, :], seq[0][1, :, :]]

    for frame in range(1, len(seq)):
        if use_angles:
            current = [inv_seq[frame][:21], inv_seq[frame][21:]]
            matches, _, _ = util.hungarian_match(prev, current, dist_fn=lambda x, y: util.angular_distance(x, y, 50))
        else:
            current = [seq[frame][0, :, :], seq[frame][1, :, :]]
            matches, _, _ = util.hungarian_match(prev, current, dist_fn=lambda x, y: torch.mean(torch.norm(x - y, dim=1)))

        if [(x[0] == prev[i]).all() for i, x in enumerate(matches)] != [True, True]:
            raise RuntimeError("Bug detected with hungarian matching")

        # check if swap is necessary
        if (matches[0][1] == current[1]).all() and (matches[1][1] == current[0]).all():
            temp = seq[frame][torch.LongTensor([1, 0])]
            seq[frame] = temp
            current.reverse()
        prev = current

    visualisation.draw_sequence(seq)


def test_align():
    """
    Tests the alignment functionality to deal with the output of fastdtw.
    """
    path1 = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 6), (6, 6),
             (7, 7), (8, 7), (9, 7), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (11, 11)]
    x1 = list(range(12))
    y1 = list(range(12))

    path2 = [(0, 0), (0, 1), (1, 1)]
    x2 = list(range(2))
    y2 = list(range(2))

    path3 = [(i, i) for i in range(21)] + \
            [(20, 21), (20, 22), (20, 23), (20, 24), (20, 25), (20, 26), (20, 27),
             (21, 28), (21, 29), (21, 30), (22, 30), (22, 31), (22, 32), (23, 33)]
    x3 = list(range(24))
    y3 = list(range(34))

    path4 = [(i, i) for i in range(21)] + \
            [(20, 21), (20, 22), (20, 23), (20, 24), (20, 25), (20, 26), (20, 27),
             (21, 28), (21, 29), (21, 30), (22, 30), (22, 31), (23, 32)]
    x4 = list(range(24))
    y4 = list(range(33))

    path = path1
    x = x1
    y = y1
    delete = align_sequences(path)
    for i in delete[0]:
        del x[i]
    for i in delete[1]:
        del y[i]
    print(x)
    print(y)


def test_mean_bone():
    """
    Tests whether mean bone length computation and bone vectors work properly.
    """
    # compute mean bone length
    path_root = "/home/embreaded/Downloads/PKUMMD/Pose_Sequences/"
    # path_root = "/globalwork/datasets/pkummd/PKU_Skeleton_Renew"
    mean, num = util.get_mean_bone_length(path_root, util.get_kinect_edges(), parse_action_sequence)
    print("Parsed " + str(num) + " poses.")

    # extract a reference pose
    pose_ref = parse_action_sequence(os.path.join(path_root, "0106-M.txt"))[0][0, :, :]
    bones_ref = util.get_bone_vectors(pose_ref, util.get_kinect_edges())

    # construct three skeletons:
    pose_mean_sized = torch.zeros(25, 3)  # a skeleton with average bone length at origin
    pose_translated = torch.zeros(25, 3)  # the original skeleton translated to origin
    pose_reconstructed = torch.zeros(25, 3)  # a reconstruction of the reference pose
    # translate pelvis of the reconstruction back to original position
    pose_reconstructed[0, :] = pose_ref[0, :]

    # build up the three new skeletons
    for i, edge in enumerate(util.get_kinect_edges()):
        # reconstruct original pose and translated pose
        pose_reconstructed[edge[1], :] = pose_reconstructed[edge[0], :] + bones_ref[0][i]
        pose_translated[edge[1], :] = pose_translated[edge[0], :] + bones_ref[0][i]
        # build average-sized skeleton
        vec = bones_ref[0][i] * mean[i] / torch.norm(bones_ref[0][i])
        pose_mean_sized[edge[1], :] = pose_mean_sized[edge[0], :] + vec

    # evaluate correct reconstruction:
    err = torch.norm(pose_ref - pose_reconstructed)
    print("Error between original pose and reconstruction:\n" + str(err.item()))
    err2 = torch.norm(pose_translated - pose_mean_sized)
    print("Frobenius norm between translated pose and average-sized pose:\n" + str(err2.item()))

    # visualization of two skeletons
    res = torch.stack((pose_reconstructed, pose_mean_sized), dim=0)
    visualisation.draw_still_poses(res, "kinect", "Average_bone_length_visualisation")


def test_preprocessing():
    """
    Can be called to test preprocessing functionality of the module parse_pkummd.
    """
    torch.manual_seed(42)

    # uncomment depending on the system this code is running on
    path_dir = "/home/embreaded/Downloads/PKUMMD/Pose_Sequences/"
    # path_dir = "/globalwork/datasets/pkummd/PKU_Skeleton_Renew/"
    file_no = "0225"

    path_l = os.path.join(path_dir, file_no + "-L.txt")
    path_m = os.path.join(path_dir, file_no + "-M.txt")
    path_r = os.path.join(path_dir, file_no + "-R.txt")

    print("Parsing... ")
    video_l = parse_action_sequence(path_l)
    video_m = parse_action_sequence(path_m)
    video_r = parse_action_sequence(path_r)
    print("Done")

    print("Tracking within each view... ")
    check_continuity(video_l)
    check_continuity(video_m)
    check_continuity(video_r)
    print("Done")

    print("Cross-view identification...")
    edges = util.get_kinect_edges()
    unwanted = [util.get_kinect_joint_by_name("hand_r"),
                util.get_kinect_joint_by_name("hand_l"),
                util.get_kinect_joint_by_name("wrist_r"),
                util.get_kinect_joint_by_name("wrist_l")]
    identify_cross_view(video_m, video_l, edges, unwanted)
    identify_cross_view(video_m, video_r, edges, unwanted)
    print("Done")

    print("Disassembling into actions...")
    video_l = disassemble_sequence(video_l, path_l)
    video_m = disassemble_sequence(video_m, path_m)
    video_r = disassemble_sequence(video_r, path_r)
    print("Done")

    print("Synchronizing...")
    for i in range(len(video_m)):
        seq_l = video_l[i]
        seq_m = video_m[i]
        seq_r = video_r[i]
        # synchronise 3 different views of the same person
        m = min(len(seq_l), len(seq_m), len(seq_r))
        # print(len(seq_l), len(seq_m), len(seq_r))
        synchronise_views(seq_l, seq_m, seq_r, edges, unwanted)
        # print(len(seq_l), len(seq_m), len(seq_r))
        print(str(m - len(seq_m)) + " frames discarded.")
    print("Done")

    # calibrate
    print("Calibrating...")
    calibrate(video_l, video_m, video_r)
    print("Done")

    for i in range(len(video_m)):
        seq_l = video_l[i]
        seq_m = video_m[i]
        seq_r = video_r[i]

        # remove implausible bone lengths
        print("Action " + str(i) + ":")
        check_bone_length(seq_r, os.path.join(path_dir, "../mean_bone_length_kinect.txt"), edges, verbose=False)
        check_bone_length(seq_l, os.path.join(path_dir, "../mean_bone_length_kinect.txt"), edges, verbose=False)
        check_bone_length(seq_m, os.path.join(path_dir, "../mean_bone_length_kinect.txt"), edges, verbose=False)

        # old code to visualize how well synchronisation worked:
        res = []
        for j in range(len(seq_m)):
            res.append(torch.stack([seq_l[j][0], seq_m[j][0], seq_r[j][0]], dim=0))
        visualisation.draw_sequence(res)

        # fuse ground truths
        print("Fusing sequences...")
        res = fuse_ground_truth_geo_median(seq_l, seq_m, seq_r, util.get_kinect_joint_by_name("spinebase"))
        print("Done")
        visualisation.draw_sequence(res)


def test_rigid():
    """
    Tests the rigid transformation estimation (Procrustes alignment).
    """
    src = torch.rand(3, 100)

    R1 = torch.tensor([[np.cos(0.5), -np.sin(0.5), 0],
                       [np.sin(0.5), np.cos(0.5), 0],
                       [0, 0, 1]], dtype=torch.float32)
    R2 = torch.tensor([[np.cos(-0.2), 0, np.sin(-0.2)],
                      [0, 1, 0],
                       [-np.sin(-0.2), 0, np.cos(-0.2)]], dtype=torch.float32)
    S = torch.tensor([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 1]], dtype=torch.float32)

    rot = R2 @ R1
    dest = rot @ src
    res, _ = util.compute_rigid(src, dest)
    print("SO3 only:")
    print(rot - res)
    print(torch.det(res))
    print(res @ res.T)

    rot = S @ R1 @ R2
    dest = rot @ src
    res, _ = util.compute_rigid(src, dest)
    print("O3 only:")
    print(rot - res)
    print(torch.det(res))
    print(res @ res.T)

    print("Error full rigid:")
    offset = torch.tensor([[2], [1], [-3]])
    rot = R1 @ R2
    dest = rot @ (src + offset)
    R, T = util.compute_rigid(src, dest)
    warped = R @ src + T
    print(torch.sum(torch.norm(warped - dest, dim=0)))


def test_fastdtw():
    """
    Tests how well fastdtw works to synchronise human motions.
    """
    path_dir = "/home/embreaded/Downloads/PKUMMD/Pose_Sequences/"

    # dont change the sequence, code only works on this sequence (advantage: massive speed-up)
    path_l = os.path.join(path_dir, "0225-L.txt")
    path_m = os.path.join(path_dir, "0225-M.txt")

    print("Parsing... ")
    video_l = parse_action_sequence(path_l)
    video_m = parse_action_sequence(path_m)
    print("Done")

    print("Cross-view identification...")
    # hardcoded swap!!! fails for other sequences!!!
    for i in range(len(video_l)):
        x = video_l[i][torch.LongTensor([1, 0])]
        video_l[i] = x
    print("Done")

    print("Disassembling into actions...")
    video_l = disassemble_sequence(video_l, path_l)
    video_m = disassemble_sequence(video_m, path_m)
    print("Done")

    print("Synchronizing...")
    edges = util.get_kinect_edges()
    unwanted = [util.get_kinect_joint_by_name("hand_r"),
                util.get_kinect_joint_by_name("hand_l"),
                util.get_kinect_joint_by_name("wrist_r"),
                util.get_kinect_joint_by_name("wrist_l")]
    # extract action with bad ground truth time stamps
    seq_l = video_l[2]
    seq_m = video_m[2]
    seq_m = seq_m[1:]  # trim so that sequence has even length
    print(len(seq_l), len(seq_m))
    inv_l = util.get_view_invariant_representation(seq_l, edges, unwanted)
    inv_m = util.get_view_invariant_representation(seq_m, edges, unwanted)

    # test match where middle view shows idling person
    print("Only start: ")
    print(fastdtw(inv_l, inv_m[:53], dist=lambda x, y: util.angular_distance(x, y, 30))[0])

    visualisation.draw_sequence(seq_m[:53])
    visualisation.draw_still_poses(seq_m[53])

    # test match where middle view shows same action as left view!s
    print("Only end: ")
    print(fastdtw(inv_l, inv_m[53:], dist=lambda x, y: util.angular_distance(x, y, 30))[0])

    visualisation.draw_sequence(seq_m[53:])
    visualisation.draw_still_poses(seq_m[-1])
    # visualisation.draw_sequence(seq_l)


if __name__ == "__main__":
    # show_sequence()
    # test_continuity()
    # benchmark_matching_features()
    # test_align()
    # test_mean_bone()
    test_preprocessing()
    # test_rigid()
    # test_fastdtw()
    quit()
