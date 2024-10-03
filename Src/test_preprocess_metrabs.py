import visualisation
from preprocess_metrabs import *


def test_fusion(tracklets=True):
    """
    Tests the preprocessing for Metrabs detections.

    Args:
        tracklets: Use tracklets if true, else use raw detections.
    """
    torch.manual_seed(42)
    if tracklets:
        path_dir = "/globalwork/sarandi/data/pkummd-more/estimates_38814fa9/"
        ending = ".npy"
    else:
        path_dir = "/globalwork/sarandi/data/pkummd-more/raw-poses-b470be9/RGB_VIDEO/"
        ending = ".pkl"
    path_kinect = "/globalwork/datasets/pkummd/PKU_Skeleton_Renew/"

    # use 66 or 107 for raw poses (others are incomplete)
    video_number = "0042"
    path_l = os.path.join(path_dir, video_number + "-L" + ending)
    path_m = os.path.join(path_dir, video_number + "-M" + ending)
    path_r = os.path.join(path_dir, video_number + "-R" + ending)

    if tracklets:
        print("Parsing... ")
        video_l = parse_tracklets(path_l)
        video_m = parse_tracklets(path_m)
        video_r = parse_tracklets(path_r)
        print("Done")

        print("left:", sum([(x[0] != 0).any() for x in video_l]), sum([(x[1] != 0).any() for x in video_l]))
        print("middle:", sum([(x[0] != 0).any() for x in video_m]), sum([(x[1] != 0).any() for x in video_m]))
        print("right:", sum([(x[0] != 0).any() for x in video_r]), sum([(x[1] != 0).any() for x in video_r]))

        print("Tracking within each view... ")
        remove_false_positives(video_l, os.path.join(path_kinect, video_number + "-L.txt"))
        remove_false_positives(video_m, os.path.join(path_kinect, video_number + "-M.txt"))
        remove_false_positives(video_r, os.path.join(path_kinect, video_number + "-R.txt"))
        print("Done")
    else:
        print("Parsing... ")
        video_l = parse_action_sequence(path_l)
        video_m = parse_action_sequence(path_m)
        video_r = parse_action_sequence(path_r)
        print("Done")

        print("Tracking within each view... ")
        # fill up sequence to have at least 2 poses per frame
        fill_second(video_l)
        fill_second(video_m)
        fill_second(video_r)

        # add zero pose to first frame
        # reason: remove potential Metrabs false positives in first frame via "determine_start"
        video_l[0] = torch.cat([video_l[0], torch.zeros(1, 24, 3)], dim=0)
        video_m[0] = torch.cat([video_l[0], torch.zeros(1, 24, 3)], dim=0)
        video_r[0] = torch.cat([video_l[0], torch.zeros(1, 24, 3)], dim=0)

        # remove all but the 2 most likely poses in first frame
        determine_start(video_l, os.path.join(path_kinect, video_number + "-L.txt"))
        determine_start(video_m, os.path.join(path_kinect, video_number + "-M.txt"))
        determine_start(video_r, os.path.join(path_kinect, video_number + "-R.txt"))

        # order poses and remove false positives in all other frames
        remove_false_positives(video_l, os.path.join(path_kinect, video_number + "-L.txt"))
        remove_false_positives(video_m, os.path.join(path_kinect, video_number + "-M.txt"))
        remove_false_positives(video_r, os.path.join(path_kinect, video_number + "-R.txt"))
        print("Done")

    print("Cross-view identification...")
    edges = get_metrabs_edges()
    parse.identify_cross_view(video_m, video_l, edges, [])
    parse.identify_cross_view(video_m, video_r, edges, [])
    print("Done")

    print("Disassembling into actions...")
    path_stamp_dir = "/globalwork/datasets/pkummd/Train_Label_PKU_final"
    time_l, time_m, time_r, _ = extract_good_stamps(path_stamp_dir, file_no=video_number)
    video_l = disassemble_good_only(video_l, time_l)
    video_m = disassemble_good_only(video_m, time_m)
    video_r = disassemble_good_only(video_r, time_r)
    print("Done")

    print("Synchronizing...")
    for i in range(len(video_m)):
        seq_l = video_l[i]
        seq_m = video_m[i]
        seq_r = video_r[i]
        # synchronise 3 different views of the same person
        m = min(len(seq_l), len(seq_m), len(seq_r))
        parse.synchronise_views(seq_l, seq_m, seq_r, edges, [])
        print(str(m - len(seq_m)) + " frames discarded.")
    print("Done")

    print("Calibrating...")
    parse.calibrate(video_l, video_m, video_r)
    print("Done")

    for i in range(len(video_m)):
        seq_l = video_l[i]
        seq_m = video_m[i]
        seq_r = video_r[i]

        # remove implausible bone lengths
        print("Action " + str(i) + ":")
        parse.check_bone_length(seq_l, "/globalwork/datasets/pkummd/mean_bone_length_metrabs.txt", edges, verbose=False)
        parse.check_bone_length(seq_m, "/globalwork/datasets/pkummd/mean_bone_length_metrabs.txt", edges, verbose=False)
        parse.check_bone_length(seq_r, "/globalwork/datasets/pkummd/mean_bone_length_metrabs.txt", edges, verbose=False)

        # old code to visualize how well synchronisation worked:
        res = []
        for j in range(len(seq_m)):
            res.append(torch.stack([seq_l[j][0], seq_m[j][0], seq_r[j][0]], dim=0))
        visualisation.draw_sequence(res, "metrabs")

        # fuse ground truths
        print("Fusing sequences...")
        res = parse.fuse_ground_truth_geo_median(seq_l, seq_m, seq_r, get_metrabs_joint_by_name("pelvi"))
        print("Done")
        visualisation.draw_sequence(res, "metrabs")


def test_mean_bone():
    """
    Tests the computation of the mean bone length.
    """
    path = "/home/embreaded/Downloads/PKUMMD/metrabs/raw-poses"
    mean, num = util.get_mean_bone_length(path, get_metrabs_edges(), parse_action_sequence)
    print(num)
    print(mean)


def test_camera():
    """
    This function can be used to visualize the handedness of the coordinate system.
    A source of error might be that PKU-MMD videos are flipped along x-axis.
    """
    path_kin = "/home/embreaded/Downloads/PKUMMD/Pose_Sequences/0066-M.txt"
    path_metrabs = "/home/embreaded/Downloads/PKUMMD/metrabs/raw-poses/0066-M.pkl"

    kinect = parse.parse_action_sequence(path_kin)
    metrabs = parse_action_sequence(path_metrabs)

    print("Frame 200")
    print(kinect[200][1:, util.get_kinect_joint_by_name("head"), :])
    print(metrabs[200][:1, get_metrabs_joint_by_name("head_"), :])

    rot = torch.tensor([[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
                        [np.sin(np.pi / 2),  np.cos(np.pi / 2), 0],
                        [                0,                  0, 1]], dtype=torch.float)

    rotated_kinect = torch.cat([(rot @ kinect[200][1, i, :])[None, None, :] for i in range(25)], dim=1)
    rotated_metrabs = torch.cat([(rot @ metrabs[200][0, i, :])[None, None, :] for i in range(24)], dim=1)

    stacked_kinect = torch.cat([rotated_kinect, kinect[200][1:, :, :]], dim=0)
    stacked_metrabs = torch.cat([rotated_metrabs, metrabs[200][:1, :, :]], dim=0)

    visualisation.draw_still_poses(stacked_kinect, "kinect")
    visualisation.draw_still_poses(stacked_metrabs, "metrabs")


def test_preprocess():
    """
    Tests whether preprocessing stages 1 to 3 work  on an example sequence.
    """
    preprocess_first(start=7, end=7)
    preprocess_second(start=7, end=7)
    preprocess_third(start=7, end=7)
    preprocess_fourth(start=7, end=7)


def test_bone_matrix():
    """
    Compares the bone vector matrix with naive bone vector computation. Should output "True" twice.
    """
    pose = torch.rand(24, 3)
    to_bones = get_bone_vector_matrix()
    to_joints = torch.inverse(to_bones)
    bones_1 = util.get_bone_vectors(pose, get_metrabs_edges())
    bones_2 = to_bones @ pose.view(72)
    print((bones_1.view(-1) == bones_2[3:]).all())
    print((pose.view(-1) == to_joints @ bones_2).all())


def test_geometric_median():
    """
    Visualises all 3 synchronised views (blue, purple, beige) and the geometric median (red).
    """
    path = "/work/erlbeck/datasets/pkummd_enhanced/calibrated"
    video_number = "0196"

    # load data
    video_l = numpy_load_sequence(os.path.join(path, video_number + "-L.npy"))
    video_m = numpy_load_sequence(os.path.join(path, video_number + "-M.npy"))
    video_r = numpy_load_sequence(os.path.join(path, video_number + "-R.npy"))
    edges = get_metrabs_edges()

    # remove implausible bone lengths
    parse.check_bone_length(video_l, "/globalwork/datasets/pkummd/mean_bone_length_metrabs.txt", edges, verbose=False)
    parse.check_bone_length(video_m, "/globalwork/datasets/pkummd/mean_bone_length_metrabs.txt", edges, verbose=False)
    parse.check_bone_length(video_r, "/globalwork/datasets/pkummd/mean_bone_length_metrabs.txt", edges, verbose=False)

    # compute geometric median
    fused = parse.fuse_ground_truth_geo_median(video_l, video_m, video_r, get_metrabs_joint_by_name("pelvi"))

    # red is geometric median
    res = [torch.stack([a[0], b[0], c[0], d[0]], dim=0) for a, b, c, d in zip(video_l, video_m, video_r, fused)]
    visualisation.draw_sequence(res[700:900:2], "metrabs")


if __name__ == "__main__":
    # test_fusion()
    # test_mean_bone()
    # test_camera()
    # test_preprocess()
    test_bone_matrix()
    # test_geometric_median()
