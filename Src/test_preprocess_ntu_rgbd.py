import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from preprocess_ntu_rgbd import *
import visualisation


def test_align_feet():
    """
    Tests ground plane alignment on a single sequence. Plots all the feet before and after rotation.
    """
    path = "/work/erlbeck/datasets/nturgbd_enhanced/final/"

    # load data and transform to list of feet coordinates
    x = torch.load(path + "S001C001P001R001A060.pt")
    x = x.view(len(x), -1, 24, 3)
    feet = x[:, :, [pm.get_metrabs_joint_by_name("l_toe"), pm.get_metrabs_joint_by_name("r_toe")], :]
    feet = feet.view(-1, 3)

    # estimate and apply rotation
    rot = align_feet(feet.numpy())
    rotated = feet @ torch.tensor(rot, dtype=torch.float).T

    # plot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2], c="r")
    ax.scatter(feet[:, 0], feet[:, 1], feet[:, 2], c="b")
    plt.show()


def test_get_all_ground_plane_transforms():
    """
    Computes ground plane alignment in each set-up (NTU RGB+D 60).
    Prints the resulting Euler angles of the rotation.
    """
    path = "/work/erlbeck/datasets/nturgbd_enhanced/final/"

    for i in range(1, 18):
        # try out different person ids (not all actors participated in all set-ups)
        for p in range(1, 41):  # 40 highest person id
            try:
                x = torch.load(path + "S{:03d}C001P{:03d}R001A060.pt".format(i, p))
                break
            except FileNotFoundError:
                continue

        # transform to list of feet
        x = x.view(len(x), -1, 24, 3)
        feet = x[:, :, [pm.get_metrabs_joint_by_name("l_toe"), pm.get_metrabs_joint_by_name("r_toe")], :]
        feet = feet.view(-1, 3)

        # estimate and apply rotation
        rot = align_feet(feet.numpy())
        rotated = x @ torch.tensor(rot, dtype=torch.float).T

        # provide Euler angles of set-up
        print(i)
        print(scipy.spatial.transform.Rotation.from_matrix(rot).as_euler("xyz", True))
        # visualisation.draw_still_poses(torch.cat([rotated[0, ...], rotated[-1, ...]], dim=0), "metrabs")


def test_all_setups():
    """
    Tests the align_all method once for each of the 32 set-ups.
    """
    path = "/work/erlbeck/datasets/nturgbd_enhanced/final/"
    rotations = align_all(path)

    for i in range(1, 33):
        # take a sequence from this set-up with a meaningful action
        for file in os.listdir(path):
            action = int(file[(-6):(-3)])
            setup = int(file[1:4])
            if setup == i and (action == 60 or action == 116):
                x = torch.load(os.path.join(path, file))
                break
            else:
                continue

        # apply rotation
        x = x.view(len(x), -1, 24, 3)
        rot = torch.tensor(rotations[i - 1], dtype=torch.float)
        rotated = x @ rot.T

        # visualise
        print(i)
        visualisation.draw_still_poses(torch.cat([rotated[0, ...], rotated[-1, ...]], dim=0), "metrabs")


def test_all_actions(setup_to_check=10):
    """
    Tests the align_all method for all actions (with id 60 / 116) in one set-up.

    Args:
        setup_to_check: The set-up number which should be visualised.
    """
    path = "/work/erlbeck/datasets/nturgbd_enhanced/final/"
    feet = []

    # extract all feet from suitable actions
    for file in os.listdir(path):
        action = int(file[(-6):(-3)])
        setup = int(file[1:4])

        # these actions include walking => robust estimation
        if action != 60 and action != 116:
            continue
        elif setup != setup_to_check:
            continue

        # reshape the feet to Nx3
        data = torch.load(os.path.join(path, file)).view(-1, 24, 3)
        data = data[:, [pm.get_metrabs_joint_by_name("l_toe"), pm.get_metrabs_joint_by_name("r_toe")], :]
        data = data.view(-1, 3)
        feet.append(data)

    # compute rotation
    data = torch.cat(feet, dim=0)
    rotation = align_feet(data.numpy())
    print(scipy.spatial.transform.Rotation.from_matrix(rotation).as_euler("xyz", True))

    for file in os.listdir(path):
        action = int(file[(-6):(-3)])
        setup = int(file[1:4])
        if action != 60 and action != 116:
            continue
        elif setup != setup_to_check:
            continue

        # reshape the feet to Nx3
        data = torch.load(os.path.join(path, file))
        data = data.view(len(data), -1, 24, 3)
        rotated = data @ rotation.T
        visualisation.draw_still_poses(torch.cat([rotated[0, ...], rotated[-1, ...]], dim=0), "metrabs")


def visualise_calibrated():
    """
    Visualises some cases where calibration failed.
    """
    # two views fail for the same person, thats why RANSAC is ignoring this person
    path_l = "/work/erlbeck/datasets/nturgbd_enhanced/calibrated/S010C001P021R001A060.npy"
    path_m = "/work/erlbeck/datasets/nturgbd_enhanced/calibrated/S010C002P021R001A060.npy"
    path_r = "/work/erlbeck/datasets/nturgbd_enhanced/calibrated/S010C003P021R001A060.npy"

    # wrong cross-person identification?
    # path_l = "/work/erlbeck/datasets/nturgbd_enhanced/calibrated/S032C001P067R001A116.npy"
    # path_m = "/work/erlbeck/datasets/nturgbd_enhanced/calibrated/S032C002P067R001A116.npy"
    # path_r = "/work/erlbeck/datasets/nturgbd_enhanced/calibrated/S032C003P067R001A116.npy"

    x = torch.tensor(np.load(path_l))
    y = torch.tensor(np.load(path_m))
    z = torch.tensor(np.load(path_r))

    res = torch.cat([x, y, z], dim=1)

    visualisation.draw_sequence(res, "metrabs")


if __name__ == "__main__":
    # test_align_feet()
    # test_get_all_ground_plane_transforms()
    # test_all_setups()
    test_all_actions(32)
    # visualise_calibrated()
