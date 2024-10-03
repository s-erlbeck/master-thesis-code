import numpy as np
import matplotlib.pyplot as plt
# the following import IS necessary, do not trust pycharm
from mpl_toolkits.mplot3d import Axes3D

import util
import preprocess_metrabs as parse


def draw_still_poses(poses, tree="kinect", filename=None):
    """
    Visualizes still poses via Matplotlib (i.e. no movement).
    x-Axis is expected to go right (w.r.t. camera POV)
    y-Axis is expected to go down (w.r.t. camera POV)
    z-Axis is expected to be principal axis.
    Coordinates are therefore in a right-handed system (in meters).

    Args:
        poses: The (batch x 25 x 3) torch tensor containing the pose.
        tree: The name of the Kinematic tree used.
        filename: The file where the figure is stored. If None, it is displayed instead.
    """
    meta_info = interpret_tensor(tree)

    fig = plt.figure(1)
    ax = configure(fig)
    draw_frame(ax, poses, meta_info)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def draw_sequence(motion_sequence, tree="kinect"):
    """
    Renders a temporal sequence of poses via Matplotlib.
    x-Axis is expected to go right (w.r.t. camera POV)
    y-Axis is expected to go down (w.r.t. camera POV)
    z-Axis is expected to be principal axis.
    Coordinates are therefore in a right-handed system (in meters).

    Args:
        motion_sequence: A list of (batch x 25 x 3) torch tensors.
        tree: The name of the Kinematic tree used.
    """
    meta_info = interpret_tensor(tree)

    for poses in motion_sequence:
        # create figure 1 or make it the currently selected
        fig = plt.figure(1)
        # clear figure
        plt.clf()
        # configure plot (axis scale etc.)
        ax = configure(fig)
        # define what is drawn (needs to call ax.plot())
        draw_frame(ax, poses, meta_info)
        # wait and then update
        plt.pause(.01)


def interpret_tensor(tree):
    """
    Helper method to correctly configure how to interpret the tensor.

    Args:
        tree: The string describing which estimator was used for the poses.

    Returns:
         The list of edges of the kinematic tree.
         The index of the head joint.
    """
    if tree == "kinect":
        edges = util.get_kinect_edges()
        head = util.get_kinect_joint_by_name("head")
    elif tree == "metrabs":
        edges = parse.get_metrabs_edges()
        head = parse.get_metrabs_joint_by_name("head_")
    else:
        raise ValueError("Unknown tree: " + str(tree))
    return edges, head


def configure(fig):
    """
    Configuration common to multiple visualizations.

    Args:
        fig: The figure object of Matplotlib.

    Returns:
        The axis object created for this figure object.
    """
    # create 3D axis
    ax = fig.add_subplot(111, projection="3d")
    # fix axis scales
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)
    return ax


def draw_frame(ax, poses, meta_info):
    """
    Draws a frame of the sequence.

    Args:
        ax: The axis object of the 3D plot.
        poses: The torch tensor of an arbitrary number of poses.
        meta_info: The tuple returned by interpret_tensor.
    """
    poses = poses.numpy()
    rainbow = iter(plt.cm.rainbow(np.linspace(0, 1, len(poses))))  # create an iterator over different colors
    for pose in poses:
        color = next(rainbow)
        draw_pose(ax, pose, meta_info, color)


def draw_pose(ax, pose, meta_info, color):
    """
    Draws a pose.

    Args:
        ax: The axis object of the 3D plot.
        pose: The numpy matrix containing a 25x3 pose.
        meta_info: The tuple returned by interpret_tensor.
        color: The color this pose should have.
    """
    edges, head = meta_info

    # draw point at head to see if pose == torch.zeros(25,3)
    # providing colors as list avoids warning
    # also swap y- and z-axis and mirror y-axis
    # => x right, y depth, z up (also right-handed)
    ax.scatter(pose[head, 0], pose[head, 2], (-1) * pose[head, 1], "o", c=[color])

    # Next line can be uncommented to provide text labels for each pose (variable text needs to be defined first)
    # ax.text(pose[i, 0], pose[i, 2], (-1) * pose[i, 1], "{}".format(str(text)))

    # draw all the edges
    for edge in edges:
        # extract start and end coordinates from matrix
        matrix = pose[[edge[0], edge[1]], :]
        # swap y- and z-axis and mirror y-axis
        # => x right, y depth, z up (also right-handed)
        ax.plot(matrix[:, 0], matrix[:, 2], (-1) * matrix[:, 1], "-", c=color)
