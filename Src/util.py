import os

import numpy as np
import scipy.optimize
import torch


kinect_joint_names = {
    "spinebase"    :  0,
    "spinemid"     :  1,
    "neck"         :  2,
    "head"         :  3,
    "shoulder_l"   :  4,
    "elbow_l"      :  5,
    "wrist_l"      :  6,
    "hand_l"       :  7,
    "shoulder_r"   :  8,
    "elbow_r"      :  9,
    "wrist_r"      : 10,
    "hand_r"       : 11,
    "hip_l"        : 12,
    "knee_l"       : 13,
    "ankle_l"      : 14,
    "foot_l"       : 15,
    "hip_r"        : 16,
    "knee_r"       : 17,
    "ankle_r"      : 18,
    "foot_r"       : 19,
    "spineshoulder": 20,
    "handtip_l"    : 21,
    "thump_l"      : 22,
    "handtip_r"    : 23,
    "thump_r"      : 24
}


kinect_edges = (
    # spine
    ("spinebase", "spinemid"),
    ("spinemid", "spineshoulder"),
    ("spineshoulder", "neck"),
    ("neck", "head"),
    # arm left
    ("spineshoulder", "shoulder_l"),
    ("shoulder_l", "elbow_l"),
    ("elbow_l", "wrist_l"),
    ("wrist_l", "hand_l"),
    ("hand_l", "handtip_l"),
    ("wrist_l", "thump_l"),  # yeah, the thump starts at the wrist
    # arm right
    ("spineshoulder", "shoulder_r"),
    ("shoulder_r", "elbow_r"),
    ("elbow_r", "wrist_r"),
    ("wrist_r", "hand_r"),
    ("hand_r", "handtip_r"),
    ("wrist_r", "thump_r"),  # yeah, the thump starts at the wrist
    # leg left
    ("spinebase", "hip_l"),
    ("hip_l", "knee_l"),
    ("knee_l", "ankle_l"),
    ("ankle_l", "foot_l"),
    # leg right
    ("spinebase", "hip_r"),
    ("hip_r", "knee_r"),
    ("knee_r", "ankle_r"),
    ("ankle_r", "foot_r"),
)


def get_kinect_joint_by_name(name):
    return kinect_joint_names[name]


def get_kinect_edges():
    return tuple((kinect_joint_names[edge[0]], kinect_joint_names[edge[1]]) for edge in kinect_edges)


def get_view_invariant_representation(pose_seq, edges, unwanted):
    """
    In order to compare two Kinect pose sequences with unknown calibration, one needs an invariant representation.
    An angle between two vectors is not changed by a euclidean transform (camera extrinsics).
    This method computes relevant joint angles.

    Args:
        pose_seq: The sequence of poses.
        edges: The kinematic tree of the pose.
        unwanted: A list of all joints which should not be considered for angle features (e.g. noisy joints like thump)
    Returns:
        A sequence of features.
    """
    res = [[] for _ in range(len(pose_seq))]
    # for all frames
    for frame, poses in enumerate(pose_seq):
        # for both poses in a frame
        for pose in poses:
            # iterate over all unordered pairs of distinct edges
            for i, edge1 in enumerate(edges):
                for j, edge2 in enumerate(edges):
                    if j <= i:
                        continue
                    # found unordered pair
                    # now check if the edges share a common joint
                    elif edge1[0] in edge2 and edge1[0] not in unwanted:
                        if edge1[0] == edge2[0]:
                            res[frame].append(compute_angle(pose[edge1[1], :] - pose[edge1[0], :],
                                                            pose[edge2[1], :] - pose[edge2[0], :]))
                        elif edge1[0] == edge2[1]:
                            res[frame].append(compute_angle(pose[edge1[1], :] - pose[edge1[0], :],
                                                            pose[edge2[0], :] - pose[edge2[1], :]))
                    elif edge1[1] in edge2 and edge1[1] not in unwanted:
                        if edge1[1] == edge2[0]:
                            res[frame].append(compute_angle(pose[edge1[0], :] - pose[edge1[1], :],
                                                            pose[edge2[1], :] - pose[edge2[0], :]))
                        elif edge1[1] == edge2[1]:
                            res[frame].append(compute_angle(pose[edge1[0], :] - pose[edge1[1], :],
                                                            pose[edge2[0], :] - pose[edge2[1], :]))
    return np.array(res)


def compute_angle(a, b):
    """
    Compute angle between two vectors in 3D.

    Args:
        a: First vector.
        b: Second vector.
    Returns:
        The angle from [0°, 180°].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    else:
        x = np.dot(a, b) / (norm_a * norm_b)
        # clipping necessary due to floating point arithmetic
        if x > 1:
            x = 1
        elif x < -1:
            x = -1
        return np.degrees(np.arccos(x))


def hungarian_match(objs1, objs2, *, dist_fn=None, threshold=np.inf):
    """
    Perform hungarian matching between objects in the two sets.

    Args:
        objs1: First list of objects.
        objs2: Second list of objects.
        dist_fn: A function pointer which invokes the distance used for matching.
        threshold: How large of a distance is a match allowed to have.

    Returns:
        List of matches (= tuples). First value is from objs1, second from objs2, third is distance.
        List of the unmatched objects from objs1.
        List of the unmatched objects from objs2.
    """
    # code adapted from István
    if len(objs1) == 0 or len(objs2) == 0:
        return [], objs1, objs2
    if dist_fn is None:
        raise TypeError("Distance function required.")

    distance_matrix = np.array([[dist_fn(obj1, obj2) for obj2 in objs2] for obj1 in objs1])
    obj1_indices, obj2_indices = scipy.optimize.linear_sum_assignment(distance_matrix)

    unmatched_objs1 = [obj for i, obj in enumerate(objs1) if i not in obj1_indices]
    unmatched_objs2 = [obj for i, obj in enumerate(objs2) if i not in obj2_indices]
    matches = []
    for i1, i2 in zip(obj1_indices, obj2_indices):
        if distance_matrix[i1, i2] < threshold:
            matches.append((objs1[i1], objs2[i2], distance_matrix[i1, i2]))
        else:
            unmatched_objs1.append(objs1[i1])
            unmatched_objs2.append(objs2[i2])
    return matches, unmatched_objs1, unmatched_objs2


def angular_distance(x, y, threshold=1):
    """
    Computes a robust distance metric between x and y.
    Each component is computed as min(1, |x-y| / threshold),
    i.e. 0 iff identical and 1 if distance between x and y is larger than threshold.
    The complete distance is the arithmetic mean of all components.

    Args:
        x: The first vector of angles.
        y: The second vector of angles.
        threshold: The threshold at which larger differences are considered as outliers.
    Returns:
        The scalar distance.
    """
    scaled = np.abs(x - y) / threshold
    trimmed = np.minimum(scaled, 1.0)
    return np.mean(trimmed)


def angular_dist_min(x, y, threshold):
    """
    A variant of angular distance where only the distance of the better matching person is considered
        instead of averaging over all joints irrespective of person.

    Args:
        x: The joint angles of 2 persons in the first frame.
        y: The joint angles of 2 persons in the second frame.
        threshold: The threshold at which larger differences are considered as outliers.
    Returns:
        The minimum angular distance of the 2 persons.
    """
    half = x.shape[0] // 2
    dist_1 = angular_distance(x[:half], y[:half], threshold)
    dist_2 = angular_distance(x[half:], y[half:], threshold)

    # if only one person visible in both views, use the other person
    if np.sum(x[:half]) == 0 and np.sum(y[:half]) == 0:
        return dist_2
    elif np.sum(x[half:]) == 0 and np.sum(y[half:]) == 0:
        return dist_1
    else:
        return min(dist_1, dist_2)


def get_mean_bone_length(path, edges, func_parse, path_res=None):
    """
    Extracts the arithmetic mean of bone lengths from a dataset.
    Stores result in a file at the following location: "path/.."

    Args:
        path: The path to where the pose sequences are stored. No other file / directory should lie there.
        edges: The kinematic tree used.
        func_parse: The function used to parse a single file inside the directory, depending on the dataset.
                    The function is expected to take the path to the file as its only argument.
                    It is expected to return an iterable of 2x25x3 torch tensors.
        path_res: The path where the mean bone length is stored.
                  If None (default), the result is stored in the parent directory of path.
    Returns:
        The torch tensor containing the average bone length.
        The number of skeletons parsed.
    """
    # remove trailing slash because how os.path.dirname works
    if path[-1] == "/":
        path = path[:-1]

    mean = torch.zeros(len(edges))
    num_skeletons = 0
    for filename in os.listdir(path):
        print("Processing " + filename, end="\r")
        sequence = func_parse(os.path.join(path, filename))
        for poses in sequence:
            # compute bone lengths (batched for efficiency)
            bones = get_bone_vectors(poses, edges)
            norms = torch.norm(bones, dim=2)

            # add each pose to the mean
            for pose, _ in enumerate(poses):
                if (norms[pose, :] > 0.0).any():
                    mean += norms[pose, :]
                    num_skeletons += 1

    print("\n", end="\r")  # move cursor to next line
    # store mean
    mean = mean / num_skeletons

    if path_res is None:
        path_res = os.path.join(os.path.dirname(path), "mean_bone_length.txt")
    else:
        path_res = os.path.join(path_res, "mean_bone_length.txt")
    with open(path_res, "w") as out:
        for value in mean:
            out.write(str(value.item()) + "\n")

    return mean, num_skeletons


def get_bone_vectors(tensor, edges):
    """
    Transforms a pose consisting of absolute joint locations to metric offset vectors.
    Supports batch processing.

    Args:
        tensor: The pose torch tensor of size (batch x 25 x 3) or (25 x 3)
        edges: The kinematic tree of the poses.

    Returns:
        The torch tensor of size (batch x #bones x 3).
    """
    if len(tensor.size()) == 2:
        tensor = tensor.view(1, -1, 3)

    res = torch.empty(tensor.size()[0], len(edges), 3)
    for i, edge in enumerate(edges):
        res[:, i, :] = tensor[:, edge[1], :] - tensor[:, edge[0], :]

    return res


def compute_rigid(source, destination):
    """
    Estimates a rigid transformation mapping source to destination using orthogonal Procrustes alignment.
    Compare https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem (last accessed: 30.09.2020)

    Args:
        source: The 3xN tensor which is to be transformed.
        destination: The 3xN tensor which is the reference.
    Returns:
        The rotation R and translation T such that R @ source + T = destination
    """
    # center point clouds at origin
    center_src = torch.mean(source, dim=1).view(3, 1)
    center_dst = torch.mean(destination, dim=1).view(3, 1)
    src = source - center_src
    dst = destination - center_dst

    # compute special orthogonal matrix
    M = dst @ src.T
    U, _, V = torch.svd(M)
    S = torch.eye(3)
    S[2, 2] = torch.det(U @ V.T)
    R = U @ S @ V.T

    # for translation, also rotate the mean
    T = center_dst - R @ center_src
    return R, T


def compute_rigid_batched(src, dst):
    """
    Same method as previous, but expects A x B x N x 3 inputs,
        where A and B are batch dimensions and N is the number of 3D points.
    Also this method assumes the data is correctly centered and just computes a rotation.

    Args:
        src: The data points to be transformed.
        dst: The data points to which src should be transformed
    Returns:
        The rotation R which transforms a point from src to dst.
        Use src @ R.permute(0, 1, 3, 2) for batched transformed.
    """
    # new implementation less error-prone than changing a function which is 1 year old
    M = dst.permute(0, 1, 3, 2) @ src
    U, _, V = torch.svd(M)
    S = torch.eye(3).repeat(src.size(0), src.size(1), 1, 1)
    S[:, :, 2, 2] = torch.det(U @ V.permute(0, 1, 3, 2))
    R = U @ S @ V.permute(0, 1, 3, 2)
    return R


def ransac_rigid(source, destination, error=0.05, num_iter=500):
    """
    Performs robust Procrustes estimation by using RANSAC scheme.

    Args:
        source: The 3xN tensor which is to be transformed.
        destination: The 3xN tensor which is the reference.
        error: The threshold in meters for distinguishing outliers from inliers.
        num_iter: The number of random samples used.
    Returns:
        The rotation R and translation T such that R @ source + T = destination
    """
    # init
    N = source.size()[1]
    best_inlier = torch.zeros(N, dtype=torch.bool)

    # do num_iter trials
    for i in range(num_iter):
        # compute hypothesis on randomly sampled points
        shuffle = torch.randperm(N)
        sample_src = source[:, shuffle][:, :3]
        sample_dst = destination[:, shuffle][:, :3]
        R, T = compute_rigid(sample_src, sample_dst)

        # compute number of inliers of current hypothesis
        warped = R @ source + T
        inlier = torch.norm(warped - destination, dim=0) <= error

        # compare to best consensus found so far
        if torch.sum(inlier) > torch.sum(best_inlier):
            best_inlier = inlier

    if torch.sum(best_inlier) == 0:
        raise ValueError("No inliers found at all. Maybe error threshold is too low: ", error)
    return compute_rigid(source[:, best_inlier], destination[:, best_inlier])


def geometric_median(X, eps):
    """
    Code to compute the geometric median of a set of points.

    Args:
        X: The numpy array of points.
        eps: The threshold in millimetres.

    Returns:
        The geometric median.
    """
    # code adapted from István
    # who probably got it from here: https://stackoverflow.com/questions/30299267/
    from scipy.spatial.distance import cdist, euclidean
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def masked_median(coords, mask, eps):
    """
    I do not know what this does, but eps is again in millimetres.
    """
    # code adapted from István
    valid_coords = coords[mask]
    if len(valid_coords) > 0:
        return geometric_median(valid_coords, eps=eps)
    else:
        return geometric_median(coords, eps=eps)


def compute_median_pose(poses, i_root, joint_validity_mask=None, eps=0.005):
    """
    Function to compute geometric median for poses.

    Args:
        poses: Iterable of poses (each in a numpy array).
        i_root: The ID of the pelvis joint for root relative computation.
        joint_validity_mask: No idea, just set it to None.
        eps: The threshold in millimetres.

    Returns:
        The geometric median pose.
    """
    # code adapted from István
    poses = np.asarray(poses)
    rootrel_poses = poses - poses[:, i_root:i_root + 1]

    if joint_validity_mask is None:
        joint_validity_mask = np.full(poses.shape[:2], True)

    rootrel_median = np.stack([masked_median(rootrel_poses[:, i], joint_validity_mask[:, i], eps=eps) for i in range(rootrel_poses.shape[1])])
    root_median = masked_median(poses[:, i_root], joint_validity_mask[:, i_root], eps=eps)
    return root_median + rootrel_median
