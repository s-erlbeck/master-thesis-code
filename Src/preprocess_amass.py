import pickle
import os

import torch

import visualisation as vis


def preprocess_amass(path_dest):
    """
    Loads the pickle of all AMASS tracklets and stores the data ready for training.

    Args:
        path_dest: The path were the torch tensors are stored for training.
    """
    # AMASS joint order: body joints (0 till 21), left hand (22 till 36), right hand (37 till 51)
    # reason for index 0 at end: Metrabs has pelvis as last joint
    ordered_rows = list(range(1, 22)) + [22, 37, 0]

    with open("/globalwork/datasets/amass/tracks_20210927.pkl", "rb") as handle:
        data = pickle.load(handle)
    with open("/globalwork/datasets/amass/frame_rate_per_track_20210927.pkl", "rb") as handle:
        framerates = pickle.load(handle)

    datasets = ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'TCD_handMocap', 'ACCAD',
                'HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'BioMotionLab_NTroje']
    skipped = set()
    i = 0

    # this one is a dictionary so iterate over values
    for key, track in data.items():
        # only use those that Mao et al use
        dataset = key.split("/")[0]
        if dataset not in datasets:
            skipped.add(dataset)
            continue

        # get fps to normalise data
        fps = round(framerates[key])

        # discard hand joints, reorder pelvis
        seq = track[:, ordered_rows, :]

        # transform axis (default: looking at camera)
        seq[:, :, 1:] *= -1
        motion = seq[:, :, [0, 2, 1]]

        # either casting or contiguous call save 4,5 GB disk space
        motion = torch.tensor(motion[:, None, :, :]).contiguous()
        torch.save(motion, os.path.join(path_dest, f"motion_{i:05d}_{fps:03d}fps-{dataset}.pt"))
        i += 1

    print("Datasets not used:")
    print(sorted(skipped))


def load_data():
    """
    Loads the pickle of the first 100 AMASS tracklets and visualizes the motion.
    """
    ordered_rows = list(range(1, 22)) + [22, 37, 0]

    with open("/globalwork/datasets/amass/tracks100.pkl", "rb") as handle:
        data = pickle.load(handle)

    for track in data:
        seq = track[:, ordered_rows, :]
        seq[:, :, 1:] *= -1
        motion = seq[:, :, [0, 2, 1]]
        vis.draw_still_poses([torch.tensor(x[None, :, :]) for x in motion][300], "metrabs")


if __name__ == "__main__":
    load_data()
    # preprocess_amass("/work/erlbeck/datasets/amass_mao_version/")
