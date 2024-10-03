import os
import math

import torch
import numpy as np

import preprocess_metrabs as pm


class MotionDataset(torch.utils.data.Dataset):

    def __init__(self, indices_p, root_dir_p, indices_n, root_dir_n, indices_a, root_dir_a, device, augmentation=None,
                 filter_threshold=0.4, input_length=60, output_length=30, frame_rate=30):
        """
        Initialises the dataset.

        Args:
            indices_p: The list of integers corresponding to PKU-MMD video numbers
                belonging to this split (train, validation or test).
            root_dir_p: The PKU-MMD root directory.
            indices_n: The list of integers corresponding to NTU-RGBD set (!) numbers
                belonging to this split (train, validation or test).
            root_dir_n: The NTU-RGBD root directory.
            indices_a: The list of integers corresponding to Amass video numbers
                belonging to this split (pre-training, skip).
            root_dir_a: The Amass root directory.
            device: The device where the whole data set is loaded to.
            augmentation: The chain of data augmentations.
            filter_threshold: The threshold in meters where Kinect sequences will be discarded due to fast motion.
                Cases where a joint moves more than 1 meter in a single frame often indicate erroneous data.
            input_length: The maximal length of the input.
            output_length: The length of the output of the seq2seq network.
            frame_rate: The framerate to which Amass sequences are downsampled.
        """
        self.indices_p = indices_p
        self.root_p = root_dir_p
        self.indices_n = indices_n
        self.root_n = root_dir_n
        self.indices_a = indices_a
        self.root_a = root_dir_a
        self.device = device
        self.augmentation = augmentation
        self.filter_threshold = filter_threshold
        self.input_length = input_length
        self.output_length = output_length
        self.frame_rate = frame_rate

        self.data = []
        self.masks = []
        self.action_labels = []

        # load data
        self.load_pkummd()
        self.load_nturgbd()
        self.load_amass()

    def load_pkummd(self):
        """
        Loads all sequences from PKU-MMD belonging to the current split.
        """
        # sort because listdir order is undefined
        files = sorted(os.listdir(self.root_p))
        for file in files:

            vid_no = int(file.split("-")[0])
            if vid_no not in self.indices_p or file[5:10] != "poses":
                continue

            poses = torch.load(os.path.join(self.root_p, file))
            times = torch.load(os.path.join(self.root_p, file[:5] + "times.pt"))
            actions = torch.load(os.path.join(self.root_p, file[:5] + "actions.pt"))

            for i in range(0, len(times), 2):
                end = times[i + 1].item()
                start = times[i].item()

                if end - start < 2 * self.output_length:
                    # compute padding length for input and output
                    num_padding = self.output_length - (end - start) // 2

                    # pad input if available => more meaningful input
                    start = max(0, start - num_padding)

                    # output must be padded due to batching!
                    # output must be masked => padding values irrelevant
                    motion = poses[start:end]
                    self.center(motion)  # center before padding!!!
                    motion = torch.cat([motion, torch.zeros(num_padding, motion.size(1), motion.size(2))], dim=0)

                    # create mask for target padding
                    masking = torch.ones(self.output_length, poses.size(1), dtype=torch.long, device=self.device)
                    masking[(self.output_length - num_padding):, :] = 0
                else:
                    motion = poses[start:end]
                    self.center(motion)
                    masking = torch.ones(self.output_length, poses.size(1), dtype=torch.long, device=self.device)
                action_label = actions[i // 2] + 200  # add 200 to differentiate from NTU labels

                # filter sequences with "teleporting" to improve quality
                diff = np.diff(motion, axis=0)
                if torch.max(torch.abs(torch.tensor(diff))) > self.filter_threshold:
                    continue

                self.data.append(motion.to(self.device))
                self.masks.append(masking)
                self.action_labels.append(action_label.repeat(motion.size(1)))  # repeat if two person present

    def load_nturgbd(self):
        """
        Loads all sequences from NTU-RGB+D belonging to the current split.
        """
        # sort because listdir order is undefined
        files = sorted(os.listdir(self.root_n))
        for file in files:
            # skip if not in current split
            setup = int(file[1:4])
            if setup not in self.indices_n:
                continue

            poses = torch.load(os.path.join(self.root_n, file)).float()
            masking = torch.ones(self.output_length, poses.size(1), dtype=torch.long, device=self.device)
            action_label = torch.tensor(int(file[(-6):(-3)]))

            # filter sequences with "teleporting" to improve quality
            diff = np.diff(poses, axis=0)
            if torch.max(torch.abs(torch.tensor(diff))) > self.filter_threshold:
                continue

            # center before potential padding
            self.center(poses)

            if len(poses) < 2 * self.output_length:
                # create padding and corresponding mask
                num_padding = self.output_length - len(poses) // 2
                poses = torch.cat([poses, torch.zeros(num_padding, poses.size(1), poses.size(2))], dim=0)
                masking[(self.output_length - num_padding):, :] = 0

            # store data
            self.data.append(poses.to(self.device))
            self.masks.append(masking)
            self.action_labels.append(action_label.repeat(poses.size(1)))  # repeat if two person present

    def load_amass(self):
        """
        Loads all sequences from Amass.
        """
        # sort because listdir order is undefined
        files = sorted(os.listdir(self.root_a))
        for file in files:
            # skip if not in current split
            file_name = file.split(".")[0]
            dataset = file_name.split("-")[1]
            if dataset not in self.indices_a:
                continue
            stride = int(file[13:16]) // self.frame_rate

            raw_data = torch.load(os.path.join(self.root_a, file)).float()
            poses = raw_data[::stride].contiguous()
            masking = torch.ones(self.output_length, poses.size(1), dtype=torch.long, device=self.device)
            action_label = torch.LongTensor([300])  # generic label for Amass

            # filter sequences with length 1
            if poses.size(0) == 1:
                continue
            poses = poses.view(-1, 1, 72)

            # center before potential padding
            self.center(poses)

            if len(poses) < 2 * self.output_length:
                # create padding and corresponding mask
                num_padding = self.output_length - len(poses) // 2
                poses = torch.cat([poses, torch.zeros(num_padding, poses.size(1), poses.size(2))], dim=0)
                masking[(self.output_length - num_padding):, :] = 0

            # store data
            self.data.append(poses.to(self.device))
            self.masks.append(masking)
            self.action_labels.append(action_label)

    @staticmethod
    def center(sequence):
        """
        Shifts the poses in-place so that the center of mass equals the origin.

        Args:
            sequence: The tensor of poses.
        """
        list_of_points = sequence.view(-1, 3)
        center = torch.mean(list_of_points, dim=0)
        list_of_points.add_((-1) * center)

    def compute_stats(self, converter):
        """
        Computes mean and standard deviation (unbiased) over the whole dataset feature-wise.

        Args:
            converter: The coordinate converter with the correct input representation.
        Returns:
            The mean pose in the desired representation.
            The standard deviation in the desired representation.
        """
        # init
        n_samples = 0
        sum = torch.zeros(1, 1, 72, device=self.device)
        sum_sqs = torch.zeros(1, 1, 72, device=self.device)

        # numerical stability is not an issue
        # more stable method differed by less than one millimeter
        for i, tensor in enumerate(self.data):
            # clip padding and re-arrange
            num_relevant = tensor.size(0) - self.masks[i].size(0) + torch.sum(self.masks[i], dim=0)[0]
            list_of_poses = converter.encode(tensor[:num_relevant, :, :].view(-1, 72))

            # compute metrics
            n_samples += list_of_poses.size(0)
            sum += torch.sum(list_of_poses, dim=0)
            sum_sqs += torch.sum(list_of_poses ** 2, dim=0)

        mean = sum / n_samples
        var_biased = sum_sqs / n_samples - mean ** 2
        var_corrected = var_biased * n_samples / (n_samples - 1)
        dev = torch.sqrt(var_corrected)
        dev[dev < 0.001] = 0.001  # avoid zero division

        return mean, dev

    def __len__(self):
        """
        Returns the number of sequences

        Returns:
            The number of sequences.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the corresponding sequence.

        Args:
            index: The index of the element to be returned.

        Returns:
            The training part of the sequence (variable length).
            The target part of the sequence (the last output_length frames).
            The masking which invalidates padded frames.
            The action id of the motion.
        """
        tensor = self.data[index]
        mask = self.masks[index]
        action = self.action_labels[index]

        if self.augmentation is not None:
            tensor = self.augmentation(tensor)

        if mask[-1, 0] == 1:
            # no padding / masking occurs => sequence long enough for random split
            len_diff = tensor.size(0) - self.output_length * 2  # number of possible splits
            # instead of uniformly drawing from [0, len_diff]
            # we draw two uniform variables and add them
            # => bias to mid split ("pyramid distribution")
            random = torch.rand(2) * len_diff / 2
            split = torch.round(random[0] + random[1])
            # offset to have at least output_length frames input
            split = int(split.item()) + self.output_length
            # trim long sequences
            start = max(0, split - self.input_length)
        else:
            # padding occurred => split where target starts
            # by construction of padding, this is at the middle of the original sequence
            split = tensor.size(0) - self.output_length
            start = 0

        X = tensor[start:split, :, :]
        Y = tensor[split:(split + self.output_length), :, :]

        return X, Y, mask, action


class ExhaustiveSet(MotionDataset):
    """
    Dataset which makes better use of long Amass sequences and is able to filter sequences where pose is idling.
    """
    def __init__(self, indices_p, root_dir_p, indices_n, root_dir_n, indices_a, root_dir_a, device, use_short, use_idle,
                 skip_rate, augmentation=None, filter_threshold=0.4, input_length=60, output_length=30, frame_rate=30):
        """
        Initialises the dataset.

        Args:
            indices_p: The list of integers corresponding to PKU-MMD video numbers
                belonging to this split (train, validation or test).
            root_dir_p: The PKU-MMD root directory.
            indices_n: The list of integers corresponding to NTU-RGBD set (!) numbers
                belonging to this split (train, validation or test).
            root_dir_n: The NTU-RGBD root directory.
            indices_a: The list of integers corresponding to Amass video numbers
                belonging to this split (pre-training, skip).
            root_dir_a: The Amass root directory.
            device: The device where the whole data set is loaded to.
            use_short: Flag deciding whether to use sequences shorter than input_length + output_length frames.
                This may make use of masking.
            use_idle: If false applies a heuristic to filter those sequences
                where nothing is happening during last frames of input.
            skip_rate: The amount of frames skipped to avoid training on redundant data.
            augmentation: The chain of data augmentations.
            filter_threshold: The threshold in meters where Kinect sequences will be discarded due to fast motion.
                Cases where a joint moves more than 1 meter in a single frame often indicate erroneous data.
            input_length: The maximal length of the input.
            output_length: The length of the output of the seq2seq network.
            frame_rate: The framerate to which Amass sequences are downsampled.
        """
        super(ExhaustiveSet, self).__init__(indices_p, root_dir_p, indices_n, root_dir_n, indices_a, root_dir_a,
                                            device, augmentation, filter_threshold, input_length, output_length,
                                            frame_rate)

        # state variable to ensure that compute_stats and extend_to_two are called in right order
        self.extended_to_two = False

        if not use_idle:
            print("Filtering sequences with little motion at the end of the conditioning input.")
        if use_short:
            print(f"Using sequences with less than {input_length + output_length} frames.")

        # implement data access like in https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/amass3d.py
        self.data_idx = []

        for i, tensor in enumerate(self.data):
            t = tensor.size(0)
            valid_frames = np.arange(0, t - (self.input_length + self.output_length) + 1, skip_rate)

            # long sequences might contain lots of idle poses
            # network should avoid seeing too many of those
            if not use_idle:
                temp = []
                joint = pm.get_metrabs_joint_by_name("r_han")

                for start_frame in valid_frames:
                    # look at the frames where input stops and prediction starts.
                    split = start_frame + self.input_length
                    around_split = tensor[(split - 5):(split + 5), :, :]
                    diff = np.diff(around_split.view(10, -1, 24, 3)[:, :, joint, :].cpu(), axis=0)
                    dist = torch.linalg.norm(torch.tensor(diff), dim=2)
                    feature = torch.max(dist, dim=1)[0]

                    # 10 mm as threshold seemed to filter easy test cases well
                    if 1000 * torch.sum(feature) > 10:
                        temp.append(start_frame)

                valid_frames = np.array(temp)

            # due to masking, we can also evaluate shorter sequences (should be used mainly for Kinect)
            if use_short and len(valid_frames) == 0:
                valid_frames = np.array([0])

            tmp_data_idx_1 = [i] * len(valid_frames)
            tmp_data_idx_2 = list(valid_frames)
            self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

    def extend_to_two(self):
        """
        Single person sequences are duplicated so that each sequence has two actors.
        Needs to be called after compute_stats!!! Trying to call compute_stats afterwards will not work.
        If this function is called and the current architecture is a transformer, be sure to provide num_people = 2 as
            parameter in get_transformer_collate_fn.
        """
        if len(self.indices_p) == 0 and len(self.indices_n) == 0:
            print("Padding to two people, however dataset is single-person only...")

        for i in range(len(self.data)):
            if self.data[i].size(1) == 1:
                self.data[i] = self.data[i].repeat(1, 2, 1)
                # number of samples in evaluation is computed via this mask
                # thus, the mask for a single person should only contain one row of 1
                self.masks[i] = torch.cat([self.masks[i], torch.zeros_like(self.masks[i])], dim=1)
                # duplicate action label because evaluation code expects batch_size == #people
                self.action_labels[i] = self.action_labels[i].repeat(2)

        # change state so that compute_stats cannot be called anymore
        self.extended_to_two = True

    def compute_stats(self, converter):
        """
        Computes mean and standard deviation (unbiased) over the whole dataset feature-wise.

        Args:
            converter: The coordinate converter with the correct input representation.
        Returns:
            The mean pose in the desired representation.
            The standard deviation in the desired representation.
        """
        assert not self.extended_to_two, "After calling extend_to_two(), data statistics would be incorrect."
        return super(ExhaustiveSet, self).compute_stats(converter)

    def __len__(self):
        """
        Returns the number of sequences

        Returns:
            The number of sequences.
        """
        return len(self.data_idx)

    def __getitem__(self, item):
        """
        Returns the corresponding sequence.

        Args:
            item: The index of the element to be returned.

        Returns:
            The training part of the sequence (variable length).
            The target part of the sequence (the last output_length frames).
            The masking which invalidates padded frames.
            The action id of the motion.
        """
        # this part is adopted from Mao et al.
        key, start_frame = self.data_idx[item]
        length = min(self.input_length + self.output_length, self.data[key].size(0))
        fs = np.arange(start_frame, start_frame + length)
        sequence = self.data[key][fs]

        # augment and get all the necessary meta info
        if self.augmentation is not None:
            sequence = self.augmentation(sequence)
        X = sequence[:(-self.output_length), :, :]
        Y = sequence[(-self.output_length):, :, :]
        mask = self.masks[key]
        action = self.action_labels[key]

        return X, Y, mask, action


def transform_rotate(sample):
    """
    Randomly rotates the poses around the y-axis, i.e., ground plane is left unchanged.

    Args:
        sample: The tensor containing an action sequence.
    Returns:
        The tensor containing the randomly rotated sequence.
    """
    phi = torch.rand(1) * 2 * math.pi
    cos = torch.cos(phi)
    sin = torch.sin(phi)
    rot = torch.tensor([[cos, 0, sin],
                        [0, 1, 0],
                        [-sin, 0, cos]], device=sample.device)
    transformed = sample.view(-1, 3) @ rot.T
    return transformed.view(sample.size())


def transform_scale(sample):
    """
    Randomly scales the poses by a factor between 0.8 and 1.2.

    Args:
        sample: The tensor containing an action sequence.
    Returns:
        The tensor containing the scaled sequence.
    """
    scale = torch.rand(1, device=sample.device) * 0.4 + 0.8
    return sample * scale


def transform_flip(sample):
    """
    With a probability of 50%, the sequence is flipped (exchanging left and right).

    Args:
        sample: The tensor containing an action sequence.
    Returns:
        The same tensor or the flipped tensor.
    """
    if torch.rand(1).item() < 0.5:
        return sample
    else:
        # permutate the joints to exchange labels like "right hand" and "left hand"
        joint_flip_perm = [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11, 13, 12, 14, 16, 15, 18, 17, 20, 19, 22, 21, 23]
        poses = sample.view(sample.size(0), sample.size(1), 24, 3)
        # permute joints first => input is not changed implicitly
        permuted_poses = poses[:, :, joint_flip_perm, :]
        permuted_poses[:, :, :, 0] *= -1  # actual flip
        return permuted_poses.view(sample.size())


def collate_action_sequences(batch):
    """
    Function to collect several action sequences into a batch.
    This function is passed to the DataLoader class from pytorch.

    Args:
        batch: A tuple of 4 lists, namely the inputs, the prediction targets, the masks and the action_ids.
    Returns:
        A dictionary containing input, target, lengths of the padded sequences, masking and action ids.
    """
    sequences, targets, masking, actions = zip(*batch)

    # concat the fixed-length target sequences
    Y = torch.cat(targets, dim=1)
    masking = torch.cat(masking, dim=1)
    actions = torch.cat(actions, dim=0)
    # allocate "padded" batch matrix
    X = torch.zeros(max([x.size(0) for x in sequences]), Y.size(1), sequences[0].size(2), device=Y.device)

    # fill the batch matrix
    lengths = []
    batch_num = 0  # needed because a sequence may consist of 1 or 2 samples (persons)
    for sequence in sequences:
        for person_num in range(sequence.size(1)):
            lengths.append(sequence.size(0))
            X[:lengths[batch_num], batch_num, :] = sequence[:, person_num, :]
            batch_num += 1

    lengths, permutation = torch.sort(torch.tensor(lengths), descending=True)
    X = X[:, permutation, :]
    Y = Y[:, permutation, :]
    masking = masking[:, permutation]
    actions = actions[permutation]
    res = {"input": X, "target": Y, "lengths": lengths, "masking": masking, "actions": actions}

    return res


def get_transformer_collate_fn(num_people):
    """
    Returns a function which will collate a batch of motion sequences.

    Args:
        num_people: The maximum number of people in the dataset. Provide 1 for Amass and 2 for Kinect.
            If 2 is provided, you should also call ExhaustiveSet.extend_to_two().
    Returns:
        A function pointer which can be provided to collate_fn parameter of pytorch's Dataloader.
    """
    def collate_batch_transformer(batch):
        """
        Function to collect several action sequences into a batch for the transformer.
        We need to pad in the beginning instead of the end, but we do not need to sort.

        Args:
            batch: A tuple of 4 lists: inputs, targets, target masks and action ids.
        Returns:
            A dictionary containing input, target, input padding masks, output padding masking and action ids.
        """
        sequences, targets, masking, actions = zip(*batch)
        len_in = max([x.size(0) for x in sequences])
        len_out, _, f_dim = targets[0].size()
        b_dim = len(targets)
        device = targets[0].device

        # stack along new batch dimension, then use view to have consistent order between input, target and mask
        Y = torch.stack(targets, dim=1).view(len_out, -1, f_dim)
        masking = torch.stack(masking, dim=1).view(len_out, -1)
        actions = torch.stack(actions, dim=0).view(-1)

        X = torch.zeros(len_in + len_out, b_dim, num_people, f_dim, device=device)
        lengths = torch.zeros(len_in + len_out,  b_dim, num_people, dtype=torch.bool, device=device)

        # fill the batch matrix
        for i, sequence in enumerate(sequences):
            start = len_in - sequence.size(0)
            X[start:len_in, i, :, :] = sequence
            lengths[:start, i, :] = True

        # init ground truth with constant velocity heuristic
        diff = X[len_in - 1, :, :, :] - X[len_in - 2, :, :, :]
        steps = torch.arange(1, len_out + 1, device=device)
        const_velo = X[len_in - 1, :, :, :].unsqueeze(0) + steps.view(len_out, 1, 1, 1) * diff.unsqueeze(0)
        X[len_in:, :, :, :] = const_velo

        if num_people == 2:
            # mask out second person if it is identical (= padded)
            # use logical or, because both temporal and person padding should be masked
            lengths[:, :, 1].logical_or_(X[:, :, 0, 0] == X[:, :, 1, 0])

        res = {"input": X, "target": Y, "lengths": lengths, "masking": masking, "actions": actions}
        return res

    return collate_batch_transformer


def split_pkummd():
    """
    Computes the video numbers belonging to train, validation and test split for PKU-MMD.

    Returns:
        A set of integers belonging to the training split.
        A set of integers belonging to the validation split.
        A set of integers belonging to the testing split.
    """
    split_pku = {
        # chosen from second half of sequences => maybe higher quality
        # also chosen such that proportion of dual person is representative
        "testing_start": [181, 325],
        "testing_length": [25, 25],
        "validation_start": [206, 300],
        "validation_length": [25, 25],
    }

    validation = set()
    for start, length in zip(split_pku["validation_start"], split_pku["validation_length"]):
        validation.update(range(start, start + length))

    testing = set()
    for start, length in zip(split_pku["testing_start"], split_pku["testing_length"]):
        testing.update(range(start, start + length))

    training = set(filter(
        lambda x: x not in validation and x not in testing,
        range(365)))

    return {"training": training,
            "validation": validation,
            "testing": testing}


def split_nturgbd():
    """
    Computes the set-up numbers belonging to train, validation and test split for NTU-RGB+D.

    Returns:
        A set of integers belonging to the training split.
        A set of integers belonging to the validation split.
        A set of integers belonging to the testing split.
    """
    split_ntu = {
        # both validation and test have 2 set-ups from NTU60 and 2 from NTU120
        # also avoid set-ups with high slope as detections seem more inaccurate
        "testing_id": [14, 15, 29, 30],
        "validation_id": [12, 13, 31, 32],
    }

    validation = set(split_ntu["validation_id"])
    testing = set(split_ntu["testing_id"])
    training = set(filter(
        lambda x: x not in validation and x not in testing,
        range(1, 33)))

    return {"training": training,
            "validation": validation,
            "testing": testing}


def split_amass():
    """
    Returns the datasets belonging to train, validation and test split for Amass.

    Returns:
        A list of dataset names belonging to the training split.
        A list of dataset names belonging to the validation split.
        A list of dataset names belonging to the testing split.
    """
    training = ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'TCD_handMocap', 'ACCAD']
    validation = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh']
    testing = ['BioMotionLab_NTroje']

    return {"training": training,
            "validation": validation,
            "testing": testing}
