import collections
import argparse

import torch
import numpy as np

import conversion
import util
import visualisation


class Metric:
    """
    Implements an abstract metric.
    Calling update will process this batch according to the metric.
    Calling evaluate will return the metric on all batches fed in so far.
    """
    def __init__(self, n_frames, device):
        """
        Initialises the metric.

        Args:
            n_frames: The number of frames to evaluate over.
            device: The device where the network outputs lie.
        """
        self.n_frames = n_frames
        self.errors = collections.defaultdict(lambda: torch.zeros(n_frames, device=device))
        self.n_samples = collections.defaultdict(lambda: torch.zeros(n_frames, device=device))

    def compute_metric(self, distances):
        """
        Method which defines how the metric is computed from euclidean distance matrix.
        Needs to be defined by subclass.

        Args:
            distances: The joint-wise distances between prediction and target in millimeters.
                The expected size is T x B x 24.
        Returns:
            The T x B resulting scores of the metric.
        """
        raise NotImplementedError("This method is abstract, instantiate a subclass instead!")

    def update(self, action_id, distances, masking):
        """
        Adds a batch of motion predictions to the underlying data structure.

        Args:
            action_id: The tensor containing the action labels of the batch.
            distances: The joint-wise distances between prediction and target in millimeters.
                The expected size is T x B x 24.
            masking: The masking of the padded frames.
        """
        if self.n_frames != distances.size(0):
            raise ValueError("Wrong dimension!")

        metric = self.compute_metric(distances)

        action_id = action_id.tolist()  # needed because keys are compared with is, not ==
        for b in range(metric.size(1)):
            self.errors[action_id[b]] += metric[:, b] * masking[:, b]
            self.n_samples[action_id[b]] += masking[:, b]

    def evaluate(self, per_action):
        """
        Computes the metric on all motions which were previously provided via update.

        Args:
            per_action: If true, compute metric for each action, else on whole dataset.
        Returns:
            The dictionary mapping action label to metric at each time step.
        """
        res = {}
        if per_action:
            for action in self.errors:
                res[action] = self.errors[action] / self.n_samples[action]
        else:
            res[0] = sum(self.errors.values()) / sum(self.n_samples.values())
        return res


class MPJPE(Metric):
    """
    Implements mean per joint positioning error in millimeters.
    """
    def compute_metric(self, distances):
        """
        Computes the arithmetic mean of the provided distances.

        Args:
            distances: The joint-wise distances between prediction and target in millimeters.
                The expected size is T x B x 24.
        Returns:
            The T x B resulting mean per joint errors.
        """
        return torch.mean(distances, dim=2)


class PCK(Metric):
    """
    Implements percentage of correct keypoints metric.
    """

    def __init__(self, n_frames, device, threshold):
        """
        Initialises the metric.

        Args:
            n_frames: The number of frames to evaluate over.
            device: The device where the network outputs lie.
            threshold: The threshold in millimeters about when a keypoint is considered to be correct.
        """
        self.threshold = threshold
        super(PCK, self).__init__(n_frames, device)

    def compute_metric(self, distances):
        """
        Computes the percentage of correct keypoints.

        Args:
            distances: The joint-wise distances between prediction and target in millimeters.
                The expected size is T x B x 24.
        Returns:
            The T x B resulting fractions (between 0 and 1).
        """
        correct_keypoints = distances <= self.threshold
        return torch.mean(correct_keypoints.float(), dim=2)


class PCK_AUC(Metric):
    """
    Computes area und the curve of percentage of correct keypoints metric.
    """

    def __init__(self, n_frames, device, max_threshold):
        """
        Initialises the metric.

        Args:
            n_frames: The number of frames to evaluate over.
            device: The device where the network outputs lie.
            max_threshold: The maximum threshold in millimeters, i.e., upper boundary of integral.
        """
        self.max_threshold = max_threshold
        super(PCK_AUC, self).__init__(n_frames, device)

    def compute_metric(self, distances):
        """
        Computes the area under the curve of the percentage of correct keypoints.

        Args:
            distances: The joint-wise distances between prediction and target in millimeters.
                The expected size is T x B x 24.
        Returns:
            The T x B resulting normalised areas under the curves.
        """
        pck_auc = torch.clamp(1 - distances / self.max_threshold, min=0)  # exact integration of AUC
        return torch.mean(pck_auc, dim=2)


def euclidean_distances(predicted, target):
    """
    Computes the euclidean distance between each predicted joint location and the corresponding target location.
    Results are in millimeters.

    Args:
        predicted: The predicted poses of dimension T x B x 24 x 3.
        target: The target poses of dimension T x B x 24 x 3.
    Returns:
        The distances between the positions.
    """
    # compute squared difference
    squared_diff = torch.nn.functional.mse_loss(predicted, target, reduction="none")
    # get euclidean distances in millimeters
    norm = 1000 * torch.sqrt(torch.sum(squared_diff, dim=3))
    return norm


def evaluate_quantitative(predictions, input_mode, len_out, relative, until, action, evaluate_joints, align_joints):
    """
    Evaluates a model quantitatively.

    Args:
        predictions: The stored predictions from the model.
        input_mode: The string describing the input representation.
        len_out: The integer determining the time horizon.
        relative: If true, evaluate root-relative poses instead of absolute poses.
        until: If true, use temporal mean instead of score at corresponding time step.
        action: If true, compute metric per action ID, else jointly.
        evaluate_joints: The numpy array containing the indices of which joints to use for evaluation.
            Should contain all joints unless comparing to authors who ignore some joints.
        align_joints: The numpy array containing the indices of which joints to use for removing
            global rotation. Global orientation should only be removed when comparing to literature,
            i.e., dataset is Amass, protocol is comparison, relative is true and until is false.

    Returns:
        A dictionary with the error metrics per action.
    """
    device = torch.device("cpu")
    converter = conversion.CoordinateConverter(input_mode, device)
    index = 2 * int(relative)

    # init metrics
    mpjpe = MPJPE(len_out, device)
    pck05 = PCK(len_out, device, 50)
    pck10 = PCK(len_out, device, 100)
    pck15 = PCK(len_out, device, 150)
    p_auc = PCK_AUC(len_out, device, 110)

    for i in range(len(predictions.files) // 4):
        # load predictions
        pred = predictions[f"{i}_prediction"]
        targ = predictions[f"{i}_target"]
        mask = predictions[f"{i}_masks"]
        a_id = predictions[f"{i}_actions"]

        pred, targ = converter.decode(torch.tensor(pred), torch.tensor(targ))[index:(index + 2)]

        # check if global orientation has to be removed
        if len(evaluate_joints) < targ.size(2):
            rot = util.compute_rigid_batched(pred[:, :, align_joints, :],
                                             targ[:, :, align_joints, :])
            temp = pred @ rot.permute(0, 1, 3, 2)
            pred = temp

        # accumulate
        distances = euclidean_distances(pred, targ)
        mpjpe.update(a_id, distances[:len_out, :, evaluate_joints], mask[:len_out, :])
        pck05.update(a_id, distances[:len_out, :, evaluate_joints], mask[:len_out, :])
        pck10.update(a_id, distances[:len_out, :, evaluate_joints], mask[:len_out, :])
        pck15.update(a_id, distances[:len_out, :, evaluate_joints], mask[:len_out, :])
        p_auc.update(a_id, distances[:len_out, :, evaluate_joints], mask[:len_out, :])

    # evaluate
    res_mpjpe = mpjpe.evaluate(action)
    res_pck05 = pck05.evaluate(action)
    res_pck10 = pck10.evaluate(action)
    res_pck15 = pck15.evaluate(action)
    res_p_auc = p_auc.evaluate(action)

    if until:
        reduce_fn = lambda x: x.mean().item()
    else:
        reduce_fn = lambda x: x[-1].item()

    res = {"names": ("  MPJPE", "PCK 050", "PCK 100", "PCK 150", "PCK AUC")}  # all names 7 char long
    for key in res_mpjpe.keys():
        res[key] = (reduce_fn(res_mpjpe[key]),
                    reduce_fn(res_pck05[key]),
                    reduce_fn(res_pck10[key]),
                    reduce_fn(res_pck15[key]),
                    reduce_fn(res_p_auc[key]),)

    return res


def evaluate_qualitative(predictions, input_mode):
    """
    Evaluates a model qualitatively.

    Args:
        predictions: The stored predictions from the model.
        input_mode: The string describing the input representation.
    """
    device = torch.device("cpu")
    converter = conversion.CoordinateConverter(input_mode, device)

    for i in range(len(predictions.files) // 4):
        # load predictions
        pred = predictions[f"{i}_prediction"]
        targ = predictions[f"{i}_target"]

        # visualize
        pred, targ, _, _ = converter.decode(torch.tensor(pred), torch.tensor(targ))
        for person in range(pred.size(1)):
            seq = [torch.stack([x, y], dim=0) for x, y in zip(pred[:, person, :, :], targ[:, person, :, :])]
            visualisation.draw_sequence(seq, "metrabs")


def pretty_print_dict(dic):
    """
    Prints a dictionary in a readable manner.
    That means that the function sorts the keys, truncates decimal places and inserts line breaks.

    Args:
        dic: The dictionary to print.
    Returns:
        The string which can be printed (or parsed).
    """
    header = dic.pop("names")
    res = f" action\t{header[0]}\t{header[1]}\t{header[2]}\t{header[3]}\t{header[4]}\n"
    for key in sorted(dic):
        res += f"{key:7d}\t{dic[key][0]:7.1f}\t{100 * dic[key][1]:7.1f}\t"
        res += f"{100 * dic[key][2]:7.1f}\t{100 * dic[key][3]:7.1f}\t{dic[key][4]:7.3f}\n"
    return res


def main():
    """
    Parses the program arguments and runs qualitative and quantitative tests and baselines accordingly.
    Use "-h" options to learn more on the available arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to the model output which was computed with prediction.py")
    parser.add_argument("-a", "--action", help="compute metric per action ID", action="store_true")
    parser.add_argument("-f", "--full", help="align prediction and ground truth with full Procrustes (all joints) "
                                             "to remove global rotation for evaluation,"
                                             "this flag only has an effect if 'relative' is provided and until "
                                             "is not provided and the dataset is Amass and the protocol is "
                                             "'comparison', otherwise global rotation IS evaluated", action="store_true")
    parser.add_argument("-r", "--relative", help="use root-relative poses for evaluation "
                                                 "instead of absolute coordinates", action="store_true")
    parser.add_argument("-s", "--short", help="compute metrics for short-term prediction (400 ms) "
                                              "instead of long-term prediction (1 s)", action="store_true")
    parser.add_argument("-u", "--until", help="use arithmetic mean over time steps "
                                              "instead of evaluating at each step independently", action="store_true")
    parser.add_argument("-v", "--visualize", help="visualize the predictions", action="store_true")
    args = parser.parse_args()

    # load stuff
    names = args.path.split("/")[-1].split("_")
    dataset = names[1]
    protocol = names[2]
    input_mode = names[3]
    if len(names) < 7:
        # baseline
        model_date = "---"
        model_name = f"Baseline {names[5].split('.')[0].title()}Motion"
    else:
        # real model
        model_date = names[5]
        model_name = names[6].split(".")[0]
    t = 0.4 if args.short else 1.0
    if protocol == "ablation":
        len_out = int(t * 30)
    elif protocol == "comparison":
        len_out = int(t * 25)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    # adapt setting if comparison to Mao is made
    evaluate_joints = np.arange(24)
    align_joints = None
    if protocol == "comparison" and args.relative and dataset == "amass":
        if args.until:
            print("Warning: literature evaluates at t=25, but temporal mean was selected!")
        else:
            print("Using less joints and removing global rotation.")
            evaluate_joints = np.arange(3, 21)
            if args.full:
                print("Removing global rotation by aligning all joints.")
                print("This may make the model seem better than it is.")
                align_joints = np.arange(24)
            else:
                print("Removing global rotation by aligning hips and spine joint.")
                print("This may make the model seem worse than it is.")
                align_joints = np.arange(3)

    # compute
    predictions = np.load(args.path)
    res = evaluate_quantitative(predictions, input_mode, len_out, args.relative, args.until,
                                args.action, evaluate_joints, align_joints)

    print(f"Name: {model_name}\n"
          f"Date: {model_date}\n"
          f"Data: {dataset} in {input_mode} representation\n"
          f"Protocol: {protocol}\n"
          f"Evaluation: {'relative' if args.relative else 'absolute'}\n"
          f"Prediction: {t:.1f} seconds\n"
          f"Reduction: {'mean over t' if args.until else 'score at t'}\n")
    print(pretty_print_dict(res))

    if args.visualize:
        print("Visualizing predictions...")
        evaluate_qualitative(predictions, input_mode)


if __name__ == "__main__":
    main()
