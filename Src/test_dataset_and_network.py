import time

import numpy as np
import torch

import baselines
import conversion
import util
import visualisation
import evaluation as eval
from rnn import *
from dataset import *


def test_network():
    """
    General check that tests the forward pass of the network.
    """
    with torch.no_grad():
        path = "/work/erlbeck/datasets/pkummd_enhanced/final/0106-poses.pt"
        inp = torch.load(path)[:60, :, ]
        seq = [poses.view(2, 24, 3) for poses in inp]

        torch.manual_seed(42)
        network = SingleMotionPredictor(True, 2, 512, 0)
        output = network(inp, torch.ones(2, dtype=torch.long) * len(seq), 25)

        # iterate over first dimension (= time) and reshape:
        output = [poses.view(2, 24, 3) for poses in output]
        norm = [np.linalg.norm(poses.view(-1, 3), axis=1) for poses in output]
        # print the norms to see if the points diverge
        print([np.mean(x) for x in norm])

        seq.extend(output)
        visualisation.draw_sequence(seq, "metrabs")


def benchmark_forward(network=SingleMotionPredictor(True, 1, 512, 0)):
    """
    Tests batching, GPU compatibility and measures duration of forward pass.

    Args:
        network: The network to benchmark.
    """
    with torch.no_grad():
        path = "/work/erlbeck/datasets/pkummd_enhanced/final/0106-poses.pt"
        seq = torch.load(path)[:60, :, :]
        inp = seq.repeat(1, 8, 1)

        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        torch.manual_seed(42)
        network = network.to(device)

        start = time.time()
        output = network(inp.to(device), torch.ones(16, dtype=torch.long) * 60, 35)
        end = time.time()

        output.to("cpu")

        print("Device:", device)
        print("Output dim:", list(output.size()))
        print("Time:", end - start)


def test_dataset():
    """
    Benchmarks the loading time of each batch.
    """
    indices_p = split_pkummd()["training"]
    indices_n = split_nturgbd()["training"]
    dataset = ExhaustiveSet(indices_p, "/work/erlbeck/datasets/pkummd_enhanced/final/",
                            indices_n, "/work/erlbeck/datasets/nturgbd_enhanced/normalised/",
                            [], "/work/erlbeck/datasets/amass_mao_version/",
                            "cuda:0", True, True, 120)
    size_batch = 5
    loader = torch.utils.data.DataLoader(dataset, batch_size=size_batch, shuffle=True, collate_fn=collate_action_sequences)

    model = SingleMotionPredictor(True, 1, 512, 0)
    model.to("cuda:0")

    durations = []
    previous = time.time()
    for batch in loader:
        next = time.time()
        durations.append(next - previous)
        model(batch["input"], batch["lengths"], 30)
        previous = time.time()

    print(max(durations))
    print(sum(durations) / len(durations))


def test_transforms():
    """
    Tests whether the data augmentation transformations work properly.
    """
    # best to maximise matplotlib window to see the transforms
    torch.manual_seed(44)  # don't change, good value to see the transforms
    indices_p = split_pkummd()["training"]
    indices_n = split_nturgbd()["training"]

    dataset = ExhaustiveSet(indices_p, "/work/erlbeck/datasets/pkummd_enhanced/final/",
                            indices_n, "/work/erlbeck/datasets/nturgbd_enhanced/normalised/",
                            [], "/work/erlbeck/datasets/amass_mao_version/",
                            "cpu", True, True, 120)

    sample = dataset[100][0]

    # transformed = transform_flip(sample)
    transformed = transform_rotate(sample)
    # transformed = transform_scale(sample)

    batched = torch.cat([sample, transformed], dim=1)
    seq = [x.view(-1, 24, 3) for x in batched]
    visualisation.draw_sequence(seq, "metrabs")


def test_baselines_qualitatively():
    """
    Visualises the output of the baselines zero-motion and constant velocity.
    """
    path = "/work/erlbeck/datasets/amass/motion_08000_100fps.pt"
    motion = torch.load(path)
    if len(motion.size()) == 4:
        motion = motion.view(-1, 1, 72)
    motion = motion[500:600, 0:1, :]
    input = motion[:50, :, :]

    baseline_zero = baselines.BaselineZeroMotion()
    baseline_velo = baselines.BaselineConstVelocity()

    out_zero = torch.cat([input, baseline_zero(input, torch.LongTensor([50]), 50)], dim=0)
    out_velo = torch.cat([input, baseline_velo(input, torch.LongTensor([50]), 50)], dim=0)

    visualisation.draw_sequence(torch.cat([out_zero, out_velo, motion], dim=1).view(100, -1, 24, 3), "metrabs")


def test_baseline_input_invariance():
    """
    Applies the baseline to different input representations and prints out the difference in predictions.
    Zero-motion baseline should behave the same for all input representations except for root-relative,
        where global position is not reconstructed.
    Const-velocity baseline should behave the same for all input representations except for root-relative.
        Naturally, global motion cannot be modelled by the baseline if all poses are translated to origin.
        More surprisingly, bone vectors do NOT behave differently than absolute coordinates.
        One might argue that in bone vector representation each joint movement also affects all child joints.
        This is however not the case because by moving one isolated joint, both adjacent bone vectors change,
        effectively nullifying the effect of the joint movement on all child joints.
    """
    torch.manual_seed(42)

    # comment out one of those
    # baseline = baselines.BaselineZeroMotion()
    baseline = baselines.BaselineConstVelocity()

    dataset = MotionDataset([300], "/work/erlbeck/datasets/pkummd_enhanced/final/",
                            [], "/work/erlbeck/datasets/nturgbd_enhanced/normalised/",
                            [], "/work/erlbeck/datasets/amass_mao_version/",
                            torch.device("cpu"), augmentation=None,
                            output_length=3)

    conv_abs = conversion.CoordinateConverter("abs", torch.device("cpu"))
    conv_rel = conversion.CoordinateConverter("rel", torch.device("cpu"))
    conv_mix = conversion.CoordinateConverter("mixed", torch.device("cpu"))
    conv_bon = conversion.CoordinateConverter("bones", torch.device("cpu"))

    tensor = dataset[0][0]
    tensor = tensor[:3, 0:1, :]

    data_abs = conv_abs.encode(tensor)
    data_rel = conv_rel.encode(tensor)
    data_mix = conv_mix.encode(tensor)
    data_bon = conv_bon.encode(tensor)

    out_abs, _, _, _ = conv_abs.decode(baseline(data_abs, torch.LongTensor([3]), 3), tensor[-1, :, :].repeat(3, 1, 1))
    out_rel, _, _, _ = conv_rel.decode(baseline(data_rel, torch.LongTensor([3]), 3), tensor[-1, :, :].repeat(3, 1, 1))
    out_mix, _, _, _ = conv_mix.decode(baseline(data_mix, torch.LongTensor([3]), 3), tensor[-1, :, :].repeat(3, 1, 1))
    out_bon, _, _, _ = conv_bon.decode(baseline(data_bon, torch.LongTensor([3]), 3), tensor[-1, :, :].repeat(3, 1, 1))

    print("relative")
    print(out_abs - out_rel)
    print("mixed")
    print(out_abs - out_mix)
    print("bones")
    print(out_abs - out_bon)


def test_metrics():
    """
    Tests the metrics MPJPE and PCK with some easy test cases.
    Test cases are evaluated separately and jointly on all actions.
    """
    dev = torch.device("cpu")
    mpj = eval.MPJPE(3, dev)
    pck = eval.PCK(3, dev, 10)

    gt = torch.zeros(3, 1, 24, 3)
    pr = torch.zeros(3, 1, 24, 3)
    pr[0, 0, 1, 2] = 0.24
    dist_abs = eval.euclidean_distances(pr, gt)
    mpj.update(torch.LongTensor([1]), dist_abs, torch.ones(3, 1))
    pck.update(torch.LongTensor([1]), dist_abs, torch.ones(3, 1))
    print(mpj.evaluate(True), f"action 1 should be {1000 * 0.24 / 24}, 0, 0")
    print(pck.evaluate(True), f"action 1 should be {23 / 24}, 1.0, 1.0")
    print()

    gt = torch.zeros(3, 1, 24, 3)
    pr = torch.zeros(3, 1, 24, 3)
    pr[2, 0, :, :] = 100  # should be filtered
    pr[1, 0, 6, 0] = 0.36  # 15 mm error mpjpe for frame 1
    pr[0, 0, 7:9, 1] = 0.12  # 10 mm error mpjpe for frame 0
    masking = torch.ones(3, 1)
    masking[2, 0] = 0
    dist_abs = eval.euclidean_distances(pr, gt)
    mpj.update(torch.LongTensor([2]), dist_abs, masking)
    pck.update(torch.LongTensor([2]), dist_abs, masking)
    print(mpj.evaluate(True), f"action 2 should be {1000 * 0.12 * 2 / 24}, {1000 * 0.36 / 24}, nan")
    print(pck.evaluate(True), f"action 2 should be {22 / 24}, {23 / 24}, nan")
    print()

    gt = torch.zeros(3, 2, 24, 3)
    pr = torch.zeros(3, 2, 24, 3)
    pr[1, 1, 18, 0] = 0.0024  # 0.1 mm error mpjpe for frame 1
    dist_abs = eval.euclidean_distances(pr, gt)
    mpj.update(torch.LongTensor([1, 1]), dist_abs, torch.ones(3, 2))
    pck.update(torch.LongTensor([1, 1]), dist_abs, torch.ones(3, 2))
    print(mpj.evaluate(True), f"action 1 should be {1000 * 0.24 / 24 / 3}, {1000 * 0.0024 / 24 / 3}, 0")
    print(pck.evaluate(True), f"action 1 should be {(23 / 24 + 1 + 1) / 3}, "
                              f"{(1 + 1 + 1) / 3}, 1.0")
    print()

    torch.set_printoptions(precision=6)
    print(mpj.evaluate(False), f"action 0 should be {(10 + 10) / 4}, {(15 + 0.1) / 4}, {0 / 3}")
    print(pck.evaluate(False), f"action 0 should be {(1 + 1 + 23 / 24 + 22 / 24) / 4}, "
                               f"{(23 / 24 + 1 + 1 + 1) / 4}, {3 / 3}")


def test_pck_auc():
    """
    Tests the metric PCK AUC.
    """
    auc = eval.PCK_AUC(12, torch.device("cpu"), 100)

    dist_abs = torch.tensor(range(1, 13))[:, None, None].repeat(1, 2, 1)
    dist_abs = dist_abs * 4

    auc.update(torch.LongTensor([1, 1]), dist_abs, torch.ones(12, 2))
    auc_val = auc.evaluate(True)[1].tolist()
    print(auc_val)
    print(sum(auc_val) / len(auc_val))


def test_batched_rigid():
    """
    Tests the batched rigid transformation estimation (Procrustes alignment).
    """
    src = torch.rand(2, 2, 10, 3)

    R1 = torch.tensor([[np.cos(0.5), -np.sin(0.5), 0],
                       [np.sin(0.5), np.cos(0.5), 0],
                       [0, 0, 1]], dtype=torch.float32)
    R2 = torch.tensor([[np.cos(-0.2), 0, np.sin(-0.2)],
                       [0, 1, 0],
                       [-np.sin(-0.2), 0, np.cos(-0.2)]], dtype=torch.float32)

    dst = torch.empty_like(src)
    for i in range(2):
        for j in range(2):
            a = R1 if i == 0 else R2
            b = R1 if j == 0 else torch.eye(3)
            transformed = a @ b @ src[i, j, :, :].T
            dst[i, j, :, :] = transformed.T

    rot = util.compute_rigid_batched(src, dst)
    print(torch.max(torch.abs(rot[0, 0] - R1 @ R1)))
    print(torch.max(torch.abs(rot[0, 1] - R1)))
    print(torch.max(torch.abs(rot[1, 0] - R2 @ R1)))
    print(torch.max(torch.abs(rot[1, 1] - R2)))
    print(torch.max(torch.abs(rot @ src.permute(0, 1, 3, 2) - dst.permute(0, 1, 3, 2))))


def collate_batch_transformer_old(batch):
    """
    Old version copied from gitlab!!! Copied to compare with new collate function.
    Function to collect several action sequences into a batch for the transformer.
    We need to pad in the beginning instead of the end, but we do not need to sort.

    Args:
        batch: A tuple of 4 lists: inputs, targets, target masks and action ids.
    Returns:
        A dictionary containing input, target, lengths of the padded sequences, masking and action ids.
    """
    sequences, targets, masking, actions = zip(*batch)
    max_length = max([x.size(0) for x in sequences])

    Y = torch.cat(targets, dim=1)
    masking = torch.cat(masking, dim=1)
    actions = torch.cat(actions, dim=0)
    X = torch.zeros(max_length + targets[0].size(0), Y.size(1), sequences[0].size(2), device=targets[0].device)
    lengths = torch.zeros(max_length + targets[0].size(0), Y.size(1), dtype=torch.bool, device=targets[0].device)

    # fill the batch matrix
    batch_num = 0  # needed because a sequence may consist of 1 or 2 samples (persons)
    for i, sequence in enumerate(sequences):
        start = max_length - sequence.size(0)
        X[start:max_length, batch_num:(batch_num + sequence.size(1)), :] = sequence

        # init ground truth with constant velocity heuristic
        diff = sequence[-1, :, :] - sequence[-2, :, :]
        const_velo = sequence[-1, :, :].unsqueeze(0) + \
                     torch.arange(1, Y.size(0) + 1, device=Y.device).view(30, 1, 1) * diff.unsqueeze(0)
        X[max_length:, batch_num:(batch_num + sequence.size(1)), :] = const_velo
        lengths[:start, batch_num:(batch_num + sequence.size(1))] = True
        batch_num += sequence.size(1)

    res = {"input": X, "target": Y, "lengths": lengths, "masking": masking, "actions": actions}
    return res


def compare_old_and_new_collate(bs=32):
    """
    Loads some data, collates transformer batch with old and with new method,
        and then compares const velocity baseline with absolute MPJPE.
    The only difference is caused by the different order of summation, which leads to slight differences
        due to floating point arithmetic not being commutative.

    Args:
        bs: The batch size used.
    """
    dev = torch.device("cpu")
    training_set = ExhaustiveSet(set(), "/globalwork/erlbeck/datasets/pkummd_enhanced/final/",
                                 set([8, 9, 10]), "/globalwork/erlbeck/datasets/nturgbd_enhanced/normalised/",
                                 set(), "/globalwork/erlbeck/datasets/amass_mao_version/",
                                 dev, augmentation=None, use_short=True,
                                 use_idle=False,
                                 skip_rate=30,
                                 filter_threshold=0.4,
                                 input_length=60,
                                 output_length=30,
                                 frame_rate=30)

    ################################ Old Method #######################################

    data_old = torch.utils.data.DataLoader(training_set,
                                           batch_size=bs,
                                           shuffle=False,
                                           collate_fn=collate_batch_transformer_old)

    batch_old = next(iter(data_old))

    eval_old = eval.MPJPE(30, dev)

    dist = eval.euclidean_distances(batch_old["input"][60:, :, :].view(30, -1, 24, 3), batch_old["target"].view(30, -1, 24, 3))

    eval_old.update(batch_old["actions"], dist, batch_old["masking"])

    res_old = eval_old.evaluate(False)[0]

    ################################ New Method #######################################

    training_set.extend_to_two()

    data_new = torch.utils.data.DataLoader(training_set,
                                           batch_size=bs,
                                           shuffle=False,
                                           collate_fn=get_transformer_collate_fn(2))

    batch_new = next(iter(data_new))

    eval_new = eval.MPJPE(30, dev)

    dist = eval.euclidean_distances(batch_new["input"][60:, :, :].view(30, -1, 24, 3), batch_new["target"].view(30, -1, 24, 3))

    eval_new.update(batch_new["actions"], dist, batch_new["masking"])

    res_new = eval_new.evaluate(False)[0]

    ######## Compare #######

    batch_new["input"] = batch_new["input"].view(90, -1, 72)

    for i in range(30):
        assert res_old[i] == res_new[i]

        select = batch_new["masking"][0, :] > 0
        print((batch_new["input"][:, select, :] == batch_old["input"]).all().item())
        print((batch_new["target"][:, select, :] == batch_old["target"]).all().item())
        print((batch_new["masking"][:, select] == batch_old["masking"]).all().item())
        print((batch_new["actions"][select] == batch_old["actions"]).all().item())

        print(str(res_old[i].item()))
        print(str(res_new[i].item()))
        print("\n")


if __name__ == "__main__":
    # test_network()
    # benchmark_forward()
    # test_dataset()
    # test_transforms()
    # test_baselines_qualitatively()
    # test_baseline_input_invariance()
    # test_metrics()
    # test_pck_auc()
    # test_batched_rigid()
    compare_old_and_new_collate()
