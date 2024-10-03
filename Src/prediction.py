import argparse

import torch
import numpy as np

import baselines
import conversion
import dataset as data
import rnn as rnn
import transformer as tr
import visualisation


def load_model(path, kinect):
    """
    Loads the model. Assumes that config dictionary has all relevant entries.

    Args:
        path: The path to the model.
        kinect: The bool indicating whether the dataset is going to be the Kinect dataset.
    Returns:
        The loaded network.
        The loaded option dictionary.
        The torch device where the network is located (GPU if available).
        The collate function used to batch samples into a single tensor.
        The boolean whether to apply person padding.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading model at {path}...")
    dictionary = torch.load(path)
    opt = dictionary["options"]["data"]

    if "lstm" in dictionary["options"]["net"]:
        collate_fn = data.collate_action_sequences
        extend_to_two = False
        network = rnn.SingleMotionPredictor(dictionary["options"]["net"]["lstm"],
                                            dictionary["options"]["net"]["num_layers"],
                                            dictionary["options"]["net"]["num_hidden"],
                                            dictionary["options"]["net"]["dropout"])
    else:
        if dictionary["options"]["net"]["person_attn"] or kinect:
            collate_fn = data.get_transformer_collate_fn(2)
            extend_to_two = True
        else:
            collate_fn = data.get_transformer_collate_fn(1)
            extend_to_two = False
        network = tr.MotionTransformer(dictionary["options"]["net"]["num_layers"],
                                       dictionary["options"]["net"]["num_embedding"],
                                       dictionary["options"]["net"]["num_hidden"],
                                       dictionary["options"]["net"]["num_heads"],
                                       dictionary["options"]["net"]["dropout"],
                                       dictionary["options"]["data"]["input_length"],
                                       dictionary["options"]["data"]["output_length"],
                                       dictionary["options"]["net"]["person_attn"],
                                       dictionary["options"]["net"]["scale_input"],
                                       dictionary["options"]["net"]["temporal_mask"]).to(device)
    network.load_state_dict(dictionary["network"], strict=True)
    network.to(device)

    return network, opt, device, collate_fn, extend_to_two


def load_model_legacy(path):
    """
    Loads an old model. Should work properly on all models except for the oldest one, but no guarantees.

    Args:
        path: The path to the model.
    Returns:
        The loaded network.
        The loaded option dictionary.
        The torch device where the network is located (GPU if available).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading model at {path}...")
    dictionary = torch.load(path)

    if len(dictionary["options"].keys()) == 3:
        # config dictionary has new structure
        opt = dictionary["options"]["data"]
        network = rnn.SingleMotionPredictor(dictionary["options"]["net"]["lstm"],
                                            dictionary["options"]["net"]["num_layers"],
                                            dictionary["options"]["net"]["num_hidden"],
                                            dictionary["options"]["net"]["dropout"])
    else:
        # legacy dictionary, need to reconstruct whether this was an LSTM or a GRU...
        opt = dictionary["options"]
        n, hidden = dictionary["network"]["rnn.weight_hh_l0"].size()
        if n // hidden == 4:
            network = rnn.SingleMotionPredictor(True, 1, hidden, 0)
        elif n // hidden == 3:
            network = rnn.SingleMotionPredictor(False, 1, hidden, 0)
        else:
            raise TypeError("Could not recognise network layout.")

    # set old default values
    if "input_length" not in opt:
        opt["input_length"] = 4 * opt["output_length"]
    if "frame_rate" not in opt:
        opt["frame_rate"] = 30

    # buffers for data normalisation may be missing
    incompatible = network.load_state_dict(dictionary["network"], strict=False)
    if len(incompatible[0]) > 0 or len(incompatible[1]) > 0:
        # check for unexpected mismatch
        if len(incompatible[1]) != 0 or len(incompatible[0]) != 2 \
                or incompatible[0][0] != "data_mean" or incompatible[0][1] != "data_std":
            raise TypeError("State dictionary not compatible: " + str(incompatible))
        # only the buffers for data normalisation are missing
        # this means that network did not normalise data
        network.register_buffer(incompatible[0][0], torch.zeros(1, 1, 24 * 3, dtype=torch.float))
        network.register_buffer(incompatible[0][1], torch.ones(1, 1, 24 * 3, dtype=torch.float))

    network.to(device)

    return network, opt, device


def load_test_data(testing, kinect, device, opt, collate_fn, extend_to_two):
    """
    Loads the test data and creates a dataloader for iteration.

    Args:
        testing: The flag indicating whether the protocol for final evaluation or for hyperparameter study is chosen.
        kinect: The flag indicating whether to test on Kinect data or Amass data.
        device: The torch device where the network is located.
        opt: The options containing input length, output length, etc.
        collate_fn: The collate function used to batch samples into a single tensor.
        extend_to_two: The boolean whether to apply person padding.
    Returns:
        The resulting dataloader.
    """
    if testing:
        # use evaluation procedure (multi-person or literature protocol)
        split = "testing"
        use_idle = True
        skip_rate = 5

        if kinect:
            # using Kinect
            use_short = True
            opt["input_length"] = 60
            opt["output_length"] = 30
        else:
            # using Amass, therefore compare to Mao et al.
            use_short = False
            opt["input_length"] = 50
            opt["output_length"] = 25
            opt["frame_rate"] = 25
    else:
        # use ablation / hyperparameter protocol
        split = "validation"
        use_short = True
        use_idle = True
        skip_rate = 30
        assert opt["output_length"] == 30, "Should predict 1 second."
        assert opt["frame_rate"] == 30, "Downsampling to 30 fps is most accurate."

    if kinect:
        split_p = data.split_pkummd()[split]
        split_n = data.split_nturgbd()[split]
        split_a = set()
        print(f"Evaluating on Kinect {split} split...")
        print("Setting filter threshold to 0.4 as a compromise between test set size and quality.")
    else:
        split_p = set()
        split_n = set()
        split_a = data.split_amass()[split]
        print(f"Evaluating on Amass {split} split...")
        print(f"Sequences are downsampled to {opt['frame_rate']}fps. "
              f"Using {opt['input_length'] / opt['frame_rate']} sec input "
              f"and {opt['output_length'] / opt['frame_rate']} sec output.")
    print(f"Sample test sequence every {skip_rate} frames.")

    test_set = data.ExhaustiveSet(split_p, "/work/erlbeck/datasets/pkummd_enhanced/final/",
                                  split_n, "/work/erlbeck/datasets/nturgbd_enhanced/normalised/",
                                  split_a, "/work/erlbeck/datasets/amass_mao_version/",
                                  device, use_short=use_short, use_idle=use_idle,
                                  skip_rate=skip_rate, augmentation=None, filter_threshold=0.4,
                                  input_length=opt["input_length"],
                                  output_length=opt["output_length"],
                                  frame_rate=opt["frame_rate"])

    if extend_to_two:
        test_set.extend_to_two()

    # always use large batch size to speed up testing
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=collate_fn)

    return test_loader


def run_evaluation(model, model_path, data_loader, data_name, protocol_name, device, options):
    """
    Runs model on the test data in data_loader, storing all outputs for later usage.

    Args:
        model: The model used for prediction.
        model_path: The path to the model.
        data_loader: The dataloader for the test data.
        data_name: The name of the dataset.
        protocol_name: The name of evaluation protocol (either comparison or ablation).
        device: The device where data and model are located.
        options: The options, especially data representation and output length.
    """
    converter = conversion.CoordinateConverter(options["input_mode"], device)
    outputs = {}

    # set seed to 42 for evaluation
    torch.manual_seed(42)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            transformed = converter.encode(batch["input"])
            output = model(transformed, batch["lengths"], options["output_length"])

            outputs[f"{i}_prediction"] = output.cpu().numpy()
            outputs[f"{i}_target"] = batch["target"].cpu().numpy()
            outputs[f"{i}_masks"] = batch["masking"].cpu().numpy().astype(np.byte)
            outputs[f"{i}_actions"] = batch["actions"].numpy().astype(np.short)

    # parse model name and sub-directory from path
    split = model_path.split("/")
    model_name = split[-1].split(".")[0]
    dir = split[-2] + "/" if len(split) > 1 else ""
    # save
    np.savez_compressed(f"/work/erlbeck/models_motion/outputs/{dir}out_{data_name}_{protocol_name}_"
                        f"{options['input_mode']}_{model_name}", **outputs)


def visualize_data(network, data_loader, device, options):
    """
    Visualizes a model's predictions qualitatively. Does not call eval().

    Args:
        network: The network to evaluate.
        data_loader: The data loader from which the samples are drawn.
        device: The device where data and network are located.
        options: The data dictionary (with keys "output_length" and "input_mode").
    """
    converter = conversion.CoordinateConverter(options["input_mode"], device)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            transformed = converter.encode(batch["input"])
            output = network(transformed, batch["lengths"], options["output_length"])
            output_abs, target_abs, _, _ = converter.decode(output, batch["target"])

            for person in range(output.size(1)):
                # move to CPU and concatenate history and future
                if isinstance(network, tr.MotionTransformer):
                    padding = batch["lengths"].view(-1, output.size(1))[:, person]
                    if padding.all():
                        continue
                    not_padding = torch.logical_not(padding)
                    input_abs = batch["input"].view(-1, output.size(1), 24, 3)[not_padding, :, :, :][:-30, :, :, :]
                else:
                    seq_len = batch["lengths"][person]
                    input_abs = batch["input"][:seq_len, :, :].view(seq_len, -1, 24, 3)
                output_p = output_abs[:, person, :, :].cpu()
                target_p = target_abs[:, person, :, :].cpu()
                input_p = input_abs[:, person, :, :].cpu()
                predicted = torch.cat([input_p, output_p], dim=0)
                real = torch.cat([input_p, target_p], dim=0)

                # visualise
                seq = [torch.stack([x, y], dim=0) for x, y in zip(predicted, real)]
                visualisation.draw_sequence(seq, "metrabs")


def main():
    """
    Parses the program arguments and runs qualitative and quantitative tests and baselines accordingly.
    Use "-h" options to learn more on the available arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="either baseline_zero, baseline_const or the path to the model")
    parser.add_argument("-t", "--testing", help="use test split instead of validation split, "
                                                "switches to comparison / multi-person protocol", action="store_true")
    parser.add_argument("-k", "--kinect", help="test on Kinect data instead of Amass data", action="store_true")
    parser.add_argument("-l", "--legacy", help="use legacy function for loading the model, "
                                               "try it out if normal loading crashes...", action="store_true")
    parser.add_argument("-v", "--visualize", help="visualize the predictions", action="store_true")
    args = parser.parse_args()

    # default parameters for baselines, loading function might override them
    dev = torch.device("cpu")
    opt = {"input_mode": "abs",
           "input_length": 60,
           "output_length": 30,
           "frame_rate": 30}
    collate_fn = data.collate_action_sequences
    extend_to_two = False

    # load model
    if args.path == "baseline_zero":
        network = baselines.BaselineZeroMotion()
    elif args.path == "baseline_const":
        network = baselines.BaselineConstVelocity()
    elif args.legacy:
        network, opt, dev = load_model_legacy(args.path)
    else:
        network, opt, dev, collate_fn, extend_to_two = load_model(args.path, args.kinect)

    # load data and run model
    loader = load_test_data(args.testing, args.kinect, dev, opt, collate_fn, extend_to_two)
    network.eval()
    data_name = "kinect" if args.kinect else "amass"
    protocol_name = "comparison" if args.testing else "ablation"
    run_evaluation(network, args.path, loader, data_name, protocol_name, dev, opt)
    print("Stored results of evaluation...")

    if args.visualize:
        print("Visualizing predictions...")
        visualize_data(network, loader, dev, opt)


if __name__ == "__main__":
    main()
