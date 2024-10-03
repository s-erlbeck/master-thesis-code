from evaluation import *


def main():
    """
    Parses the program arguments and runs qualitative and quantitative tests and baselines accordingly.
    Use "-h" options to learn more on the available arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to the model output which was computed with prediction.py")
    parser.add_argument("-f", "--full", help="align prediction and ground truth with full Procrustes (all joints) "
                                             "to remove global rotation for evaluation,"
                                             "this flag only has an effect if 'relative' is provided and until "
                                             "is not provided and the dataset is Amass and the protocol is "
                                             "'comparison', otherwise global rotation IS evaluated", action="store_true")
    parser.add_argument("-r", "--relative", help="use root-relative poses for evaluation "
                                                 "instead of absolute coordinates", action="store_true")
    parser.add_argument("-u", "--until", help="use arithmetic mean over time steps "
                                              "instead of evaluating at each step independently", action="store_true")
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

    # iterate over 0.4 prediction and 1.0 prediction
    res = ["", ""]
    for i, t in enumerate([0.4, 1.0]):
        if protocol == "ablation":
            len_out = int(t * 30)
        elif protocol == "comparison":
            len_out = int(t * 25)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        res[i] = evaluate_quantitative(predictions, input_mode, len_out, args.relative, args.until,
                                       False, evaluate_joints, align_joints)

    # results
    print(f"Name: {model_name}\n"
          f"Date: {model_date}\n"
          f"Data: {dataset} in {input_mode} representation\n"
          f"Protocol: {protocol}\n"
          f"Evaluation: {'relative' if args.relative else 'absolute'}\n"
          f"Reduction: {'mean over t' if args.until else 'score at t'}\n")

    tex = ""
    tex += f"& {res[0][0][0]:7.1f} "         + f"& {res[1][0][0]:7.1f} &"
    tex += f"& {res[0][0][2] * 100:7.1f} "   + f"& {res[1][0][2] * 100:7.1f} &"
    tex += f"& {res[0][0][4]:7.3f} "         + f"& {res[1][0][4]:7.3f} "

    print(tex)


if __name__ == "__main__":
    main()
