import torch
import numpy as np
import matplotlib.pyplot as plt

import conversion
import prediction
from visualisation import *


####################################################
# Don't judge this code, the deadline is this week #
####################################################


def draw_still(poses, azimuth, elevation, remove, save=False, lim=False, dpi=False, draw_frame_fn=False, ax=None):
    """
    Visualizes still poses via Matplotlib (i.e. no movement) similar to visualisation.draw_still_poses.
    The key difference is that the camera angles can be defined and that the axis scale can be removed.

    Args:
        poses: The (batch x 25 x 3) torch tensor containing the pose.
        azimuth: The azimuth camera angle in degrees.
        elevation: The elevation of the camera in degrees.
        remove: If true, remove scale of axis (tick labels).
        save: The name of the file to save to. If False, visualize instead.
        lim: The lower and upper bound for x-scale, should have length 2.
        dpi: If false, store as pdf, else with corresponding dpi. Only relevant if save=True.
        draw_frame_fn: If provided, use function pointer instead of visualisation.draw_frame().
        ax: A previous plot axis object. If None, a new plot is created.
    """
    meta_info = interpret_tensor("metrabs")

    if ax is None:
        fig = plt.figure(1)
        ax = configure(fig)
        suppress_legend = False
    else:
        suppress_legend = True

    if not lim:
        lim = [-1, 1]
    else:
        assert lim[1] - lim[0] == 2
    ax.set_xlim3d(lim[0], lim[1])
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    if remove:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
    ax.view_init(elev=elevation, azim=azimuth)
    if not draw_frame_fn:
        draw_frame(ax, poses, meta_info)
    else:
        draw_frame_fn(ax, poses, meta_info, suppress_legend)
    if not save:
        plt.show()
    elif not dpi:
        plt.savefig(save + ".pdf", bbox_inches="tight")
    else:
        plt.savefig(save + ".png", bbox_inches="tight", dpi=dpi)
    return ax


def amass():
    """
    Visualizes interesting Amass sequences.
    """
    # path to Amass
    path = "/work/erlbeck/models_motion/outputs/"
    trans = "bones_model_Nov19-01:32:44_tr-inp-bon.npz"
    lstm = "bones_model_Nov24-23:04:58_lstm-lay2.npz"
    gru = "bones_model_Nov24-23:04:28_gru-lay2.npz"
    tra_amass = np.load(path + "claix18_models/out_amass_comparison_" + trans)
    lst_amass = np.load(path + "rnn_final/out_amass_comparison_" + lstm)
    gru_amass = np.load(path + "rnn_final/out_amass_comparison_" + gru)

    # settings for Amass
    con = conversion.CoordinateConverter("bones", torch.device("cpu"))
    frame_ids = [9, 19, 24]  # corresponds to 0.4s, 0.8s, 1.0s at 25Hz
    batch_ids = [9, 3]
    seque_ids = [40, 127]
    azimuth = [-90, -110]
    elevation = [0, 30]
    remove = [True, True]
    lim = [[-1, 1], [0, 2]]

    for i, j, azi, ele, rem, lim in zip(batch_ids, seque_ids, azimuth, elevation, remove, lim):
        # extract predictions and ground truth and convert to locations
        p_tra = tra_amass[f"{i}_prediction"]
        t_tra = tra_amass[f"{i}_target"]
        p_lst = lst_amass[f"{i}_prediction"]
        t_lst = lst_amass[f"{i}_target"]
        p_gru = gru_amass[f"{i}_prediction"]
        t_gru = gru_amass[f"{i}_target"]
        p_tra, t_tra, _, _ = con.decode(torch.tensor(p_tra), torch.tensor(t_tra))
        p_lst, t_lst, _, _ = con.decode(torch.tensor(p_lst), torch.tensor(t_lst))
        p_gru, t_gru, _, _ = con.decode(torch.tensor(p_gru), torch.tensor(t_gru))

        # rnn batches are sorted by length but transformer batches are not
        # resort both to align
        _, perm_t = torch.sort(t_tra[0, :, 0, 0])
        _, perm_l = torch.sort(t_lst[0, :, 0, 0])
        p_tra, t_tra = p_tra[:, perm_t, :, :], t_tra[:, perm_t, :, :]
        p_lst, t_lst = p_lst[:, perm_l, :, :], t_lst[:, perm_l, :, :]
        p_gru, t_gru = p_gru[:, perm_l, :, :], t_gru[:, perm_l, :, :]
        assert (t_tra == t_lst).all() and (t_lst == t_gru).all()

        # only return one ground truth as assert ensures equality
        seq = [torch.stack([x_t, x_l, x_g, gt], dim=0)
               for x_t, x_l, x_g, gt
               in zip(p_tra[frame_ids, j, :, :],
                      p_lst[frame_ids, j, :, :],
                      p_gru[frame_ids, j, :, :],
                      t_gru[frame_ids, j, :, :])]

        # visualize with following color code:
        # red: ground truth
        # beige: GRU
        # cyan: LSTM
        # purple: Transformer
        # draw_sequence(seq, "metrabs")
        draw_still(seq[0], azi, ele, rem, f"amass_{i}_12", lim)
        draw_still(seq[1], azi, ele, rem, f"amass_{i}_24", lim)
        draw_still(seq[2], azi, ele, rem, f"amass_{i}_30", lim)


def kinect():
    """
    Visualizes interesting Kinect sequences.
    """
    # path to Kinect
    path = "/work/erlbeck/models_motion/outputs/"
    trans = "bones_model_Nov28-18:32:05_kin-tr-fi4.npz"
    lstm = "bones_model_Dec01-18:53:56_kin-lstm.npz"
    gru = "bones_model_Nov28-17:43:08_kin-gru-fi4.npz"
    tra_kinect = np.load(path + "claix18_models/out_kinect_comparison_" + trans)
    lst_kinect = np.load(path + "rnn_final/out_kinect_comparison_" + lstm)
    gru_kinect = np.load(path + "rnn_final/out_kinect_comparison_" + gru)

    # settings for Kinect
    frame_ids = [11, 23, 29]  # corresponds to 0.4s, 0.8s, 1.0s at 30Hz
    con = conversion.CoordinateConverter("bones", torch.device("cpu"))

    # extract predictions and ground truth and convert to locations
    tra, lst, gru = tra_kinect, lst_kinect, gru_kinect
    p_tra = tra[f"19_prediction"]
    t_tra = tra[f"19_target"]
    a_tra = tra[f"19_actions"]
    p_lst = lst[f"19_prediction"]
    t_lst = lst[f"19_target"]
    a_lst = lst[f"19_actions"]
    p_gru = gru[f"19_prediction"]
    t_gru = gru[f"19_target"]
    a_gru = gru[f"19_actions"]
    p_tra, t_tra, _, _ = con.decode(torch.tensor(p_tra), torch.tensor(t_tra))
    p_lst, t_lst, _, _ = con.decode(torch.tensor(p_lst), torch.tensor(t_lst))
    p_gru, t_gru, _, _ = con.decode(torch.tensor(p_gru), torch.tensor(t_gru))

    # only keep hugging action
    p_tra = p_tra[:, a_tra == 216, :, :]
    t_tra = t_tra[:, a_tra == 216, :, :]
    p_lst = p_lst[:, a_lst == 216, :, :]
    t_lst = t_lst[:, a_lst == 216, :, :]
    p_gru = p_gru[:, a_gru == 216, :, :]
    t_gru = t_gru[:, a_gru == 216, :, :]

    # select an even index (person 1) and the consecutive one (person 2)
    t_i1 = 0 * 2
    t_i2 = t_i1 + 1

    # find corresponding ground truths and check
    human1 = torch.argmax((t_lst[0, :, 0, 0] == t_tra[0, t_i1, 0, 0]).float()).item()
    human2 = torch.argmax((t_lst[0, :, 0, 0] == t_tra[0, t_i2, 0, 0]).float()).item()
    assert (t_tra[:, t_i1, :, :] == t_lst[:, human1, :, :]).all() and (t_tra[:, t_i1, :, :] == t_gru[:, human1, :, :]).all()
    assert (t_tra[:, t_i2, :, :] == t_lst[:, human2, :, :]).all() and (t_tra[:, t_i2, :, :] == t_gru[:, human2, :, :]).all()

    # stack predictions and visualize with following color code:
    # red: ground truth
    # beige: GRU
    # cyan: LSTM
    # purple: Transformer
    seq = [torch.stack([x_t, x_l, x_g, gt], dim=0)  # person 1
           for x_t, x_l, x_g, gt
           in zip(p_tra[frame_ids, :, :, :][:, t_i1, :, :],
                  p_lst[frame_ids, :, :, :][:, human1, :, :],
                  p_gru[frame_ids, :, :, :][:, human1, :, :],
                  t_gru[frame_ids, :, :, :][:, human1, :, :])]
    draw_still(seq[0], 45, 30, True, "kinect_1_12")
    draw_still(seq[1], 45, 30, True, "kinect_1_24")
    draw_still(seq[2], 45, 30, True, "kinect_1_30")
    seq = [torch.stack([x_t, x_l, x_g, gt], dim=0)  # person 2
           for x_t, x_l, x_g, gt
           in zip(p_tra[frame_ids, :, :, :][:, t_i2, :, :],
                  p_lst[frame_ids, :, :, :][:, human2, :, :],
                  p_gru[frame_ids, :, :, :][:, human2, :, :],
                  t_gru[frame_ids, :, :, :][:, human2, :, :])]
    draw_still(seq[0], 135, 30, True, "kinect_2_12")
    draw_still(seq[1], 135, 30, True, "kinect_2_24")
    draw_still(seq[2], 135, 30, True, "kinect_2_30")


def plot_errors():
    str = ["Zero Baseline", "Const Baseline", "LSTM", "GRU", "Transformer"]
    abs = [254.7, 444.9, 366.7, 152.2, 200.9]
    rel = [119.4, 425.7, 79.2, 97.3, 100.0]

    # axis break idea: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html

    # create two subplots: one for data, one for outlier
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 4]})
    fig.subplots_adjust(hspace=0.05)
    ax1.set_ylim(408, 443)
    ax2.set_ylim(60, 129)
    plt.ylabel("rel error")
    plt.xlabel("abs error")

    # plot and annotate
    ax1.scatter(abs, rel, c=["lime", "green", "blue", "royalblue", "navy"])
    ax2.scatter(abs, rel, c=["lime", "green", "blue", "royalblue", "navy"])
    for i, txt in enumerate(str):
        # text slightly offset to appear next to dot
        ax1.annotate(txt, (abs[i] - 66, rel[i] + 4))
        ax2.annotate(txt, (abs[i], rel[i] + 2))
    ax2.axhline(67.2, color="r")
    ax2.annotate("Mao 2020", (150, 67.2 + 1))

    # remove axis and ticks
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.set_xticks([])

    # slanted lines to show axis break
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    # visualize arrow to show best model
    # ax1.annotate("", xy=(180, 415), xytext=(200, 435), arrowprops=dict(arrowstyle="->"))
    # ax1.annotate("better", (195, 423))

    plt.savefig("amass_comparison.png", dpi=300)


def draw_frame_4_colors(ax, poses, meta_info, suppress_legend=False):
    """
    Draws a frame of the sequence. Has predefined colors and uses legend.
    Expected order: transformer, LSTM, GRU, ground truth

    Args:
        ax: The axis object of the 3D plot.
        poses: The torch tensor of an arbitrary number of poses.
        meta_info: The tuple returned by interpret_tensor.
        suppress_legend: If True, do not create legend.
    """
    poses = poses.numpy()
    rainbow = ["c", "b", "m", "r"]
    labels = ["TF", "LSTM", "GRU", "real"]
    edges, head = meta_info
    for pose in poses:
        color = rainbow.pop(0)

        # draw point at head to see if pose == torch.zeros(25,3)
        # providing colors as list avoids warning
        # also swap y- and z-axis and mirror y-axis
        # => x right, y depth, z up (also right-handed)
        ax.scatter(pose[head, 0], pose[head, 2], (-1) * pose[head, 1], "o", c=[color], label=labels.pop(0))

        # Next line can be uncommented to provide text labels for each pose (variable text needs to be defined first)
        # ax.text(pose[i, 0], pose[i, 2], (-1) * pose[i, 1], "{}".format(str(text)))

        # draw all the edges
        for edge in edges:
            # extract start and end coordinates from matrix
            matrix = pose[[edge[0], edge[1]], :]
            # swap y- and z-axis and mirror y-axis
            # => x right, y depth, z up (also right-handed)
            ax.plot(matrix[:, 0], matrix[:, 2], (-1) * matrix[:, 1], "-", c=color)
    if not suppress_legend:
        ax.legend()


def amass_with_gt():
    """
    Visualizes interesting Amass sequences. Also fetches input.
    """
    # path to models
    device = torch.device("cpu")
    path_gru = "/globalwork/erlbeck/models_motion/models/rnn_final/model_Nov24-23:04:28_gru-lay2.pt"
    path_lstm = "/globalwork/erlbeck/models_motion/models/rnn_final/model_Nov24-23:04:58_lstm-lay2.pt"
    path_tra = "/work/erlbeck/models_motion/models/claix18_models/model_Nov19-01:32:44_tr-inp-bon.pt"

    # load data and run model
    network_gru, opt, _, collate_fn, extend_to_two = prediction.load_model(path_gru, kinect=False)
    network_lst, opt2, _, collate_fn2, extend_to_two2 = prediction.load_model(path_lstm, kinect=False)
    network_tra, optt, _, collate_fnt, extend_to_twot = prediction.load_model(path_tra, kinect=False)

    assert collate_fn == collate_fn2 and extend_to_two == extend_to_two2 and opt == opt2
    assert extend_to_two == extend_to_twot and opt == optt

    network_gru.to(device)
    network_lst.to(device)
    network_tra.to(device)
    loader = prediction.load_test_data(testing=True, kinect=False, device=device, opt=opt, collate_fn=collate_fn, extend_to_two=extend_to_two)
    loader_tra = prediction.load_test_data(testing=True, kinect=False, device=device, opt=optt, collate_fn=collate_fnt, extend_to_two=extend_to_twot)
    network_gru.eval()
    network_lst.eval()
    network_tra.eval()
    converter = conversion.CoordinateConverter(opt["input_mode"], device)

    seque_ids = [127, 40]
    azimuth = [-110, -90]
    elevation = [30, 0]
    lim = [[0, 2], [-1, 1]]

    # set seed to 42 for evaluation
    torch.manual_seed(42)
    with torch.no_grad():
        iter_t = iter(loader_tra)
        for i, batch in enumerate(loader):
            batch_tra = next(iter_t)
            if i == 3 or i == 9:
                transformed_l = converter.encode(batch["input"])
                transformed_t = converter.encode(batch_tra["input"])
                output_gru = network_gru(transformed_l, batch["lengths"], opt["output_length"])
                output_lst = network_lst(transformed_l, batch["lengths"], opt["output_length"])
                output_tra = network_tra(transformed_t, batch_tra["lengths"], optt["output_length"])

                p_gru, t_gru, _, _ = converter.decode(output_gru, batch["target"])
                p_lst, t_lst, _, _ = converter.decode(output_lst, batch["target"])
                p_tra, t_tra, _, _ = converter.decode(output_tra, batch_tra["target"])

                # rnn batches are sorted by length but transformer batches are not
                # resort both to align
                _, perm_t = torch.sort(t_tra[0, :, 0, 0])
                _, perm_l = torch.sort(t_lst[0, :, 0, 0])
                p_tra, t_tra = p_tra[:, perm_t, :, :], t_tra[:, perm_t, :, :]
                p_lst, t_lst = p_lst[:, perm_l, :, :], t_lst[:, perm_l, :, :]
                p_gru, t_gru = p_gru[:, perm_l, :, :], t_gru[:, perm_l, :, :]
                assert (t_tra == t_lst).all() and (t_lst == t_gru).all()

                sid = seque_ids.pop(0)
                azi = azimuth.pop(0)
                ele = elevation.pop(0)
                lim = lim.pop(0)

                seq_input = [x.view(1, 24, 3) * torch.ones(4, 1, 1) for x in batch["input"][:, perm_l, :][:, sid, :]]
                seq = [torch.stack([x_t, x_l, x_g, gt], dim=0)
                       for x_t, x_l, x_g, gt
                       in zip(p_tra[:, sid, :, :],
                              p_lst[:, sid, :, :],
                              p_gru[:, sid, :, :],
                              t_gru[:, sid, :, :])]
                all = seq_input + seq

                for k in range(len(all)):
                    print(k)
                    draw_still(all[k], azi, ele, True, f"videos/better_amass_{i}_{k:02d}", lim, 300, draw_frame_4_colors)


def kinect_with_gt():
    """
    Visualizes interesting Kinect sequences. Also fetches input.
    """
    # path to models
    device = torch.device("cpu")
    path_gru = "/globalwork/erlbeck/models_motion/models/rnn_final/model_Nov28-17:43:08_kin-gru-fi4.pt"
    path_lstm = "/globalwork/erlbeck/models_motion/models/rnn_final/model_Dec01-18:53:56_kin-lstm.pt"
    path_tra = "/work/erlbeck/models_motion/models/claix18_models/model_Nov28-18:32:05_kin-tr-fi4.pt"

    # load data and run model
    network_gru, opt, _, collate_fn, extend_to_two = prediction.load_model(path_gru, kinect=True)
    network_lst, opt2, _, collate_fn2, extend_to_two2 = prediction.load_model(path_lstm, kinect=True)
    network_tra, optt, _, collate_fnt, extend_to_twot = prediction.load_model(path_tra, kinect=True)

    assert collate_fn == collate_fn2 and extend_to_two == extend_to_two2 and opt == opt2
    assert opt == optt

    network_gru.to(device)
    network_lst.to(device)
    network_tra.to(device)
    loader = prediction.load_test_data(testing=True, kinect=True, device=device, opt=opt, collate_fn=collate_fn, extend_to_two=extend_to_two)
    loader_tra = prediction.load_test_data(testing=True, kinect=True, device=device, opt=optt, collate_fn=collate_fnt, extend_to_two=extend_to_twot)
    network_gru.eval()
    network_lst.eval()
    network_tra.eval()
    converter = conversion.CoordinateConverter(opt["input_mode"], device)

    seque_ids = [127, 40]
    azimuth = [-110, -90]
    elevation = [30, 0]
    lim = [[0, 2], [-1, 1]]

    # set seed to 42 for evaluation
    torch.manual_seed(42)
    with torch.no_grad():
        iter_t = iter(loader_tra)
        for i, batch in enumerate(loader):
            batch_tra = next(iter_t)
            if i == 19:
                transformed_l = converter.encode(batch["input"])
                transformed_t = converter.encode(batch_tra["input"])
                output_gru = network_gru(transformed_l, batch["lengths"], opt["output_length"])
                output_lst = network_lst(transformed_l, batch["lengths"], opt["output_length"])
                output_tra = network_tra(transformed_t, batch_tra["lengths"], optt["output_length"])

                p_gru, t_gru, _, _ = converter.decode(output_gru, batch["target"])
                p_lst, t_lst, _, _ = converter.decode(output_lst, batch["target"])
                p_tra, t_tra, _, _ = converter.decode(output_tra, batch_tra["target"])

                # only keep hugging action
                a_tra = batch_tra["actions"]
                a_lst = a_gru = batch["actions"]
                p_tra = p_tra[:, a_tra == 216, :, :]
                t_tra = t_tra[:, a_tra == 216, :, :]
                p_lst = p_lst[:, a_lst == 216, :, :]
                t_lst = t_lst[:, a_lst == 216, :, :]
                p_gru = p_gru[:, a_gru == 216, :, :]
                t_gru = t_gru[:, a_gru == 216, :, :]

                # select an even index (person 1) and the consecutive one (person 2)
                t_i1 = 0 * 2
                t_i2 = t_i1 + 1

                # find corresponding ground truths and check
                human1 = torch.argmax((t_lst[0, :, 0, 0] == t_tra[0, t_i1, 0, 0]).float()).item()
                human2 = torch.argmax((t_lst[0, :, 0, 0] == t_tra[0, t_i2, 0, 0]).float()).item()
                assert (t_tra[:, t_i1, :, :] == t_lst[:, human1, :, :]).all() and (t_tra[:, t_i1, :, :] == t_gru[:, human1, :, :]).all()
                assert (t_tra[:, t_i2, :, :] == t_lst[:, human2, :, :]).all() and (t_tra[:, t_i2, :, :] == t_gru[:, human2, :, :]).all()

                # concat inputs and outputs
                seq_in1 = [x.view(1, 24, 3) * torch.ones(4, 1, 1) for x in batch["input"][:, a_gru == 216, :][:, human1, :]]
                seq_out1 = [torch.stack([x_t, x_l, x_g, gt], dim=0)  # person 1
                            for x_t, x_l, x_g, gt
                            in zip(p_tra[:, :, :, :][:, t_i1, :, :],
                                   p_lst[:, :, :, :][:, human1, :, :],
                                   p_gru[:, :, :, :][:, human1, :, :],
                                   t_gru[:, :, :, :][:, human1, :, :])]
                seq1 = seq_in1 + seq_out1

                seq_in2 = [x.view(1, 24, 3) * torch.ones(4, 1, 1) for x in batch["input"][:, a_gru == 216, :][:, human2, :]]
                seq_out2 = [torch.stack([x_t, x_l, x_g, gt], dim=0)  # person 2
                            for x_t, x_l, x_g, gt
                            in zip(p_tra[:, :, :, :][:, t_i2, :, :],
                                   p_lst[:, :, :, :][:, human2, :, :],
                                   p_gru[:, :, :, :][:, human2, :, :],
                                   t_gru[:, :, :, :][:, human2, :, :])]
                seq2 = seq_in2 + seq_out2

                for k in range(len(seq1)):
                    print(k)
                    ax = draw_still(seq1[k], -90, 30, True, dpi=300, draw_frame_fn=draw_frame_4_colors)
                    draw_still(seq2[k], -90, 30, True, f"videos/kinect_{k:02d}.png", dpi=300, draw_frame_fn=draw_frame_4_colors, ax=ax)


def amass_only_ground_truth():
    """
    Visualizes interesting Amass sequences for problem formulation slide.
    """
    # path to data
    path = "/work/erlbeck/models_motion/outputs/"
    gru = "bones_model_Nov24-23:04:28_gru-lay2.npz"
    gru_amass = np.load(path + "rnn_final/out_amass_comparison_" + gru)

    # extract data
    frame_ids = [0, 8, 16, 24]
    con = conversion.CoordinateConverter("bones", torch.device("cpu"))
    p_gru = gru_amass["3_prediction"]
    t_gru = gru_amass["3_target"]
    p_gru, t_gru, _, _ = con.decode(torch.tensor(p_gru), torch.tensor(t_gru))

    # sort
    _, perm_l = torch.sort(t_gru[0, :, 0, 0])
    p_gru, t_gru = p_gru[:, perm_l, :, :], t_gru[:, perm_l, :, :]
    seq = [x.view(1, 24, 3) for x in t_gru[frame_ids, 127, :, :]]
    seq[2] = seq[2] * torch.ones(2, 1, 1)  # duplicate to obtain different coloring
    seq[3] = seq[3] * torch.ones(2, 1, 1)

    # visualize
    draw_still(seq[0], -110, 30, True, f"amass_3_00", [0, 2])
    draw_still(seq[1], -110, 30, True, f"amass_3_08", [0, 2])
    draw_still(seq[2], -110, 30, True, f"amass_3_16", [0, 2])
    draw_still(seq[3], -110, 30, True, f"amass_3_24", [0, 2])


def vis_pos_enc():
    """
    Visualizes positional encoding.
    """
    x = np.arange(0, 4 * np.pi, 0.1)
    y1 = np.sin(x)
    plt.figure(1)
    plt.plot(x, np.sin(x / 1.00), label="Feature 1")
    plt.plot(x, np.sin(x / 1.54), label="Feature 7")
    plt.plot(x, np.sin(x / 2.37), label="Feature 13")
    plt.legend()
    plt.xlabel("Position")
    plt.ylabel("Encoding")
    plt.savefig("positional_enc.png", dpi=300)



if __name__ == "__main__":
    # amass()
    # kinect()
    # plot_errors()
    # amass_with_gt()
    # kinect_with_gt()
    # amass_only_ground_truth()
    vis_pos_enc()