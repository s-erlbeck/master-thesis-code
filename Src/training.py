import subprocess
import argparse
import os
import importlib
from datetime import datetime

import torch
import torchvision.transforms as tf
import wandb

import dataset as data
import rnn
import transformer as tr
import preprocess_metrabs as pm
import evaluation as eval
import conversion as con


class MaskedMotionLoss(torch.nn.Module):
    """
    Error criterion for motion prediction which masks padded frame from backpropagation.
    """

    def __init__(self, loss, weight_abs, input_mode, device):
        """
        Initialises the loss.

        Args:
            loss: Either "l1" or "l2" to specify the desired point error criterion.
            weight_abs: The importance of the absolute loss term.
                If input_mode is set to "rel", this will be set to zero.
            input_mode: The string describing the current input representation ("abs", "rel", "mixed" or "bones").
            device: The device where the network is located.
        """
        super(MaskedMotionLoss, self).__init__()
        self.converter = con.CoordinateConverter(input_mode, device)

        if loss == "l1":
            self.loss = torch.nn.L1Loss(reduction="none")
        elif loss == "l2":
            self.loss = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(loss + " is unknown.")

        if not 0 <= weight_abs <= 1:
            raise ValueError(f"Weight needs to be from [0, 1], but was found to be {weight_abs}.")

        if input_mode == "rel":
            print("Input mode is root-relative, set absolute loss term to zero.")
            self.weight_abs = 0.0
        else:
            self.weight_abs = weight_abs

    def loss_helper(self, predicted, target, masking):
        """
        Internal helper function computing the masked loss.

        Args:
            predicted: The predicted poses.
            target: The target poses.
            masking: The masking of invalid target frames.
        Returns:
            The resulting loss.
        """
        # for each pose, compute masked loss
        point_loss = torch.sum(self.loss(predicted, target), dim=3)  # sum over three entries to get distance
        pose_loss = torch.mean(point_loss, dim=2)  # mean -> invariance to number of joints
        masked_pose_loss = pose_loss * masking

        # separate normalisation over time and over batch
        # otherwise a short motion would contribute less even if per-frame-loss is constant in batch
        # cannot use mean function due to possibly padded frames (dim 0) or duplicate person (dim 1)
        temporal_sum = torch.sum(masked_pose_loss, dim=0)
        lengths = torch.sum(masking, dim=0)
        # lengths should only be zero where temporal_sum is zero => avoid zero division error there
        # in case length would be zero elsewhere, loss should become NaN to make clear that there is a bug
        lengths[temporal_sum == 0] = 1
        sequence_loss = temporal_sum / lengths
        batch_loss = torch.sum(sequence_loss) / torch.sum(masking, dim=1)[0]
        return batch_loss

    def forward(self, predicted, target, masking):
        """
        The forward pass of the loss module.

        Args:
            predicted: The predicted poses in the previously specified input representation.
            target: The target poses in absolute coordinates.
            masking: The masking of the padded frames.
        Returns:
            The loss of the minibatch.
        """
        pred_abs, target_abs, pred_rel, target_rel = self.converter.decode(predicted, target)
        loss_abs = self.loss_helper(pred_abs, target_abs, masking)
        loss_rel = self.loss_helper(pred_rel, target_rel, masking)
        return self.weight_abs * loss_abs + (1 - self.weight_abs) * loss_rel


def training(dir, config_file, log, kinect, transformer):
    """
    Trains a neural network on the dataset and stores resulting weights on disk.

    Args:
        dir: The name of the directory for the checkpoints.
        config_file: The configuration file used for training, located in configs.
            If not specified, config.py from root is used instead.
        log: Flag whether logging should be enabled.
        kinect: Flag whether Kinect data should be used instead of Amass.
        transformer: If true, a transformer model is trained instead of an RNN.
    """
    torch.manual_seed(42)

    if not os.path.isdir(f"/globalwork/erlbeck/models_motion/models/{dir}"):
        raise OSError(f"Path not found: /globalwork/erlbeck/models_motion/models/{dir}/")

    # load the correct configuration file depending on user input
    if config_file is not None:
        config_module = importlib.import_module(f"configs.{config_file}")
        name = config_file.replace("_", "-")  # during evaluation, _ is used as delimiter in file names
    else:
        config_module = importlib.import_module("config")
        name = None
    opt_d = config_module.opt_data
    opt_t = config_module.opt_training
    if transformer:
        opt_n = config_module.opt_transformer
    else:
        opt_n = config_module.opt_rnn
    options = {"data": opt_d,
               "train": opt_t,
               "net": opt_n}

    date = datetime.now()
    if log:
        # do this here already so that all print statements are stored in W&B as well
        wandb.init(project="Motion_Modelling_Transformer", entity="embreaded", config=options)
        if name is not None:
            wandb.run.name = name

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transforms = tf.Compose([tf.Lambda(data.transform_flip),
                             tf.Lambda(data.transform_rotate),
                             tf.Lambda(data.transform_scale)])

    if transformer:
        if opt_n["person_attn"] or kinect:
            num_people = 2
            extend_to_two = True
        else:
            # dataset is single-person and user does not want to have person attention
            num_people = 1
            extend_to_two = False
        collate_fn = data.get_transformer_collate_fn(num_people)
        network = tr.MotionTransformer(num_layers=opt_n["num_layers"],
                                       embedding_dim=opt_n["num_embedding"],
                                       hidden_dim=opt_n["num_hidden"],
                                       num_heads=opt_n["num_heads"],
                                       dropout=opt_n["dropout"],
                                       len_in=opt_d["input_length"],
                                       len_out=opt_d["output_length"],
                                       person_attn=opt_n["person_attn"],
                                       scale_input=opt_n["scale_input"],
                                       temp_mask=opt_n["temporal_mask"]).to(device)
    else:
        # RNN expects batch dimension to equal people dimension (always models people independently)
        extend_to_two = False
        collate_fn = data.collate_action_sequences
        network = rnn.SingleMotionPredictor(lstm=opt_n["lstm"],
                                            num_layers=opt_n["num_layers"],
                                            dim_hidden=opt_n["num_hidden"],
                                            dropout=opt_n["dropout"]).to(device)

    if kinect:
        split_p = data.split_pkummd()
        split_n = data.split_nturgbd()
        split_a = {"training": set(), "validation": set()}
        print("Training on Kinect data...")
    else:
        split_p = {"training": set(), "validation": set()}
        split_n = {"training": set(), "validation": set()}
        split_a = data.split_amass()
        print("Training on Amass data...")

    training_set = data.ExhaustiveSet(split_p["training"], "/globalwork/erlbeck/datasets/pkummd_enhanced/final/",
                                      split_n["training"], "/globalwork/erlbeck/datasets/nturgbd_enhanced/normalised/",
                                      split_a["training"], "/globalwork/erlbeck/datasets/amass_mao_version/",
                                      device, augmentation=transforms, use_short=True,
                                      use_idle=opt_d["use_idle"],
                                      skip_rate=opt_d["skip_rate"],
                                      filter_threshold=opt_d["threshold_filter"],
                                      input_length=opt_d["input_length"],
                                      output_length=opt_d["output_length"],
                                      frame_rate=opt_d["frame_rate"])

    validation_set = data.ExhaustiveSet(split_p["validation"], "/globalwork/erlbeck/datasets/pkummd_enhanced/final/",
                                        split_n["validation"], "/globalwork/erlbeck/datasets/nturgbd_enhanced/normalised/",
                                        split_a["validation"], "/globalwork/erlbeck/datasets/amass_mao_version/",
                                        device, augmentation=None, use_short=True,
                                        use_idle=True,
                                        skip_rate=120,  # use a large skip rate so that validation is fast
                                        filter_threshold=opt_d["threshold_filter"],
                                        input_length=opt_d["input_length"],
                                        output_length=opt_d["output_length"],
                                        frame_rate=opt_d["frame_rate"])

    print("Size of training set:", len(training_set))
    print("Size of validation set:", len(validation_set))

    train_loader = torch.utils.data.DataLoader(training_set,
                                               batch_size=opt_t["batch_size"],
                                               shuffle=True,
                                               collate_fn=collate_fn)

    validation_loader = torch.utils.data.DataLoader(validation_set,
                                                    batch_size=128,  # always use large batch for speed-up
                                                    shuffle=False,
                                                    collate_fn=collate_fn)

    if not transformer and opt_n["init_glorot"]:
        print("Using Glorot and orthogonal weight initialisation...")
        network.apply(rnn.init_weights)
    else:
        print("Using default pytorch weight initialisation...")

    converter = con.CoordinateConverter(opt_d["input_mode"], device)

    if opt_d["normalise"]:
        print("Normalising features to zero-mean unit variance...")
        mean, dev = training_set.compute_stats(converter)
        network.set_mean_and_std(mean, dev)

    if extend_to_two:
        # after network is aware of data mean and std dev, duplicate single-person for correct transformer input format
        training_set.extend_to_two()
        validation_set.extend_to_two()

    criterion = MaskedMotionLoss(opt_t["loss"], opt_t["importance_abs"], opt_d["input_mode"], device)

    optim = torch.optim.AdamW(network.parameters(), lr=opt_t["learning_rate"], betas=opt_t["betas"],
                              weight_decay=opt_t["weight_decay"])

    if "transformer_lr" in opt_n and opt_n["transformer_lr"]:
        # use 4000 warm-up steps like "Attention is all you need"
        scheduler_transformer = tr.TransformerLRScheduler(optim, network.embedding_dim, 4000)
        # call step so that initial learning rate is overridden
        scheduler_transformer.step()
    else:
        scheduler_transformer = None
        scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=opt_t["lr_exp_decay"])
        scheduler_step = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=opt_t["milestones"])

    if log:
        wandb.watch(network)

    if opt_t["inc_len_rate"] > 0:
        output_len = 0
    else:
        output_len = opt_d["output_length"]

    for epoch in range(opt_t["num_epochs"]):

        error = 0
        loss_dict = {}  # reset every epoch to enable logging every x-th epoch
        outputs = []
        print("Started training epoch", epoch)

        if output_len < opt_d["output_length"]:
            output_len = epoch // opt_t["inc_len_rate"] + 1
            print(f"Network is trained to predict {output_len} frames.")

        # training epoch
        for i, batch in enumerate(train_loader):
            # zero the gradients
            optim.zero_grad()

            # forward
            transformed = converter.encode(batch["input"])
            output = network(transformed, batch["lengths"], output_len)
            loss = criterion(output, batch["target"][:output_len, :, :], batch["masking"][:output_len, :])

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), opt_t["clip_grad_norm"])
            optim.step()
            if scheduler_transformer is not None:
                scheduler_transformer.step()

            error += loss.item()
            outputs.append(torch.max(torch.abs(output)).item())

        # compute metrics on validation set
        if epoch % 1 == 0 and log:
            validation(network, validation_loader, criterion, converter, loss_dict, opt_d)

        # store checkpoint
        if epoch % 10 == 0 and epoch > 0:
            torch.save({"options": options,
                        "epoch": epoch,
                        "network": network.state_dict(),
                        "optimizer": optim.state_dict(),
                        "date": date.strftime("%b/%d %H:%M:%S"),
                        }, f"/globalwork/erlbeck/models_motion/models/{dir}/checkpoint_{epoch:03d}.pt")

        # decay learning rate after current epoch
        if scheduler_transformer is None:
            if epoch in opt_t["milestones"]:
                print(f"Decaying learning rate to {scheduler_step.get_last_lr()[0]:.1e}...")
            scheduler_exp.step()
            scheduler_step.step()

        # log results
        if log:
            loss_dict["loss/training"] = error / len(train_loader)
            loss_dict["loss/output_mag"] = max(outputs)
            wandb.log(loss_dict)

    # training finished
    print("Finished training.")

    if name is None:
        name = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    torch.save({"options": options,
                "network": network.state_dict(),
                "date": date.strftime("%b/%d %H:%M:%S"),
                }, f"/globalwork/erlbeck/models_motion/models/{dir}/model_{date.strftime('%b%d-%H:%M:%S')}_{name}.pt")


def validation(network, validation_loader, criterion, converter, loss_dict, opt_d):
    """
    Computes useful metrics on validation split to show current performance of network.

    Args:
        network: The network which is being trained.
        validation_loader: The data loader for validation set.
        criterion: The loss object used in training.
        converter: The coordinate converter to transform input representations.
        loss_dict: The dictionary where metrics are stored for logging in wandb.
        opt_d: The options specified by the user.
    """
    network.eval()
    device = next(network.parameters()).device
    state = torch.get_rng_state()
    if torch.cuda.is_available():
        state_cuda = torch.cuda.get_rng_state(device)

    validation_loss = 0
    mpj_abs = eval.MPJPE(opt_d["output_length"], device)
    pck_abs = eval.PCK(opt_d["output_length"], device, 50)
    auc_abs = eval.PCK_AUC(opt_d["output_length"], device, 110)
    mpj_rel = eval.MPJPE(opt_d["output_length"], device)
    pck_rel = eval.PCK(opt_d["output_length"], device, 50)
    auc_rel = eval.PCK_AUC(opt_d["output_length"], device, 110)
    mpj_han = eval.MPJPE(opt_d["output_length"], device)
    hand_idx = [pm.get_metrabs_joint_by_name("l_han"), pm.get_metrabs_joint_by_name("r_han")]

    # fix seed to always evaluate on same splits
    torch.manual_seed(42)
    with torch.no_grad():
        for batch in validation_loader:
            transformed = converter.encode(batch["input"])
            output = network(transformed, batch["lengths"], opt_d["output_length"])
            loss = criterion(output, batch["target"], batch["masking"])
            pred_abs, targ_abs, pred_rel, targ_rel = converter.decode(output, batch["target"])
            dist_abs = eval.euclidean_distances(pred_abs, targ_abs)
            dist_rel = eval.euclidean_distances(pred_rel, targ_rel)

            # update evaluation metrics
            mpj_abs.update(batch["actions"], dist_abs, batch["masking"])
            pck_abs.update(batch["actions"], dist_abs, batch["masking"])
            auc_abs.update(batch["actions"], dist_abs, batch["masking"])
            mpj_rel.update(batch["actions"], dist_rel, batch["masking"])
            pck_rel.update(batch["actions"], dist_rel, batch["masking"])
            auc_rel.update(batch["actions"], dist_rel, batch["masking"])
            mpj_han.update(batch["actions"], dist_abs[:, :, hand_idx], batch["masking"])
            validation_loss += loss.item()

    # evaluate the metrics and write to log
    res_mpj_abs = mpj_abs.evaluate(False)[0]
    res_pck_abs = pck_abs.evaluate(False)[0]
    res_auc_abs = auc_abs.evaluate(False)[0]
    res_mpj_rel = mpj_rel.evaluate(False)[0]
    res_pck_rel = pck_rel.evaluate(False)[0]
    res_auc_rel = auc_rel.evaluate(False)[0]
    res_mpj_han = mpj_han.evaluate(False)[0]

    loss_dict["absolute/mpjpe_t=01"] = res_mpj_abs[0].item()
    loss_dict["absolute/mpjpe_t=12"] = res_mpj_abs[11].item()
    loss_dict["absolute/mpjpe_t=30"] = res_mpj_abs[29].item()

    loss_dict["absolute/mean/mpjpe"] = res_mpj_abs.mean().item()
    loss_dict["absolute/mean/pck_50"] = res_pck_abs.mean().item()
    loss_dict["absolute/mean/pck_auc"] = res_auc_abs.mean().item()

    loss_dict["relative/mpjpe_t=01"] = res_mpj_rel[0].item()
    loss_dict["relative/mpjpe_t=12"] = res_mpj_rel[11].item()
    loss_dict["relative/mpjpe_t=30"] = res_mpj_rel[29].item()

    loss_dict["relative/mean/mpjpe"] = res_mpj_rel.mean().item()
    loss_dict["relative/mean/pck_50"] = res_pck_rel.mean().item()
    loss_dict["relative/mean/pck_auc"] = res_auc_rel.mean().item()

    loss_dict["hand_pos/mpjpe_t=01"] = res_mpj_han[0].item()
    loss_dict["hand_pos/mpjpe_t=12"] = res_mpj_han[11].item()
    loss_dict["hand_pos/mpjpe_t=30"] = res_mpj_han[29].item()

    loss_dict["loss/validation"] = validation_loss / len(validation_loader)

    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state_cuda, device)
    torch.set_rng_state(state)
    network.train()


def main():
    """
    The main functionality wrapping arg parsing and training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="specify the subdirectory where model is stored")
    parser.add_argument("-c", "--config_file", help="specify the configuration of network and training")
    parser.add_argument("-l", "--log", help="enable logging via \"Weights & Biases\"", action="store_true")
    parser.add_argument("-k", "--kinect", help="train on Kinect data instead of Amass", action="store_true")
    parser.add_argument("-t", "--transformer", help="use transformer instead of RNN", action="store_true")
    args = parser.parse_args()
    training(**vars(args))


if __name__ == "__main__":
    main()
