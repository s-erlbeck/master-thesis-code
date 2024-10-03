import torch


class BaselineZeroMotion(torch.nn.Module):
    """
    Implements zero motion heuristic for motion prediction.
    """

    def __init__(self):
        super(BaselineZeroMotion, self).__init__()

    def forward(self, batch, lengths, output_length=30):
        """
        Applies heuristic on batches of poses.
        Expects camera coordinates (absolute coordinates) for compatibility with BaselineConstVelocity.

        Args:
            batch: The batch of past poses.
            lengths: The lengths of the sequences.
            output_length: The number of frames to predict.
        Returns:
            The predicted poses according to the heuristic.
        """
        # get last poses
        batch_dim = batch.size(1)
        last = batch[lengths - 1, list(range(batch_dim)), :]

        # repeat them
        res = last.repeat(output_length, 1, 1)
        return res


class BaselineConstVelocity(torch.nn.Module):
    """
    Implements constant velocity heuristic for motion prediction.
    Each joint is shifted independently using the motion of the last two known frames.
    """

    def __init__(self):
        super(BaselineConstVelocity, self).__init__()

    def forward(self, batch, lengths, output_length=30):
        """
        Applies heuristic on batches of poses.
        Expects camera coordinates (absolute coordinates).
        The reason is that the baseline should be invariant to the input representation for better comparabtility.
        Actually, it is invariant except for root-relative representation where global motion is lost.

        Args:
            batch: The batch of past poses.
            lengths: The lengths of the sequences.
            output_length: The number of frames to predict.
        Returns:
            The predicted poses according to the heuristic.
        """
        # get last frame motion
        batch_dim = batch.size(1)
        last = batch[lengths - 1, list(range(batch_dim)), :]
        sec_to_last = batch[lengths - 2, list(range(batch_dim)), :]
        diff = last - sec_to_last

        # construct result
        res = last.repeat(output_length, 1, 1)
        for i in range(output_length):
            res[i:, :, :] += diff
        return res
