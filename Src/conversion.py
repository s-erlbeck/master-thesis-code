import torch

import preprocess_metrabs as pm


class CoordinateConverter:
    """
    Class to convert poses from and to camera coordinates (absolute coordinates).
    Using this class allows to write code which is agnostic of the pose representation.
    """

    def __init__(self, input_mode, device):
        """
        Initialises the converter.

        Args:
            input_mode: The string describing the current input representation ("abs", "rel", "mixed" or "bones").
            device: The device where the network is located.
        """
        # use function pointers to avoid this case distinction inside encode / decode function
        # this is important as encode / decode will be called a lot during training
        if input_mode == "abs":
            self.ptr_encode = self.to_absolute
            self.ptr_decode = self.from_absolute
        elif input_mode == "rel":
            self.ptr_encode = self.to_relative
            self.ptr_decode = self.from_relative
        elif input_mode == "mixed":
            self.ptr_encode = self.to_mixed
            self.ptr_decode = self.from_mixed
        elif input_mode == "bones":
            self.ptr_encode = self.to_bones
            self.ptr_decode = self.from_bones
        else:
            raise ValueError(str(input_mode) + " is not a supported input mode.")

        self.rel_mat = pm.get_root_relative_matrix(False).to(device)
        self.root_id = pm.get_metrabs_joint_by_name("pelvi")
        self.mixed_mat = pm.get_root_relative_matrix(True).to(device)
        self.mixed_inv = torch.inverse(self.mixed_mat)
        self.bone_mat = pm.get_bone_vector_matrix().to(device)
        self.bone_inv = torch.inverse(self.bone_mat)

    def encode(self, history):
        """
        Transforms the poses to whatever representation was passed to input_mode.
        Expects dimension TxBx72 (times, batch, 24 joints in 3D flattened).

        Args:
            history: The known history in absolute coordinates.
        Returns:
            The history in whatever representation was passed to input_mode as TxBx72.
        """
        return self.ptr_encode(history)

    def to_absolute(self, history):
        """
        Internal function of converter class.

        Args:
            history: The known history in absolute coordinates.
        Returns:
            The history in absolute coordinates as TxBx72.
        """
        return history

    def to_relative(self, history):
        """
        Internal function of converter class.

        Args:
            history: The known history in absolute coordinates.
        Returns:
            The history in root-relative coordinates as TxBx72.
        """
        return history @ self.rel_mat.T

    def to_mixed(self, history):
        """
        Internal function of converter class.

        Args:
            history: The known history in absolute coordinates.
        Returns:
            The history in mixed coordinates as TxBx72.
        """
        return history @ self.mixed_mat.T

    def to_bones(self, history):
        """
        Internal function of converter class.

        Args:
            history: The known history in absolute coordinates.
        Returns:
            The history as bone vectors as TxBx72.
        """
        return history @ self.bone_mat.T

    def decode(self, predicted, target):
        """
        Transforms the poses to absolute representation.
        Expects dimension TxBx72.
        All outputs have dimension TxBx24x3.

        Args:
            predicted: The predicted poses in whatever representation was passed to input_mode.
            target: The target poses in absolute coordinates.
        Returns:
            The predicted poses in absolute coordinates.
            The target poses in absolute coordinates.
            The predicted poses in root-relative coordinates.
            The target poses in root-relative coordinates.
        """
        return self.ptr_decode(predicted, target)

    def reshape(self, *args):
        """
        Internal function of converter class to reshape the pose representations.

        Args:
            *args: A variable amount of tensors of dimension T x B x 72.
        Returns:
            A tuple of the tensors with dimension T x B x 24 x 3.
        """
        dim = torch.Size((args[0].size(0), args[0].size(1), 24, 3))
        return tuple(map(lambda x: x.view(dim), args))

    def from_absolute(self, predicted, target):
        """
        Internal function of converter class.

        Args:
            predicted: The predicted poses in absolute coordinates.
            target: The target poses.
        Returns:
            The predicted poses in absolute coordinates.
            The target poses in absolute coordinates.
            The predicted poses in root-relative coordinates.
            The target poses in root-relative coordinates.
        """
        return self.reshape(predicted,
                            target,
                            self.to_relative(predicted),
                            self.to_relative(target))

    def from_relative(self, predicted, target):
        """
        Internal function of converter class.

        Args:
            predicted: The predicted poses in root-relative coordinates.
            target: The target poses.
        Returns:
            The predicted poses in absolute coordinates.
            The target poses in absolute coordinates.
            The predicted poses in root-relative coordinates.
            The target poses in root-relative coordinates.
        """
        return self.reshape(predicted,
                            target,
                            torch.clone(predicted),
                            self.to_relative(target))

    def from_mixed(self, predicted, target):
        """
        Internal function of converter class.

        Args:
            predicted: The predicted poses in mixed coordinate representation.
            target: The target poses.
        Returns:
            The predicted poses in absolute coordinates.
            The target poses in absolute coordinates.
            The predicted poses in root-relative coordinates.
            The target poses in root-relative coordinates.
        """
        transformed = predicted @ self.mixed_inv.T
        return self.reshape(transformed,
                            target,
                            self.to_relative(transformed),
                            self.to_relative(target))

    def from_bones(self, predicted, target):
        """
        Internal function of converter class.

        Args:
            predicted: The predicted poses as bone vectors.
            target: The target poses.
        Returns:
            The predicted poses in absolute coordinates.
            The target poses in absolute coordinates.
            The predicted poses in root-relative coordinates.
            The target poses in root-relative coordinates.
        """
        transformed = predicted @ self.bone_inv.T
        return self.reshape(transformed,
                            target,
                            self.to_relative(transformed),
                            self.to_relative(target))
