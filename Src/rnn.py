import torch.nn as nn
import torch


class SingleMotionPredictor(nn.Module):

    def __init__(self, lstm, num_layers, dim_hidden, dropout):
        """
        Initialises the network.

        Args:
            lstm: If true use an LSTM, else use GRU.
            num_layers: The number of stacked RNN layers.
            dim_hidden: The dimensionality of the hidden state.
            dropout: The probability that RNN outputs are set to zero.
        """
        super(SingleMotionPredictor, self).__init__()

        # init layers
        if lstm:
            self.rnn = nn.LSTM(input_size=24 * 3,
                               hidden_size=dim_hidden,
                               num_layers=num_layers,
                               dropout=dropout)
            self.map_internal_rnn = self.map_internal_lstm
        else:
            self.rnn = nn.GRU(input_size=24 * 3,
                              hidden_size=dim_hidden,
                              num_layers=num_layers,
                              dropout=dropout)
            self.map_internal_rnn = self.map_internal_gru
        self.linear = nn.Linear(dim_hidden, 24 * 3)

        # init data statistics
        self.register_buffer("data_mean", torch.zeros(1, 1, 24 * 3, dtype=torch.float))
        self.register_buffer("data_std", torch.ones(1, 1, 24 * 3, dtype=torch.float))

    def set_mean_and_std(self, mean, dev):
        """
        Setter for data mean and data standard deviation.

        Args:
            mean: The mean.
            dev: The standard deviation.
        """
        self.data_mean[:] = mean
        self.data_std[:] = dev

    def map_internal_lstm(self, internal):
        """
        Helper method to extract hidden state from internal LSTM state.

        Args:
            internal: The internal state of the LSTM.
        Returns:
            The hidden state of the LSTM.
        """
        return internal[0][-1, :, :]

    def map_internal_gru(self, internal):
        """
        Helper method to extract hidden state from internal GRU state.

        Args:
            internal: The internal state of the GRU.
        Returns:
            The hidden state of the GRU.
        """
        return internal[-1, :, :]

    def forward(self, batch_u, lengths, output_length):
        """
        Forwards a batch through the network.
        The samples are expected to be sorted descendingly by temporal length.

        Args:
            batch_u: The zero-padded unnormalised batch of dimensions time x batch x features
            lengths: The lengths in descending order.
            output_length: The number of frames to be predicted into the future.
        Returns:
            The future poses as predicted by the network.
        """
        batch = (batch_u - self.data_mean) / self.data_std
        batch_dim = batch.size()[1]

        # Encoder:
        pack = nn.utils.rnn.pack_padded_sequence(batch, lengths - 1)  # do not feed last known pose to encoder
        _, internal = self.rnn(pack)  # output irrelevant for encoder

        # Decoder:
        feedback = batch[lengths - 1, list(range(batch_dim)), :]  # init output feedback loop to last known ground truth
        feedback = feedback[None, :, :]  # add temporal dimension
        output = batch.new_empty(output_length, batch_dim, 24 * 3)  # allocate on correct device
        for t in range(output_length):  # output feedback loop
            _, internal = self.rnn(feedback, internal)
            change = self.linear(self.map_internal_rnn(internal))
            # do not use inplace operation (matters because of autograd):
            feedback = feedback + change
            # do not detach to have additional gradient flow through feedback as well:
            output[t, :, :] = feedback

        return output * self.data_std + self.data_mean


def init_weights(layer):
    """
    Initialises the weights of the RNN-architecture according to best practices.
    Usage: network.apply(init_weights)

    Args:
        layer: The layer which weights and biases are initialised inplace.
    """
    if isinstance(layer, nn.Linear):
        # yay, easy
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.RNNBase):
        # RNN weights are stacked into 4 matrices:
        # weight and bias for the input connection and for the recurrent connection each
        weights = layer.state_dict()
        # reconstruct the network architecture
        dim_hidden = weights["weight_hh_l0"].size(1)
        num_layers = len(weights) // 4
        num_gates = weights["weight_hh_l0"].size(0) // dim_hidden

        # gain computation only works for LSTM and GRU
        if not (num_gates == 3 or num_gates == 4):
            raise TypeError("Expects GRU or LSTM, otherwise gain might be wrong.")
        # for each layer
        for l in range(num_layers):
            # for each gate (3 for GRU, 4 for LSTM)
            for g in range(num_gates):
                # gate weights are concatenated into one matrix, so we need the indices
                start = g * dim_hidden
                end = (g + 1) * dim_hidden

                # the second gate (for GRU and LSTM) uses tanh instead of sigmoid
                if g == 2:
                    gain = nn.init.calculate_gain("tanh")
                else:
                    gain = nn.init.calculate_gain("sigmoid")

                # use orthogonal for recurrent weight due to gradient stability
                nn.init.xavier_uniform_(weights[f"weight_ih_l{l}"][start:end, :], gain)
                nn.init.orthogonal_(weights[f"weight_hh_l{l}"][start:end, :], gain)
                # bias to zero
                nn.init.zeros_(weights[f"bias_ih_l{l}"][start:end])
                nn.init.zeros_(weights[f"bias_hh_l{l}"][start:end])
