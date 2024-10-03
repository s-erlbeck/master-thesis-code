import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    """
    Computes positional encoding using sine and cosine waves.
    Code is adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout, len_in, len_out):
        """
        Initialises the internal vectors.

        Args:
            d_model: The dimensionality of the embedded vectors of the model.
            dropout: The dropout probability.
            len_in: The maximum amount of input frames.
            len_out: The maximum amount of output frames.
        """
        super(PositionalEncoding, self).__init__()
        self.len_in = len_in
        self.len_out = len_out
        self.dropout = nn.Dropout(p=dropout)

        # position are the frame indices
        position = torch.arange(-len_in, len_out).unsqueeze(1)  # add dimension to compute outer product
        # the exponent of the negative logarithm turns the factor into a divisor
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe is the positional encoding which is added
        pe = torch.zeros(len_in + len_out, 1, 1, d_model)  # dimensions 1 for batch and people
        pe[:, 0, 0, 0::2] = torch.sin(position * div_term)  # outer product between position and divisor
        pe[:, 0, 0, 1::2] = torch.cos(position * div_term)  # outer product between position and divisor
        self.register_buffer('pe', pe)

    def forward(self, src, current_in, current_out):
        """
        Applies the additive positional encoding to the input.

        Args:
            src: The input tensor of shape seq_len x batch_size x #people x embedding_dim.
            current_in: The maximum input sequence length of the batch.
            current_out: The number of frames to predict (should not exceed len_out).
        """
        src = src + self.pe[(self.len_in - current_in):(self.len_in + current_out), :, :, :]
        return self.dropout(src)


class CustomAttentionLayer(nn.Module):
    """
    A layer of a typical transformer, adapted to motion forecasting.
    Consists of a temporal attention module and a two-layer feed-forward perceptron.
    Both modules apply dropout, skip connections and layer normalisation.
    """
    def __init__(self, embedding_dim, hidden_dim, num_heads, dropout, person_attn):
        """
        Initialises the custom attention layer.

        Args:
            embedding_dim: The dimensionality of the embedding vectors, i.e., of input and output.
            hidden_dim: The dimension of the hidden state of the two-layer perceptron.
            num_heads: The number of attention heads applied in parallel. Must divide embedding_dim.
            dropout: The dropout probability.
            person_attn: The flag whether to use person attention as well.
        """
        super(CustomAttentionLayer, self).__init__()

        # code adapted from pytorch
        self.time_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, add_zero_attn=True)

        # either use person-attention or skip that sublayer
        if person_attn:
            # only use 2 heads because there is a maximum of 2 people
            self.person_attn = PersonAttention(embedding_dim, 2, dropout=dropout)
        else:
            # cannot use torch.nn.Identity, because that one expects one argument only
            self.person_attn = self.identity

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def identity(self, src, padding_mask):
        """
        Dummy function to replace person attention if desired.

        Args:
            src: The input tensor.
            padding_mask: A padding mask for person attention, which is ignored
        Returns:
            The input src unaltered.
        """
        return src

    def forward(self, src, padding_mask, temp_mask):
        """
        The forward pass of this custom attention layer.

        Args:
            src: The input or the output of the previous layer of size seq_len x batch_size x #people x embedding_dim.
            padding_mask: The keys and values to ignore because their tokens correspond to padding.
                A value of True masks the corresponding token.
                Is expected to be of size seq_len x batch_size x #people.
            temp_mask: The mask to filter invalid query-key combination. Commonly used to filter future inputs.
                A value of True masks the corresponding pair.
                Is expected to be of size seq_len x seq_len.
        Returns:
            The output of dimension seq_len x batch_size x #people embedding_dim.
        """
        # code adapted from pytorch
        t, b, p, f = src.size()

        # temporal attention
        frames = src.view(t, b * p, f)
        temp_padding = padding_mask.view(t, b * p)
        frames2 = self.time_attn(frames, frames, frames, key_padding_mask=temp_padding.T, attn_mask=temp_mask)[0]
        src2 = frames2.view(t, b, p, f)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # person attention / identity
        src = self.person_attn(src, padding_mask)

        # feed forward network
        src2 = self.linear2(self.dropout(torch.nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class PersonAttention(nn.Module):
    """
    A layer that wraps the person attention module.
    While temporal attention is always necessary, this layer may be undesired so it is wrapped into a class.
    """
    def __init__(self, embedding_dim, num_heads, dropout):
        """
        Initialises the person attention module.

        Args:
            embedding_dim: The dimensionality of the embedding vectors, i.e., of input and output.
            num_heads: The number of attention heads applied in parallel. Must divide embedding_dim.
            dropout: The dropout probability.
        """
        super(PersonAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, add_zero_attn=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, src, padding_mask):
        """
        The forward pass of the attention layer.

        Args:
            src: The input or the output of the previous layer of size seq_len x batch_size x #people x embedding_dim.
            padding_mask: The keys and values to ignore because their tokens correspond to padding.
                A value of True masks the corresponding token.
                Is expected to be of size seq_len x batch_size x #people.
        Returns:
            The output of dimension seq_len x batch_size x #people x embedding_dim.
        """
        t, b, p, f = src.size()

        people = src.permute(2, 0, 1, 3).view(p, t * b, f)
        person_padding = padding_mask.permute(2, 0, 1).view(p, t * b)
        people2 = self.attn(people, people, people, key_padding_mask=person_padding.T)[0]
        src2 = people2.view(p, t, b, f).permute(1, 2, 0, 3)
        src = src + self.dropout(src2)
        src = self.norm(src)

        return src


class MotionTransformer(nn.Module):
    """
    The transformer for motion modeling.
    Only consists of an encoder, as the data is unimodal (pose frames only)
        unlike e.g. a transformer for translation (source language vs target language).
    """
    def __init__(self, num_layers, embedding_dim, hidden_dim, num_heads, dropout,
                 len_in, len_out, person_attn, scale_input, temp_mask):
        """
        Initialises the network.

        Args:
            num_layers: The number of custom attention layers in the model.
            embedding_dim: The dimensionality of the embedding vectors.
            hidden_dim: The dimension of the hidden state of the two-layer perceptrons.
            num_heads: The number of attention heads applied in parallel. Must divide embedding_dim.
            dropout: The dropout probability.
            len_in: The maximum input length.
            len_out: The maximum output length.
            person_attn: The flag whether to use person attention as well.
            scale_input: The flag whether to scale the embedding like the original transformer.
            temp_mask: The flag indicating whether attention should only apply backwards through time.
                Unlike traditional transformers, our transformer sees an in interpolation of the future
                instead of the real future so temporal masking is optional.
        """
        super(MotionTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.len_in = len_in
        self.len_out = len_out

        if scale_input:
            self.scale = math.sqrt(self.embedding_dim)
        else:
            self.scale = 1

        # store the mask to prevent temporal attention to look into future
        all_true = torch.ones(len_in + len_out, len_in + len_out, dtype=torch.bool)
        if temp_mask:
            self.register_buffer("temp_mask", torch.triu(all_true, diagonal=1))
        else:
            self.register_buffer("temp_mask", torch.logical_not(all_true))

        # init layers
        self.embedding = nn.Linear(72, embedding_dim)
        self.pos_enc = PositionalEncoding(embedding_dim, dropout, len_in, len_out)
        self.layers = nn.ModuleList()
        self.layers.extend(
            [CustomAttentionLayer(embedding_dim=embedding_dim,
                                  hidden_dim=hidden_dim,
                                  num_heads=num_heads,
                                  dropout=dropout,
                                  person_attn=person_attn)
             for i in range(num_layers)])
        self.projection = nn.Linear(embedding_dim, 72)

        # init data statistics
        self.register_buffer("data_mean", torch.zeros(1, 1, 1, 24 * 3, dtype=torch.float))
        self.register_buffer("data_std", torch.ones(1, 1, 1, 24 * 3, dtype=torch.float))

    def set_mean_and_std(self, mean, dev):
        """
        Setter for data mean and data standard deviation.

        Args:
            mean: The mean.
            dev: The standard deviation.
        """
        self.data_mean[:] = mean
        self.data_std[:] = dev

    def forward(self, batch_u, padding_mask, output_length):
        """
        The forward method of the transformer.

        Args:
            batch_u: The unnormalised batched input and the interpolation of the output.
                Is expected to be of size seq_len x batch_size x #people x (#joints * 3)
            padding_mask: The keys and values to ignore because their tokens correspond to padding.
                A value of True masks the corresponding token.
                Is expected to be of seq_len x batch_size x #people.
            output_length: The number of frames to predict, must be at most len_out.
        Returns:
            The output of size output_length x (batch_size * #people) x (#joints * 3).
        """
        # cut off later parts in case of output length schedule
        max_len = batch_u.size(0) - self.len_out
        batch_u = batch_u[:(max_len + output_length), :, :, :]
        padding_mask = padding_mask[:(max_len + output_length), :, :]
        score_mask = self.temp_mask[:(max_len + output_length), :(max_len + output_length)]

        # normalise and embed
        batch = (batch_u - self.data_mean) / self.data_std
        tokens = self.pos_enc(self.scale * self.embedding(batch), max_len, output_length)

        # run network
        for layer in self.layers:
            tokens = layer(tokens, padding_mask, score_mask)

        # back-project and residual connection around network
        change = self.projection(tokens[max_len:, :, :, :])
        output = batch[max_len - 1, :, :, :].unsqueeze(0) + torch.cumsum(change, dim=0)
        output_norm = output * self.data_std + self.data_mean
        return output_norm.view(output_length, -1, batch.size(-1))


class TransformerLRScheduler():
    """
    Implements learning rate schedule of original transformer.
    The class mimics the schedulers from pytorch but does not inherit the base class
        because these schedulers insist on an initial learning rate.
    """
    def __init__(self, optimizer, d_model, warmup_steps):
        """
        Initialises the scheduler. The learning rate starts at a small value, increases
        linearly for warmup_steps updates and then uses exponential decay.

        Args:
            optimizer: The optimizer used for learning.
            d_model: The embedding dimension of the transformer.
            warmup_steps: The number of steps to increase the learning rate for.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """
        Computes a new learning rate and sets it to all parameter groups.
        Should be called once before the first update to override the initial learning rate.
        """
        current_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = current_lr

    def get_lr(self):
        """
        Computes the new learning rate. Is called implicitly via step()

        Returns:
            The new learning rate.
        """
        self.current_step += 1
        decay = self.current_step ** (-0.5)
        warmup = self.current_step * self.warmup_steps ** (-1.5)
        return self.d_model ** (-0.5) * min(decay, warmup)
