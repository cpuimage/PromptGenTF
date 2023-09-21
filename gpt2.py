# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os

import numpy as np
import tensorflow as tf
import torch
from safetensors import safe_open


def split_c_proj(name, value, num_head=12, axis=0):
    c_proj_name = 'attn.c_proj.weight'
    str_name = str(name)
    if str_name.endswith(c_proj_name):
        if num_head is not None:
            value = np.reshape(value, (num_head, value.shape[axis] // num_head, value.shape[1]))
        return True, value
    return False, None


def split_attn(name, value, num_head=12, axis=0):
    weight_name = 'attn.c_attn.weight'
    bais_name = 'attn.c_attn.bias'
    str_name = str(name)
    if str_name.endswith(weight_name):
        q_value, k_value, v_value = np.split(np.transpose(value, (1, 0)), 3, axis=axis)
        if num_head is not None:
            q_value = np.reshape(q_value, (num_head, q_value.shape[axis] // num_head, q_value.shape[1]))
            k_value = np.reshape(k_value, (num_head, k_value.shape[axis] // num_head, k_value.shape[1]))
            v_value = np.reshape(v_value, (num_head, v_value.shape[axis] // num_head, v_value.shape[1]))

        q_name = str_name.replace(weight_name, 'attn.query.weight')
        k_name = str_name.replace(weight_name, 'attn.key.weight')
        v_name = str_name.replace(weight_name, 'attn.value.weight')
        return True, ((q_name, np.transpose(q_value, (2, 0, 1))), (k_name, np.transpose(k_value, (2, 0, 1))),
                      (v_name, np.transpose(v_value, (2, 0, 1))))
    elif str_name.endswith(bais_name):
        q_value, k_value, v_value = np.split(value, 3, axis=0)
        if num_head is not None:
            q_value = np.reshape(q_value, (num_head, q_value.shape[0] // num_head))
            k_value = np.reshape(k_value, (num_head, k_value.shape[0] // num_head))
            v_value = np.reshape(v_value, (num_head, v_value.shape[0] // num_head))
        q_name = str_name.replace(bais_name, 'attn.query.bias')
        k_name = str_name.replace(bais_name, 'attn.key.bias')
        v_name = str_name.replace(bais_name, 'attn.value.bias')
        return True, ((q_name, q_value), (k_name, k_value), (v_name, v_value))
    return False, None


def rebuild_state_dict(state_dict, num_head=12, axis=0):
    new_state_dict = {}
    state_dict_keys = list(state_dict.keys()).copy()
    for idx, key in enumerate(state_dict_keys):
        value = state_dict[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().numpy()
        active, qkv = split_attn(key, value, num_head=num_head, axis=axis)
        if active is False:
            new_state_dict[key] = value
        else:
            (q_name, q_value), (k_name, k_value), (v_name, v_value) = qkv
            new_state_dict[q_name] = q_value
            new_state_dict[k_name] = k_value
            new_state_dict[v_name] = v_value
        active, new_value = split_c_proj(key, value, num_head=num_head, axis=axis)
        if active is False:
            new_state_dict[key] = value
        else:
            new_state_dict[key] = new_value
    return new_state_dict


TORCH_CKPT_MAPPING = {'gpt2': [
    ('transformer.wte.weight', None),  # (50257, 768)
    ('transformer.wpe.weight', None),  # (1024, 768)
    ('transformer.h.0.ln_1.weight', None),  # (768,)
    ('transformer.h.0.ln_1.bias', None),  # (768,)
    ('transformer.h.0.attn.query.weight', None),  # (768, 12, 64)
    ('transformer.h.0.attn.query.bias', None),  # (12, 64)
    ('transformer.h.0.attn.key.weight', None),  # (768, 12, 64)
    ('transformer.h.0.attn.key.bias', None),  # (12, 64)
    ('transformer.h.0.attn.value.weight', None),  # (768, 12, 64)
    ('transformer.h.0.attn.value.bias', None),  # (12, 64)
    ('transformer.h.0.attn.c_proj.weight', None),  # (768, 12, 64)
    ('transformer.h.0.attn.c_proj.bias', None),  # (768,)
    ('transformer.h.0.ln_2.weight', None),  # (768,)
    ('transformer.h.0.ln_2.bias', None),  # (768,)
    ('transformer.h.0.mlp.c_fc.weight', None),  # (768, 3072)
    ('transformer.h.0.mlp.c_fc.bias', None),  # (3072,)
    ('transformer.h.0.mlp.c_proj.weight', None),  # (3072, 768)
    ('transformer.h.0.mlp.c_proj.bias', None),  # (768,)
    ('transformer.h.1.ln_1.weight', None),  # (768,)
    ('transformer.h.1.ln_1.bias', None),  # (768,)
    ('transformer.h.1.attn.query.weight', None),  # (768, 12, 64)
    ('transformer.h.1.attn.query.bias', None),  # (12, 64)
    ('transformer.h.1.attn.key.weight', None),  # (768, 12, 64)
    ('transformer.h.1.attn.key.bias', None),  # (12, 64)
    ('transformer.h.1.attn.value.weight', None),  # (768, 12, 64)
    ('transformer.h.1.attn.value.bias', None),  # (12, 64)
    ('transformer.h.1.attn.c_proj.weight', None),  # (768, 12, 64)
    ('transformer.h.1.attn.c_proj.bias', None),  # (768,)
    ('transformer.h.1.ln_2.weight', None),  # (768,)
    ('transformer.h.1.ln_2.bias', None),  # (768,)
    ('transformer.h.1.mlp.c_fc.weight', None),  # (768, 3072)
    ('transformer.h.1.mlp.c_fc.bias', None),  # (3072,)
    ('transformer.h.1.mlp.c_proj.weight', None),  # (3072, 768)
    ('transformer.h.1.mlp.c_proj.bias', None),  # (768,)
    ('transformer.h.2.ln_1.weight', None),  # (768,)
    ('transformer.h.2.ln_1.bias', None),  # (768,)
    ('transformer.h.2.attn.query.weight', None),  # (768, 12, 64)
    ('transformer.h.2.attn.query.bias', None),  # (12, 64)
    ('transformer.h.2.attn.key.weight', None),  # (768, 12, 64)
    ('transformer.h.2.attn.key.bias', None),  # (12, 64)
    ('transformer.h.2.attn.value.weight', None),  # (768, 12, 64)
    ('transformer.h.2.attn.value.bias', None),  # (12, 64)
    ('transformer.h.2.attn.c_proj.weight', None),  # (768, 12, 64)
    ('transformer.h.2.attn.c_proj.bias', None),  # (768,)
    ('transformer.h.2.ln_2.weight', None),  # (768,)
    ('transformer.h.2.ln_2.bias', None),  # (768,)
    ('transformer.h.2.mlp.c_fc.weight', None),  # (768, 3072)
    ('transformer.h.2.mlp.c_fc.bias', None),  # (3072,)
    ('transformer.h.2.mlp.c_proj.weight', None),  # (3072, 768)
    ('transformer.h.2.mlp.c_proj.bias', None),  # (768,)
    ('transformer.h.3.ln_1.weight', None),  # (768,)
    ('transformer.h.3.ln_1.bias', None),  # (768,)
    ('transformer.h.3.attn.query.weight', None),  # (768, 12, 64)
    ('transformer.h.3.attn.query.bias', None),  # (12, 64)
    ('transformer.h.3.attn.key.weight', None),  # (768, 12, 64)
    ('transformer.h.3.attn.key.bias', None),  # (12, 64)
    ('transformer.h.3.attn.value.weight', None),  # (768, 12, 64)
    ('transformer.h.3.attn.value.bias', None),  # (12, 64)
    ('transformer.h.3.attn.c_proj.weight', None),  # (768, 12, 64)
    ('transformer.h.3.attn.c_proj.bias', None),  # (768,)
    ('transformer.h.3.ln_2.weight', None),  # (768,)
    ('transformer.h.3.ln_2.bias', None),  # (768,)
    ('transformer.h.3.mlp.c_fc.weight', None),  # (768, 3072)
    ('transformer.h.3.mlp.c_fc.bias', None),  # (3072,)
    ('transformer.h.3.mlp.c_proj.weight', None),  # (3072, 768)
    ('transformer.h.3.mlp.c_proj.bias', None),  # (768,)
    ('transformer.h.4.ln_1.weight', None),  # (768,)
    ('transformer.h.4.ln_1.bias', None),  # (768,)
    ('transformer.h.4.attn.query.weight', None),  # (768, 12, 64)
    ('transformer.h.4.attn.query.bias', None),  # (12, 64)
    ('transformer.h.4.attn.key.weight', None),  # (768, 12, 64)
    ('transformer.h.4.attn.key.bias', None),  # (12, 64)
    ('transformer.h.4.attn.value.weight', None),  # (768, 12, 64)
    ('transformer.h.4.attn.value.bias', None),  # (12, 64)
    ('transformer.h.4.attn.c_proj.weight', None),  # (768, 12, 64)
    ('transformer.h.4.attn.c_proj.bias', None),  # (768,)
    ('transformer.h.4.ln_2.weight', None),  # (768,)
    ('transformer.h.4.ln_2.bias', None),  # (768,)
    ('transformer.h.4.mlp.c_fc.weight', None),  # (768, 3072)
    ('transformer.h.4.mlp.c_fc.bias', None),  # (3072,)
    ('transformer.h.4.mlp.c_proj.weight', None),  # (3072, 768)
    ('transformer.h.4.mlp.c_proj.bias', None),  # (768,)
    ('transformer.h.5.ln_1.weight', None),  # (768,)
    ('transformer.h.5.ln_1.bias', None),  # (768,)
    ('transformer.h.5.attn.query.weight', None),  # (768, 12, 64)
    ('transformer.h.5.attn.query.bias', None),  # (12, 64)
    ('transformer.h.5.attn.key.weight', None),  # (768, 12, 64)
    ('transformer.h.5.attn.key.bias', None),  # (12, 64)
    ('transformer.h.5.attn.value.weight', None),  # (768, 12, 64)
    ('transformer.h.5.attn.value.bias', None),  # (12, 64)
    ('transformer.h.5.attn.c_proj.weight', None),  # (768, 12, 64)
    ('transformer.h.5.attn.c_proj.bias', None),  # (768,)
    ('transformer.h.5.ln_2.weight', None),  # (768,)
    ('transformer.h.5.ln_2.bias', None),  # (768,)
    ('transformer.h.5.mlp.c_fc.weight', None),  # (768, 3072)
    ('transformer.h.5.mlp.c_fc.bias', None),  # (3072,)
    ('transformer.h.5.mlp.c_proj.weight', None),  # (3072, 768)
    ('transformer.h.5.mlp.c_proj.bias', None),  # (768,)
    ('transformer.ln_f.weight', None),  # (768,)
    ('transformer.ln_f.bias', None),  # (768,)
]
}


def _gpt_2_kernel_initializer(stddev=0.02):
    return tf.keras.initializers.RandomNormal(stddev=stddev)


def clone_initializer(initializer):
    """Clones an initializer to ensure a new seed.

    As of tensorflow 2.10, we need to clone user passed initializers when
    invoking them twice to avoid creating the same randomized initialization.
    """
    # If we get a string or dict, just return as we cannot and should not clone.
    if not isinstance(initializer, tf.keras.initializers.Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)


def _check_masks_shapes(inputs, padding_mask, attention_mask):
    mask = padding_mask
    if hasattr(inputs, "_keras_mask") and mask is None:
        mask = inputs._keras_mask
    if mask is not None:
        if len(mask.shape) != 2:
            raise ValueError(
                "`padding_mask` should have shape "
                "(batch_size, target_length). "
                f"Received shape `{mask.shape}`."
            )
    if attention_mask is not None:
        if len(attention_mask.shape) != 3:
            raise ValueError(
                "`attention_mask` should have shape "
                "(batch_size, target_length, source_length). "
                f"Received shape `{mask.shape}`."
            )


def compute_causal_mask(batch_size, input_length, output_length, cache_index=0):
    """Compute a causal attention mask for a transformer decoder.

    Args:
        batch_size: batch size for the mask.
        input_length: the length of key/value tensors in the attention layer.
        output_length: the length of query tensors in the attention layer.
        cache_index: the current index for cached generation. If passed, the
            query sequence will be considered to start at `cache_index` rather
            than zero. For example, a causal mask with `output_length=1` and
            `cache_index=5` would allow the query tensor to attend to the first
            five positions of the key/value tensors.

    Return:
        A causal attention mask with shape
        `(batch_size, output_length, input_length)` that can be passed to a
        attention layer.
    """
    i = tf.expand_dims(tf.range(output_length), axis=1) + cache_index
    j = tf.range(input_length)
    mask = tf.expand_dims(tf.cast(i >= j, dtype="int32"), axis=0)
    return tf.broadcast_to(mask, (batch_size, output_length, input_length))


def merge_padding_and_attention_mask(
        inputs,
        padding_mask,
        attention_mask,
):
    """Merge the padding mask with a customized attention mask.

    Args:
        inputs: the input sequence.
        padding_mask: the 1D padding mask, of shape
            [batch_size, sequence_length].
        attention_mask: the 2D customized mask, of shape
            [batch_size, sequence1_length, sequence2_length].

    Return:
        A merged 2D mask or None. If only `padding_mask` is provided, the
        returned mask is padding_mask with one additional axis.
    """
    _check_masks_shapes(inputs, padding_mask, attention_mask)
    mask = padding_mask
    if hasattr(inputs, "_keras_mask"):
        if mask is None:
            # If no padding mask is explicitly provided, we look for padding
            # mask from the input data.
            mask = inputs._keras_mask
        else:
            logging.warning(
                "You are explicitly setting `padding_mask` while the `inputs` "
                "have built-in mask, so the built-in mask is ignored."
            )
    if mask is not None:
        # Add an axis for broadcasting, the attention mask should be 2D
        # (not including the batch axis).
        mask = tf.cast(tf.expand_dims(mask, axis=1), "int32")
    if attention_mask is not None:
        attention_mask = tf.cast(attention_mask, "int32")
        if mask is None:
            return attention_mask
        else:
            return tf.minimum(mask, attention_mask)
    return mask


class PositionEmbedding(tf.keras.layers.Layer):
    """A layer which learns a position embedding for inputs sequences.

    This class assumes that in the input tensor, the last dimension corresponds
    to the features, and the dimension before the last corresponds to the
    sequence.

    This layer does not supporting masking, but can be combined with a
    `keras.layers.Embedding` for padding mask support.

    Args:
        sequence_length: The maximum length of the dynamic sequence.
        initializer: The initializer to use for the embedding weights. Defaults
            to `"glorot_uniform"`.
        seq_axis: The axis of the input tensor where we add the embeddings.

    Examples:


    Combine with a token embedding.
    ```python
    seq_length = 50
    vocab_size = 5000
    embed_dim = 128
    inputs =tf.keras.Input(shape=(seq_length,))
    token_embeddings =tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim
    )(inputs)
    position_embeddings = keras_nlp.layers.PositionEmbedding(
        sequence_length=seq_length
    )(token_embeddings)
    outputs = token_embeddings + position_embeddings
    ```

    Reference:
     - [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
    """

    def __init__(
            self,
            sequence_length,
            initializer="glorot_uniform",
            **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError(
                "`sequence_length` must be an Integer, received `None`."
            )
        self.sequence_length = int(sequence_length)
        self.initializer = tf.keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": tf.keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="weight",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, start_index=0):
        shape = tf.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        # trim to match the length of the input sequence, which might be less
        # than the sequence_length of the layer.
        position_embeddings = tf.convert_to_tensor(self.position_embeddings)
        position_embeddings = tf.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return tf.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape


def split(x, num_or_size_splits=1, axis=-1, use_tf_split=True):
    if x is None:
        return x
    if use_tf_split:
        return tf.split(x, num_or_size_splits=num_or_size_splits, axis=axis)
    num_elements = x.get_shape().as_list()[axis]
    indices = np.linspace(0, num_elements, num_or_size_splits + 1, dtype=np.int32)
    return [tf.gather(x, range(indices[index], indices[index + 1]), axis=axis) for index in range(len(indices) - 1)]


class CachedMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    """MultiHeadAttention layer with cache support.

    This layer is suitable for use in autoregressive decoding. It can be used
    to cache decoder self-attention and cross-attention. The forward pass
    can happen in one of three modes:

    - No cache, same as regular multi-head attention.
    - Static cache (`cache_update_index` is None). In this case, the
        cached key/value projections will be used and the input values will
        be ignored.
    - Updated cache (`cache_update_index` is not None). In this case, new
        key/value projections are computed using the input, and spliced into
        the cache at the specified index.

    Note that caching is useful only during inference and should not be used
    during training.

    We use the notation `B`, `T`, `S` below, where `B` is the batch dimension,
    `T` is the target sequence length, and `S` in the source sequence length.
    Note that during generative decoding, `T` is usually 1 (you are
    generating a target sequence of length one to predict the next token).

    Call arguments:
        query: Query `Tensor` of shape `(B, T, dim)`.
        value: Value `Tensor` of shape `(B, S*, dim)`. if `cache` is None`, `S*`
            must equal `S` and match the shape of `attention_mask`. If cache` is
            not `None`, `S*` can be any length less than `S`, and the computed
            value will be spliced into `cache` at `cache_update_index`.
        key: Optional key `Tensor` of shape `(B, S*, dim)`. If `cache` is
            `None`, `S*` must equal `S` and match the shape of
            `attention_mask`. If `cache` is not `None`, `S*` can be any length
            less than `S`, and the computed value will be spliced into `cache`
            at `cache_update_index`.
        attention_mask: a boolean mask of shape `(B, T, S)`. `attention_mask`
            prevents attention to certain positions. The boolean mask specifies
            which query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        cache: a dense float Tensor. The key/value cache, of shape
            `[B, 2, S, num_heads, key_dims]`, where `S` must agree with the
            `attention_mask` shape. This argument is intended for use during
            generation to avoid recomputing intermediate state.
        cache_update_index: a int or int Tensor, the index at which to update
            `cache` (usually the index of the current token being processed
            when running generation). If `cache_update_index=None` while `cache`
            is set, the cache will not be updated.

    Returns:
        An `(attention_output, cache)` tuple. `attention_output` is the result
        of the computation, of shape `(B, T, dim)`, where `T` is for target
        sequence shapes and `dim` is the query input last dimension if
        `output_shape` is `None`. Otherwise, the multi-head outputs are
        projected to the shape specified by `output_shape`. `cache` is the
        updated cache.
    """

    def call(
            self,
            query,
            value,
            key=None,
            attention_mask=None,
            cache=None,
            cache_update_index=None,
            cache_update_mask=None,
    ):
        if (
                hasattr(self, "_build_from_signature")
                and not self._built_from_signature
        ):
            self._build_from_signature(query=query, value=value, key=key)

        if key is None:
            key = value

        query = self._query_dense(query)

        # If cache is not `None`, we will use the cache to compute the final key
        # and value tensors. If `cache_update_index` is not None, we will first
        # update the cache before use. To do this, we first call the
        # `_key_dense` and `_value_dense` layers, and copy the outputs into the
        # cache at the specified index. `cache = None` handles the training
        # case, where we don't use the cache at all.
        if cache is not None:
            key_cache, value_cache = tf.unstack(cache, num=2, axis=1)
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update = self._key_dense(key)
                value_update = self._value_dense(value)
                if cache_update_mask is not None:
                    # todo cache 可以先乘完 inv_mask 再传进来
                    inv_mask = (1. - cache_update_mask)
                    key = key_cache * inv_mask + key_update * cache_update_mask
                    value = value_cache * inv_mask + value_update * cache_update_mask
                else:
                    start = [0, cache_update_index, 0, 0]
                    from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice
                    key = dynamic_update_slice(key_cache, key_update, start)
                    value = dynamic_update_slice(value_cache, value_update, start)
                cache = tf.stack((key, value), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )
            key = self._key_dense(key)
            value = self._value_dense(value)

        query = tf.multiply(
            query,
            1.0 / tf.sqrt(tf.cast(self._key_dim, query.dtype)),
        )
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = self._dropout_layer(attention_scores)

        attention_output = tf.einsum(
            self._combine_equation, attention_scores, value
        )
        attention_output = self._output_dense(attention_output)
        return attention_output, cache


class TransformerDecoder(tf.keras.layers.Layer):
    """Transformer decoder.

    This class follows the architecture of the transformer decoder layer in the
    paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
    can instantiate multiple instances of this class to stack up a decoder.

    By default, this layer will apply a causal mask to the decoder attention layer.
    This layer will correctly compute an attention mask from an implicit
    Keras padding mask (for example, by passing `mask_zero=True` to a
    `keras.layers.Embedding` layer). See the Masking and Padding
    [guide](https://keras.io/guides/understanding_masking_and_padding/)
    for more details.

    This layer can be called with either one or two inputs. The number of inputs
    must be consistent across all calls. The options are as follows:
        `layer(decoder_sequence)`: no cross-attention will be built into the
            decoder block. This is useful when building a "decoder-only"
            transformer such as GPT-2.
        `layer(decoder_sequence, encoder_sequence)`: cross-attention will be
            built into the decoder block. This is useful when building an
            "encoder-decoder" transformer, such as the original transformer
            model described in Attention is All You Need.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in MultiHeadAttention.
        dropout: float. the dropout value, shared by
            MultiHeadAttention and feedforward network. Defaults to `0.`.
        activation: string or `keras.activations`. the
            activation function of feedforward network.
            Defaults to `"relu"`.
        layer_norm_epsilon: float. The eps value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense and multiheaded
            attention layers. Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense and multiheaded
            attention layers. Defaults to `"zeros"`.
        normalize_first: bool. If True, the inputs to the
            attention layer(s) and the intermediate dense layer are normalized
            (similar to GPT-2). If set to False, outputs of attention layer and
            intermediate dense layer are normalized (similar to BERT).
            Defaults to `False`.
        name: string. The name of the layer. Defaults to `None`.
        **kwargs: other keyword arguments.

    Examples:
    ```python
    # Create a single transformer decoder layer.
    decoder = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=64, num_heads=8)

    # Create a simple model containing the decoder.
    decoder_input = keras.Input(shape=(10, 64))
    encoder_input = keras.Input(shape=(10, 64))
    output = decoder(decoder_input, encoder_input)
    model = keras.Model(
        inputs=(decoder_input, encoder_input),
        outputs=output,
    )

    # Call decoder on the inputs.
    decoder_input_data = np.random.uniform(size=(2, 10, 64))
    encoder_input_data = np.random.uniform(size=(2, 10, 64))
    decoder_output = model((decoder_input_data, encoder_input_data))
    ```

    References:
     - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

    """

    def __init__(
            self,
            intermediate_dim,
            num_heads,
            dropout=0,
            activation="relu",
            layer_norm_epsilon=1e-05,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name=None,
            **kwargs,
    ):
        # Work around for model saving, we need to ensure our model is built
        # immediately after restoring from config.
        decoder_sequence_shape = kwargs.pop("decoder_sequence_shape", None)
        encoder_sequence_shape = kwargs.pop("encoder_sequence_shape", None)

        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = tf.keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.supports_masking = True
        self._decoder_sequence_shape = None
        self._encoder_sequence_shape = None
        if decoder_sequence_shape:
            self.build(decoder_sequence_shape, encoder_sequence_shape)

    def build(self, decoder_sequence_shape, encoder_sequence_shape=None, ):
        self._decoder_sequence_shape = decoder_sequence_shape
        self._encoder_sequence_shape = encoder_sequence_shape
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = decoder_sequence_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        head_dim = int(hidden_dim // self.num_heads)
        if head_dim == 0:
            raise ValueError(
                "Attention `head_dim` computed cannot be zero. "
                f"The `hidden_dim` value of {hidden_dim} has to be equal to "
                f"or greater than `num_heads` value of {self.num_heads}.")
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon, name="ln_1", )
        # Self attention layers.
        self.attn = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            name="attn")
        self._self_attention_dropout = tf.keras.layers.Dropout(rate=self.dropout, )

        # Cross attention layers are optional.
        self.crossattention = None
        if encoder_sequence_shape:
            self.crossattention = CachedMultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=head_dim,
                value_dim=head_dim,
                dropout=self.dropout,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                name="crossattention")
            self.ln_cross_attn = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon,
                                                                    name="ln_cross_attn")
            self._cross_attention_dropout = tf.keras.layers.Dropout(
                rate=self.dropout, )
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon, name="ln_2")
        # Feedforward layers.
        self.mlp = MLP(self.intermediate_dim, hidden_dim, resid_pdrop=self.dropout, activation=self.activation,
                       name="mlp")
        # Create layers based on input shape.
        self.built = True

    def __call__(
            self,
            decoder_sequence,
            encoder_sequence=None,
            **kwargs,
    ):
        if not self.built:
            decoder_sequence_shape = decoder_sequence.shape
            encoder_sequence_shape = None
            if encoder_sequence is not None:
                encoder_sequence_shape = encoder_sequence.shape
            self.build(decoder_sequence_shape, encoder_sequence_shape)
        return super().__call__(
            decoder_sequence, encoder_sequence=encoder_sequence, **kwargs
        )

    def call(
            self,
            decoder_sequence,
            encoder_sequence=None,
            decoder_padding_mask=None,
            decoder_attention_mask=None,
            encoder_padding_mask=None,
            encoder_attention_mask=None,
            self_attention_cache=None,
            self_attention_cache_update_index=None,
            self_attention_cache_update_mask=None,
            cross_attention_cache=None,
            cross_attention_cache_update_index=None,
            cross_attention_cache_update_mask=None,
            use_causal_mask=True,
    ):
        """Forward pass of the TransformerDecoder.

        Args:
            decoder_sequence: a Tensor. The decoder input sequence.
            encoder_sequence: a Tensor. The encoder input sequence. For decoder
                only models (like GPT2), this should be left `None`. Once the
                model is called once without an encoder_sequence, you cannot
                call it again with encoder_sequence.
            decoder_padding_mask: a boolean Tensor, the padding mask of decoder
                sequence, must be of shape
                `[batch_size, decoder_sequence_length]`.
            decoder_attention_mask: a boolean Tensor. Customized decoder
                sequence mask, must be of shape
                `[batch_size, decoder_sequence_length, decoder_sequence_length]`.
            encoder_padding_mask: a boolean Tensor, the padding mask of encoder
                sequence, must be of shape
                `[batch_size, encoder_sequence_length]`.
            encoder_attention_mask: a boolean Tensor. Customized encoder
                sequence mask, must be of shape
                `[batch_size, encoder_sequence_length, encoder_sequence_length]`.
            self_attention_cache: a dense float Tensor. The cache of key/values
                pairs in the self-attention layer. Has shape
                `[batch_size, 2, max_seq_len, num_heads, key_dims]`.
            self_attention_cache_update_index: an int or int Tensor, the index
                at which to update the `self_attention_cache`. Usually, this is
                the index of the current token being processed during decoding.
            cross_attention_cache: a dense float Tensor. The cache of
                key/value pairs in the cross-attention layer. Has shape
                `[batch_size, 2, S, num_heads, key_dims]`.
            cross_attention_cache_update_index:  an int or int Tensor, the index
                at which to update the `cross_attention_cache`. Usually, this is
                either `0` (compute the entire `cross_attention_cache`), or
                `None` (reuse a previously computed `cross_attention_cache`).
            use_causal_mask: bool, defaults to `True`. If true, a causal mask
                (masking out future input) is applied `on the decoder sequence.
        Returns:
            One of three things, depending on call arguments:
            - `outputs`, if `self_attention_cache` is `None.
            - `(outputs, self_attention_cache)`, if `self_attention_cache` is
              set and the layer has no cross-attention.
            - `(outputs, self_attention_cache, cross_attention_cache)`, if
              `self_attention_cache` and `cross_attention_cache` are set and
              the layer has cross-attention.
        """

        has_encoder_sequence = encoder_sequence is not None

        has_cross_attention = self.crossattention is not None
        if not has_cross_attention and has_encoder_sequence:
            raise ValueError(
                "The number of call arguments to "
                "`keras_nlp.layers.TransformerDecoder` should not change. "
                "Use `layer(decoder_sequence, encoder_sequence)` to "
                "build a layer with cross attention, or "
                "`layer(decoder_sequence)` to build a layer without. "
                "This layer has been built without cross attention, but "
                "you are trying to call it with encoder_sequence."
            )
        elif has_cross_attention and not has_encoder_sequence:
            raise ValueError(
                "The number of call arguments to "
                "`keras_nlp.layers.TransformerDecoder` should not change. "
                "Use `layer(decoder_sequence, encoder_sequence)` to "
                "build a layer with cross attention, or "
                "`layer(decoder_sequence)` to build a layer without. "
                "This layer has been built with cross attention, but "
                "you did not provide encoder_sequence."
            )

        has_self_attention_cache = self_attention_cache is not None
        has_cross_attention_cache = cross_attention_cache is not None
        if has_cross_attention and (
                has_self_attention_cache != has_cross_attention_cache
        ):
            raise ValueError(
                "When calling `keras_nlp.layers.TransformerDecoder` with "
                "cross-attention (with both `encoder_sequence` and "
                "`decoder_sequence`), `self_attention_cache` and "
                "`cross_attention_cache` should both be set or both be `None`. "
                "One cannot be `None` while the other is not. Received: "
                f"self_attention_cache={self_attention_cache}, "
                f"cross_attention_cache={cross_attention_cache}."
            )

        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=decoder_sequence,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            use_causal_mask=use_causal_mask,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=self_attention_cache_update_index,
        )

        x = decoder_sequence  # Intermediate result.

        # Self attention block.
        residual = x
        x = self.ln_1(x)
        x, self_attention_cache = self.attn(
            query=x,
            value=x,
            attention_mask=self_attention_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
            cache_update_mask=self_attention_cache_update_mask,
        )
        x = self._self_attention_dropout(x)
        x = x + residual
        # Cross attention is optional.
        if has_cross_attention:
            # Compute cross attention mask.
            cross_attention_mask = merge_padding_and_attention_mask(
                encoder_sequence, encoder_padding_mask, encoder_attention_mask
            )

            # Cross attention block.
            residual = x
            x = self.ln_cross_attn(x)
            x, cross_attention_cache = self.crossattention(
                query=x,
                value=encoder_sequence,
                attention_mask=cross_attention_mask,
                cache=cross_attention_cache,
                cache_update_index=cross_attention_cache_update_index,
                cache_update_mask=cross_attention_cache_update_mask
            )
            x = self._cross_attention_dropout(x)
            x = x + residual

        # Feedforward block.
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + residual
        if self_attention_cache is not None:
            if has_cross_attention:
                return x, self_attention_cache, cross_attention_cache
            else:
                return x, self_attention_cache
        else:
            return x

    def _compute_self_attention_mask(
            self,
            decoder_sequence,
            decoder_padding_mask,
            decoder_attention_mask,
            use_causal_mask,
            self_attention_cache,
            self_attention_cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        if use_causal_mask:
            batch_size = tf.shape(decoder_sequence)[0]
            input_length = output_length = tf.shape(decoder_sequence)[1]
            # We need to handle a rectangular causal mask when doing cached
            # decoding. For generative inference, `decoder_sequence` will
            # generally be length 1, and `cache` will be the full generation length.
            if self_attention_cache is not None:
                input_length = tf.shape(self_attention_cache)[2]

            causal_mask = compute_causal_mask(
                batch_size,
                input_length,
                output_length,
                0
                if self_attention_cache_update_index is None
                else self_attention_cache_update_index,
            )
            return (
                tf.minimum(decoder_mask, causal_mask)
                if decoder_mask is not None
                else causal_mask
            )
        return decoder_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": tf.keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": tf.keras.initializers.serialize(
                    self.bias_initializer
                ),
                "normalize_first": self.normalize_first,
                "decoder_sequence_shape": self._decoder_sequence_shape,
                "encoder_sequence_shape": self._encoder_sequence_shape,
            }
        )
        return config

    def compute_output_shape(self, decoder_sequence_shape):
        return decoder_sequence_shape


class MLP(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, hidden_dim, resid_pdrop=0.1, activation=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        self.c_fc = tf.keras.layers.Dense(intermediate_dim, kernel_initializer=_gpt_2_kernel_initializer(stddev=0.02),
                                          bias_initializer=_gpt_2_kernel_initializer(stddev=0.02), name="c_fc")
        self.c_proj = tf.keras.layers.Dense(hidden_dim, kernel_initializer=_gpt_2_kernel_initializer(stddev=0.02),
                                            bias_initializer=_gpt_2_kernel_initializer(stddev=0.02), name="c_proj")
        self.dropout = tf.keras.layers.Dropout(resid_pdrop)
        self.act = activation

    def call(self, x, training=False):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2


class ReverseEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding, **kwargs):
        super().__init__(**kwargs)
        self.embedding = embedding

    def call(self, inputs):
        kernel = tf.transpose(tf.convert_to_tensor(self.embedding.embeddings))
        return tf.matmul(inputs, kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + (self.embedding.embeddings.shape[0],)


class GPT2(tf.keras.Model):
    def load_weights_from_ckpt(self, ckpt_path, save=False):
        filename = os.path.basename(ckpt_path)
        print("loading:[{}]".format(filename))
        state_dict = {}
        if ckpt_path.endswith(".safetensors"):
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        new_state_dict = rebuild_state_dict(state_dict)
        print("loaded :[{}]".format(filename))
        weights = self.weights
        for module_name in ['gpt2']:
            module_weights = []
            for i, (key, perm) in enumerate(TORCH_CKPT_MAPPING[module_name]):
                if isinstance(perm, str) and perm.endswith(".npy"):
                    w = np.load(perm)
                else:
                    w = new_state_dict[key]
                    if isinstance(w, torch.Tensor):
                        w = w.detach().numpy()
                    if perm is not None:
                        w = np.transpose(w, perm)
                if weights[i].shape != w.shape:
                    print("Wrong :[{},{}]".format(weights[i].name, key))
                module_weights.append(w)
            self.set_weights(module_weights)
            print("Loaded %d weights for %s" % (len(module_weights), module_name))
        if save:
            self.save("gpt2.h5")
            print("save safetensors:{} to h5 done.".format(filename))

    def __init__(
            self,
            vocabulary_size=50257,
            num_layers=6,
            num_heads=12,
            hidden_dim=768,
            intermediate_dim=3072,
            dropout=0.1,
            max_sequence_length=1024,
            **kwargs):
        # Inputs
        head_dim = hidden_dim // num_heads
        token_ids = tf.keras.Input(shape=(None,), dtype="int32", name="token_ids")
        # padding_mask = tf.keras.Input(shape=(None,), dtype="int32", name="padding_mask")
        cache = tf.keras.Input(shape=(num_layers, 2, None, num_heads, head_dim,), dtype="float32", name="cache")
        cache_update_mask = tf.keras.Input(shape=(None, num_heads, head_dim,), dtype="float32",
                                           name="cache_update_mask")
        cache_update_index = tf.keras.Input(shape=(), batch_size=1, dtype="int32", name="start_index")
        # Embed tokens, positions.
        transformer_wte = tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=_gpt_2_kernel_initializer(stddev=0.01),
            name="transformer.wte",
        )
        token_embedding = transformer_wte(token_ids)
        # Can't use `TokenAndPositionEmbedding` layer here because of different
        # initializers.
        # cache_update_index = 0
        position_embedding = PositionEmbedding(
            sequence_length=max_sequence_length,
            initializer=_gpt_2_kernel_initializer(stddev=0.02),
            name="transformer.wpe",
        )(token_embedding, start_index=cache_update_index[0])

        # Sum and apply dropout to embeddings.
        x = tf.keras.layers.Add(name="embeddings_add")(
            (token_embedding, position_embedding)
        )
        x = tf.keras.layers.Dropout(
            dropout,
            name="embeddings_dropout")(x)
        # Apply successive transformer decoder blocks.
        caches = []
        current_caches = tf.unstack(cache, num=num_layers, axis=1)
        for i in range(num_layers):
            current_cache = current_caches[i]
            x, next_cache = TransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                layer_norm_epsilon=1e-05,
                activation=lambda x: tf.keras.activations.gelu(x, approximate=True),
                kernel_initializer=_gpt_2_kernel_initializer(stddev=0.02),
                name=f"transformer.h.{i}",
            )(x,
              # decoder_padding_mask=padding_mask,
              self_attention_cache=current_cache,
              self_attention_cache_update_mask=cache_update_mask,
              self_attention_cache_update_index=cache_update_index, )
            caches.append(next_cache)

        outcache = tf.stack(caches, axis=1)
        hidden_states = tf.keras.layers.LayerNormalization(
            name="transformer.ln_f",
            epsilon=1e-05,
            dtype="float32",
        )(x)
        logits = ReverseEmbedding(transformer_wte, name="reverse_embedding", )(hidden_states)
        outputs = (logits, hidden_states, outcache,)
        super().__init__(
            # inputs=(token_ids, padding_mask, cache, cache_update_mask, cache_update_index),
            inputs=(token_ids, cache, cache_update_mask, cache_update_index),
            outputs=outputs,
            **kwargs,
        )
        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config

    @property
    def reverse_embedding(self):
        return self.get_layer("reverse_embedding")

    @property
    def token_embedding(self):
        return self.get_layer("transformer.wte")

    @property
    def position_embedding(self):
        return self.get_layer("transformer.wpe")


def main():
    model = GPT2(vocabulary_size=50257,
                 num_layers=6,
                 num_heads=12,
                 hidden_dim=768,
                 intermediate_dim=3072,
                 max_sequence_length=1024)
    pytorch_model = r"pytorch_model.bin"
    model.load_weights_from_ckpt(pytorch_model)


if __name__ == "__main__":
    main()
