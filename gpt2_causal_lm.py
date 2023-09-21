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

import numpy as np

from sampler import TopKSampler, TopPSampler


class GPT2CausalLM(object):
    """An end-to-end GPT2 model for causal langauge modeling.

       A causal language model (LM) predicts the next token based on previous
       tokens.

       This model has a `generate()` method, which generates text based on a
       prompt.  By  default, `"top_k"` sampling will be used.

       Disclaimer: Pre-trained models are provided on an "as is" basis, without
       warranties or conditions of any kind. The underlying model is provided by a
       third party and subject to a separate license, available
       [here](https://github.com/openai/gpt-2).

       Args:
           backbone: A `GPT2Backbone` instance.
           preprocessor: A `GPT2CausalLMPreprocessor` or `None`.
               If `None`, this model will not apply preprocessing, and inputs
               should be preprocessed before calling the model.
       """

    def __init__(self, backbone, preprocessor=None, sampler="top_k", k=5, p=0.1, seed=None, compute_dtype="float32"):
        self.compute_dtype = compute_dtype
        self.backbone = backbone
        self.preprocessor = preprocessor
        self._sampler = TopKSampler(k=k, seed=seed) if sampler == "top_k" else TopPSampler(p=p, k=k, seed=seed)

    @staticmethod
    def slice_update(inputs, updates, start_indices):
        # Generate list of indices arrays for each dimension
        indices = [
            np.arange(start, start + length)
            for start, length in zip(start_indices, updates.shape)]
        # Use np.ix_ to create a multidimensional index array
        mesh = np.ix_(*indices)
        inputs[mesh] = updates
        return inputs

    def call_with_cache(self, token_ids, cache, cache_update_index):
        """Forward pass of `GPT2CausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, max_length)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of current inputs in the
                whole sequence.

        Returns:
            A (logits, hidden_states, cache) tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the decoding cache.
        """
        batch_size = token_ids.shape[0]
        max_length = token_ids.shape[1]
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
        cache_update_mask = np.zeros([batch_size, cache.shape[3], num_heads, head_dim], dtype=self.compute_dtype)
        cache_update_updates = np.ones([batch_size, max_length, num_heads, head_dim], dtype=self.compute_dtype)
        start = [0, cache_update_index, 0, 0]
        cache_update_mask = self.slice_update(cache_update_mask, cache_update_updates, start)
        logits, hidden_states, cache = self.backbone.predict_on_batch(
            (token_ids, cache, cache_update_mask, np.ones([batch_size], dtype=np.int32) * cache_update_index))
        return logits, hidden_states, cache

    def _build_cache(self, token_ids, padding_mask):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = token_ids.shape[0]
        max_length = token_ids.shape[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        cache = np.zeros(shape, dtype=self.compute_dtype)
        cache_update_mask = np.ones([batch_size, max_length, num_heads, head_dim], dtype=self.compute_dtype)
        # Seed the cache.
        _, hidden_states, cache = self.backbone.predict_on_batch(
            (token_ids, cache, cache_update_mask, np.zeros([batch_size], dtype=np.int32)))
        return hidden_states, cache

    def generate_step(self, inputs, end_token_id=None):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"token_ids"` and `"padding_mask"`.

        Args:
            inputs: A dictionary with two keys `"token_ids"` and
                `"padding_mask"` and batched tensor values.
            end_token_id: The id of the end token to stop on. If all
                sequences have produced a new `end_token_id`, generation
                will stop.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(token_ids, padding_mask)
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = np.sum(padding_mask.astype("int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = np.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = index - 1
            prompt = prompt[:, cache_update_index:cache_update_index + 1]
            logits, hidden_states, cache = self.call_with_cache(
                prompt,
                cache,
                cache_update_index)
            return (np.squeeze(logits, axis=1),
                    np.squeeze(hidden_states, axis=1),
                    cache)

        token_ids = self._sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            end_token_id=end_token_id,
            hidden_states=hidden_states)

        # Compute an output padding mask with the token ids we updated.
        if end_token_id is not None:
            # Build a mask of `end_token_id` locations not in the original
            # prompt (not in locations where `padding_mask` is True).
            end_locations = np.logical_and(
                np.equal(token_ids, end_token_id),
                np.logical_not(padding_mask))
            end_locations = end_locations.astype("int32")
            # Use cumsum to get ones in all locations after end_locations.
            cumsum = np.cumsum(end_locations, axis=-1).astype("int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            padding_mask = np.logical_not(overflow.astype("bool"))
        else:
            # Without early stopping, all locations will have been updated.
            padding_mask = np.ones_like(token_ids, dtype="bool")
        return {"token_ids": token_ids, "padding_mask": padding_mask}

    def generate(self, inputs, max_length=None):
        """Generate text given prompt `inputs`.

        This method generates text based on given `inputs`. The sampling method
        used for generation can be set via the `compile()` method.

        If a `preprocessor` is attached to the model, `inputs` will be
        preprocessed inside the `generate()` function and should match the
        structure expected by the `preprocessor` layer (usually raw strings).
        If a `preprocessor` is not attached, inputs should match the structure
        expected by the `backbone`. See the example usage above for a
        demonstration of each.

        Args:
            inputs: python data, tensor data. If a
                `preprocessor` is attached to the model, `inputs` should match
                the structure expected by the `preprocessor` layer. If a
                `preprocessor` is not attached, `inputs` should match the
                structure expected the the `backbone` model.
            max_length: Optional. int. The max length of the generated sequence.
                Will default to the max configured `sequence_length` of the
                `preprocessor`. If `preprocessor` is `None`, `inputs` should be
                should be padded to the desired maximum length and this argument
                will be ignored.
        """
        # Setup our three main passes.
        # 1. Optionally preprocessing strings to dense integer tensors.
        # 2. Generate new tokens via a compiled function on dense tensors.
        # 3. Optionally postprocess dense integer tensors back to string.
        if self.preprocessor is not None:
            end_token_id = self.preprocessor.tokenizer.end_token_id

        def preprocess(x):
            return self.preprocessor.generate_preprocess(
                x, sequence_length=max_length)

        def generate(x):
            return self.generate_step(x, end_token_id=end_token_id)

        def postprocess(x):
            return self.preprocessor.generate_postprocess(x)

        if self.preprocessor is not None:
            # Fast path for non-dataset, single-batch input.
            inputs = [preprocess(x) for x in inputs]

        outputs = [generate(x) for x in inputs]

        if self.preprocessor is not None:
            outputs = [postprocess(x) for x in outputs]

        return outputs


class StartEndPacker(object):
    """Adds start and end tokens to a sequence and pads to a fixed length.

    This layer is useful when tokenizing inputs for tasks like translation,
    where each sequence should include a start and end marker. It should
    be called after tokenization. The layer will first trim inputs to fit, then
    add start/end tokens, and finally pad, if necessary, to `sequence_length`.

    Args:
        sequence_length: int. The desired output length.
        start_value: int/str/list/tuple. The ID(s) or token(s) that are to be
            placed at the start of each sequence. The dtype must match the dtype
            of the input tensors to the layer. If `None`, no start value will be
            added.
        end_value: int/str/list/tuple. The ID(s) or token(s) that are to be
            placed at the end of each input segment. The dtype must match the
            dtype of the input tensors to the layer. If `None`, no end value
            will be added.
        pad_value: int/str. The ID or token that is to be placed into the
            unused positions after the last segment in the sequence. If `None`,
            0 or "" will be added depending on the dtype of the input tensor.
        return_padding_mask: bool. Whether to return a boolean padding mask of
            all locations that are filled in with the `pad_value`.

    Call arguments:
        inputs:  A list of python strings.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.
        add_start_value: Pass `False` to not append a start value for this
            input.
        add_end_value: Pass `False` to not append an end value for this
            input.
    """

    def __init__(self, sequence_length, start_value=None, end_value=None, pad_value=None, return_padding_mask=False):
        self.sequence_length = sequence_length
        self._start_value = start_value
        self._end_value = end_value
        self.start_value = start_value
        self.end_value = end_value
        self.pad_value = pad_value
        self.return_padding_mask = return_padding_mask

    def __call__(self, inputs, sequence_length=None, add_start_value=True, add_end_value=True):
        x = list(inputs)  # Intermediate result.
        sequence_length = sequence_length or self.sequence_length
        # Concatenate start and end tokens.
        if add_start_value and self.start_value is not None:
            start_value = self.start_value
            x.insert(0, start_value)
        if add_end_value and self.end_value is not None:
            end_value = self.end_value
            # Trim to leave room for end token.
            x = x[: sequence_length - len(end_value)]
            x.append(end_value)
        x_len = len(x)
        pad_value = self.pad_value
        if pad_value is None:
            pad_value = 0
        if x_len != sequence_length:
            # Pad to desired length.
            pad_len = (sequence_length - x_len)
            outputs = x + [pad_value] * pad_len
            mask = [True] * x_len + [False] * pad_len
        else:
            outputs = x
            mask = [True] * x_len
        if self.return_padding_mask:
            return outputs, mask
        return outputs

    def get_config(self):
        config = {
            "sequence_length": self.sequence_length,
            "start_value": self._start_value,
            "end_value": self._end_value,
            "pad_value": self.pad_value,
            "return_padding_mask": self.return_padding_mask}
        return config

    def compute_output_shape(self, inputs_shape):
        inputs_shape = list(inputs_shape)
        inputs_shape[-1] = self.sequence_length
        return tuple(inputs_shape)


class GPT2CausalLMPreprocessor(object):
    """GPT2 Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_nlp.models.GPT2CausalLM`. By default, it will take in batches of
    strings, and return outputs in a `(x, y, sample_weight)` format, where the
    `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_nlp.models.GPT2CausalLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A GPT2Tokenizer instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence.

    Call arguments:
        x: A string, list of python strings.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    Examples:
    ```python
    # Tokenize and pack a single sentence.
    sentence = ("League of legends")
    preprocessor(sentence)
    # Same output.
    preprocessor("League of legends")

    # Tokenize a batch of sentences.
    sentences = (["Taco tuesday", "Fish taco please!"])
    preprocessor(sentences)
    # Same output.
    preprocessor(["Taco tuesday", "Fish taco please!"])
    ```
    """

    def __init__(self, tokenizer, sequence_length=1024, add_start_token=True, add_end_token=True):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token
        self.packer = StartEndPacker(
            start_value=tokenizer.start_token_id,
            end_value=tokenizer.end_token_id,
            pad_value=tokenizer.pad_token_id,
            sequence_length=sequence_length,
            return_padding_mask=True)

    def get_config(self):
        config = {
            "sequence_length": self.sequence_length,
            "add_start_token": self.add_start_token,
            "add_end_token": self.add_end_token}
        return config

    def __call__(self, x, y=None, sample_weight=None, sequence_length=None):
        if y is not None or sample_weight is not None:
            logging.warning(
                "`GPT2CausalLMPreprocessor` generates `y` and `sample_weight` "
                "based on your input data, but your data already contains `y` "
                "or `sample_weight`. Your `y` and `sample_weight` will be "
                "ignored.")
        sequence_length = sequence_length or self.sequence_length
        x = self.tokenizer(x)
        # Pad with one extra token to account for the truncation below.
        token_ids, padding_mask = self.packer(x, sequence_length=sequence_length + 1,
                                              add_start_value=self.add_start_token,
                                              add_end_value=self.add_end_token)
        # The last token does not have a next token, so we truncate it out.
        # Target `y` will be the next token.
        y, sample_weight = np.asarray([token_ids[:-1]], np.int32), np.asarray([padding_mask[:-1]], "bool")
        x = {"token_ids": y,
             "padding_mask": sample_weight}
        return x, y, sample_weight

    def generate_preprocess(self, x, sequence_length=None):
        """Covert strings to integer token input for generation.

        Similar to calling the layer for training, this method takes in strings
        or tensor strings, tokenizes and packs the input, and computes a padding
        mask masking all inputs not filled in with a padded value.

        Unlike calling the the layer for training, this method does not compute
        labels and will never append a `tokenizer.end_token_id` to the end of
        the sequence (as generation is expected to continue at the end of the
        inputted prompt).
        """
        token_id = self.tokenizer.encode(x)
        token_ids, padding_mask = self.packer(
            token_id, sequence_length=sequence_length, add_end_value=False)
        return {"token_ids": np.asarray([token_ids], np.int32),
                "padding_mask": np.asarray([padding_mask], "bool")}

    @staticmethod
    def boolean_mask(tensor, mask, axis=None):
        def _apply_mask_1d(reshaped_tensor, mask, axis=None):
            """Mask tensor along dimension 0 with a 1-D mask."""
            indices = np.where(mask)[0]
            return np.take(reshaped_tensor, indices, axis=axis)

        tensor = np.asarray(tensor)
        mask = np.asarray(mask)
        shape_mask = mask.shape
        ndims_mask = len(shape_mask)
        shape_tensor = tensor.shape
        if ndims_mask == 0:
            raise ValueError("mask cannot be scalar.")
        if axis is None:
            axis = 0
        axis_value = np.squeeze(axis)
        if axis_value is not None:
            axis = axis_value
            if shape_tensor[axis:axis + ndims_mask] != shape_mask:
                raise ValueError("Mask shape does not match tensor shape.")
        leading_size = np.prod(shape_tensor[axis:axis + ndims_mask], dtype=np.int32)
        tensor = np.reshape(tensor,
                            np.concatenate((shape_tensor[:axis], [leading_size], shape_tensor[axis + ndims_mask:]),
                                           axis=0).astype(np.int32))

        if axis_value is not None:
            first_dim = np.prod(shape_tensor[axis:axis + ndims_mask], dtype=np.int32)
            tensor_shape = np.concatenate((shape_tensor[:axis], [first_dim], shape_tensor[axis + ndims_mask:]),
                                          axis=0).astype(np.int32)
            tensor = np.reshape(tensor, tensor_shape)
        mask = mask.reshape(-1)
        return _apply_mask_1d(tensor, mask, axis)

    def generate_postprocess(self, x):
        """Covert integer token output to strings for generation.
        This method reverses `generate_preprocess()`, by first removing all
        padding and start/end tokens, and then converting the integer sequence
        back to a string.
        """
        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        if not isinstance(token_ids, np.ndarray):
            token_ids = np.asarray(token_ids)
        if not isinstance(padding_mask, np.ndarray):
            padding_mask = np.asarray(padding_mask)
        # Strip any special tokens during detokenization (e.g. the start and
        # end markers). In the future we could make this configurable.
        padding_mask = padding_mask & (token_ids != self.tokenizer.end_token_id)
        token_ids = self.boolean_mask(token_ids, padding_mask)
        return self.tokenizer.decode(token_ids)


def main():
    from gpt2 import GPT2
    from gpt2_tokenizer import GPT2Tokenizer
    sequence_length = 128
    max_length = 1024
    seed = 123
    tokenizer = GPT2Tokenizer()
    preprocessor = GPT2CausalLMPreprocessor(tokenizer=tokenizer, sequence_length=sequence_length)
    gpt2_backbone = GPT2(vocabulary_size=50257, num_layers=6, num_heads=12, hidden_dim=768, intermediate_dim=3072,
                         max_sequence_length=1024)
    # download form:
    # https://huggingface.co/AUTOMATIC/promptgen-lexart/blob/main/pytorch_model.bin
    pytorch_model = "pytorch_model.bin"
    gpt2_backbone.load_weights_from_ckpt(pytorch_model)
    gpt2_lm = GPT2CausalLM(
        backbone=gpt2_backbone,
        preprocessor=preprocessor, seed=seed)
    z = gpt2_lm.generate(["a cat."], max_length=max_length)
    print(z)


if __name__ == "__main__":
    main()
