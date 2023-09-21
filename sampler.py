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
import threading

import numpy as np

GLOBAL_STATE_TRACKER = threading.local()
GLOBAL_SETTINGS_TRACKER = threading.local()


class SeedGenerator:
    """Generates variable seeds upon each call to a RNG-using function.

    In Keras, all RNG-using methods (such as `keras_core.random.normal()`)
    are stateless, meaning that if you pass an integer seed to them
    (such as `seed=42`), they will return the same values at each call.
    In order to get different values at each call, you must use a
    `SeedGenerator` instead as the seed argument. The `SeedGenerator`
    object is stateful.

    Example:

    ```python
    seed_gen = keras_core.random.SeedGenerator(seed=42)
    values = keras_core.random.normal(shape=(2, 3), seed=seed_gen)
    new_values = keras_core.random.normal(shape=(2, 3), seed=seed_gen)
    ```

    Usage in a layer:

    ```python
    class Dropout(keras_core.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras_core.random.SeedGenerator(1337)

        def call(self, x, training=False):
            if training:
                return keras_core.random.dropout(
                    x, rate=0.5, seed=self.seed_generator
                )
            return x
    ```
    """

    def __init__(self, seed=None, **kwargs):
        if kwargs:
            raise ValueError(f"Unrecognized keyword arguments: {kwargs}")

        self._initial_seed = seed
        if seed is None:
            def make_default_seed():
                import random as python_random
                return python_random.randint(1, int(1e9))

            seed = make_default_seed()

        if not isinstance(seed, int):
            raise ValueError(
                "Argument `seed` must be an integer. " f"Received: seed={seed}"
            )

        self.state = np.asarray([seed, 0], dtype="uint32")

    def next(self, ordered=True):
        seed_state = self.state
        # Use * 1 to create a copy
        new_seed_value = seed_state * 1
        if ordered:
            increment = np.array([0, 1], dtype="uint32")
            self.state = (seed_state + increment)
        else:
            # This produces a sequence of near-unique numbers
            # between 0 and 1M
            self.state = ((seed_state + 1) * 5387 % 933199)
        return new_seed_value


def set_global_attribute(name, value):
    setattr(GLOBAL_STATE_TRACKER, name, value)


def get_global_attribute(name, default=None, set_to_default=False):
    attr = getattr(GLOBAL_STATE_TRACKER, name, None)
    if attr is None and default is not None:
        attr = default
        if set_to_default:
            set_global_attribute(name, attr)
    return attr


def global_seed_generator():
    gen = get_global_attribute("global_seed_generator")
    if gen is None:
        gen = SeedGenerator()
        set_global_attribute("global_seed_generator", gen)
    return gen


def draw_seed(seed):
    if isinstance(seed, SeedGenerator):
        return seed.next()
    elif isinstance(seed, int):
        return np.asarray([seed, 0], dtype="uint32")
    elif seed is None:
        return global_seed_generator().next(ordered=False)
    raise ValueError(
        "Argument `seed` must be either an integer "
        "or an instance of `SeedGenerator`. "
        f"Received: seed={seed} (of type {type(seed)})"
    )


def softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def categorical(logits, num_samples, dtype="int64", seed=None):
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    output = []
    for logits_instance in logits:
        probabilities = softmax(logits_instance)
        classes = np.arange(logits_instance.shape[-1])
        samples = rng.choice(classes, size=num_samples, p=probabilities)
        output.append(samples)
    return np.array(output).astype(dtype)


def top_k(x, k, sorted=False):
    sorted_indices = np.argsort(x, axis=-1)[..., ::-1]
    sorted_values = np.sort(x, axis=-1)[..., ::-1]

    if sorted:
        # Take the k largest values.
        top_k_values = sorted_values[..., :k]
        top_k_indices = sorted_indices[..., :k]
    else:
        # Partition the array such that all values larger than the k-th
        # largest value are to the right of it.
        top_k_values = np.partition(x, -k, axis=-1)[..., -k:]
        top_k_indices = np.argpartition(x, -k, axis=-1)[..., -k:]

        # Get the indices in sorted order.
        idx = np.argsort(-top_k_values, axis=-1)

        # Get the top k values and their indices.
        top_k_values = np.take_along_axis(top_k_values, idx, axis=-1)
        top_k_indices = np.take_along_axis(top_k_indices, idx, axis=-1)

    return top_k_values, top_k_indices


class Sampler:
    """sampler class.

       This sampler is implemented on greedy search, i.e., always picking up the
       token of the largest probability as the next token.

       Call arguments:
           {{call_args}}

       Examples:
       ```python
       # Use a simple alphabet of lowercase characters with ids in range [0, 25].
       int_lookup = {i: chr(i + ord('a')) for i in range(26)}
       char_lookup = {v: k for k, v in int_lookup.items()}
       batch_size, length, vocab_size = 1, 12, len(int_lookup)

       def next(prompt, cache, index):
           hidden_states = np.ones((batch_size, 10))
           # A uniform distribution over our alphabet.
           logits = np.ones((batch_size, vocab_size))
           return logits, hidden_states, cache

       output = GreedySampler()(
           next=next,
           prompt=np.full((batch_size, length,), char_lookup['z'], dtype="int32"),
           index=5,
       )
       print(["".join([int_lookup[i] for i in s]) for s in output.numpy()])
       # >>> ['zzzzzaaaaaaa']
       ```
       """

    def __init__(self, temperature=1.0, ):
        self.temperature = temperature
        self._seed_generators = []

    def __setattr__(self, name, value):
        # We could update to the `Tracker` class from keras-core if our needs
        # become more advanced (e.g. list assignment, nested trackables). For
        # now, we only track `SeedGenerator` instances directly on the sampler.
        if isinstance(value, SeedGenerator):
            self._seed_generators.append(value)
        return super().__setattr__(name, value)

    @property
    def variables(self):
        variables = []
        for sg in self._seed_generators:
            variables.append(sg.state)
        return variables

    def __call__(self, next, prompt, cache=None, index=0, mask=None, end_token_id=None, hidden_states=None, ):
        max_length = prompt.shape[-1]
        if mask is None:
            mask = np.zeros_like(prompt, dtype="bool")
        # `ops.while_loop` will not accept `None` as a value for `loop_vars`.
        cache = () if cache is None else cache

        def cond(prompt, cache, index):
            if end_token_id is None:
                return True
            # Stop if all sequences have produced a *new* end_token_id.
            end_tokens = (prompt == end_token_id) & (~mask)
            prompt_done = np.any(end_tokens, axis=-1)
            return np.logical_not(np.all(prompt_done))

        def slice_update(inputs, updates, start_indices):
            # Generate list of indices arrays for each dimension
            indices = [
                np.arange(start, start + length)
                for start, length in zip(start_indices, updates.shape)
            ]
            # Use np.ix_ to create a multidimensional index array
            mesh = np.ix_(*indices)
            inputs[mesh] = updates
            return inputs

        def body(prompt, cache, index):
            # Compute the softmax distribution for the next token.
            logits, _, cache = next(prompt, cache, index)
            probabilities = softmax(logits / self.temperature)
            # Compute the next token.
            next_token = self.get_next_token(probabilities)
            next_token = np.where(mask[:, index], prompt[:, index], next_token)
            # Update the prompt with the next token.
            next_token = next_token[:, None]
            prompt = slice_update(prompt, next_token, [0, index])
            # Return the next prompt, cache and incremented index.
            return prompt, cache, index + 1

        prompt, _, _ = self.run_loop(
            cond,
            body,
            loop_vars=(prompt, cache, index),
            maximum_iterations=(max_length - index),
        )
        return prompt

    def run_loop(self, cond, body, loop_vars=None, maximum_iterations=None):
        """Run ops.while_loops with a `StatelessScope` if necessary."""
        iteration = 0
        while cond(*loop_vars) and (maximum_iterations is None or iteration < maximum_iterations):
            loop_vars = body(*loop_vars)
            iteration += 1
        return loop_vars

    def get_next_token(self, probabilities):
        """Get the next token.
        Args:
            probabilities: a Tensor, the probability distribution for next
                token over all vocab tokens.
        Get the next token based on given probability distribution over tokens.
        Subclasses must implement this method.
        """
        return np.argmax(probabilities, axis=-1)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {"temperature": self.temperature}


class TopPSampler(Sampler):
    """Top-P Sampler class.

    This sampler implements top-p search algorithm. Top-p search selects tokens
    from the smallest subset of output probabilities that sum to greater than
    `p`. Put in another way, top-p will first order token predictions by
    likelihood, and ignore all tokens after the cumulative probability of
    selected tokens exceeds `p`, then select a token from the remaining tokens.

    Args:
        p: float, the `p` value of top-p.
        k: int. If set, this argument defines a
            heuristic "top-k" cutoff applied before the "top-p" sampling. All
            logits not in the top `k` will be discarded, and the remaining
            logits will be sorted to find a cutoff point for `p`. Setting this
            arg can significantly speed sampling up by reducing the number
            of tokens to sort. Defaults to `None`.
        seed: int. The random seed. Defaults to `None`.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    # Use a simple alphabet of lowercase characters with ids in range [0, 25].
    int_lookup = {i: chr(i + ord('a')) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    batch_size, length, vocab_size = 1, 12, len(int_lookup)

    def next(prompt, cache, index):
        hidden_states = np.ones((batch_size, 10))
        # A uniform distribution over our alphabet.
        logits = np.ones((batch_size, vocab_size))
        return logits, hidden_states, cache

    output = TopPSampler(p=0.1)(
        next=next,
        prompt=np.full((batch_size, length,), char_lookup['z'], dtype="int32"),
        index=5,
    )
    print(["".join([int_lookup[i] for i in s]) for s in output.numpy()])
    # >>> ['zzzzzbabcccb']
    ```
    """

    def __init__(self, p=0.1, k=None, seed=None, **kwargs, ):
        super().__init__(**kwargs)
        self.p = p
        self.k = k
        self.seed = seed
        self.seed_generator = SeedGenerator(seed)

    def get_next_token(self, probabilities):
        cutoff = probabilities.shape[1]
        if self.k is not None:
            # If `k` is set, only sample from top `k` tokens.
            cutoff = self.k
        sorted_preds, sorted_indices = top_k(
            probabilities, k=cutoff, sorted=True
        )
        # Calculate cumulative probability distribution.
        cumulative_probabilities = np.cumsum(sorted_preds, axis=-1)
        # Create a mask for the tokens to keep.
        keep_mask = cumulative_probabilities <= self.p
        # Shift to include the last token that exceed p.
        shifted_keep_mask = np.concatenate(
            [np.ones_like(keep_mask[:, :1]), keep_mask[:, :-1]], axis=-1)
        # Filter out unmasked tokens and sample from filtered distribution.
        probabilities = np.where(
            shifted_keep_mask,
            sorted_preds,
            np.zeros(sorted_preds.shape, dtype=sorted_preds.dtype))
        sorted_next_token = categorical(
            np.log1p(probabilities),
            1,
            seed=self.seed_generator,
            dtype="int32")
        output = np.take_along_axis(sorted_indices, sorted_next_token, axis=-1)
        return np.squeeze(output, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "p": self.p,
                "k": self.k,
                "seed": self.seed,
            }
        )
        return config


class TopKSampler(Sampler):
    """Top-K Sampler class.

    This sampler implements top-k search algorithm. Briefly, top-k algorithm
    randomly selects a token from the tokens of top K probability, with
    selection chance determined by the probability.

    Args:
        k: int, the `k` value of top-k.
        seed: int. The random seed. Defaults to `None`.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    # Use a simple alphabet of lowercase characters with ids in range [0, 25].
    int_lookup = {i: chr(i + ord('a')) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    batch_size, length, vocab_size = 1, 12, len(int_lookup)

    def next(prompt, cache, index):
        hidden_states = np.ones((batch_size, 10))
        # A uniform distribution over our alphabet.
        logits = np.ones((batch_size, vocab_size))
        return logits, hidden_states, cache

    output = TopKSampler(k=3)(
        next=next,
        prompt=np.full((batch_size, length,), char_lookup['z'], dtypes="int32"),
        index=5,
    )
    print(["".join([int_lookup[i] for i in s]) for s in output.numpy()])
    # >>> ['zzzzzacbbcaa']
    ```
    """

    def __init__(self, k=5, seed=None, **kwargs, ):
        super().__init__(**kwargs)
        self.k = k
        self.seed = seed
        self.seed_generator = SeedGenerator(seed)

    def get_next_token(self, probabilities):
        # Filter out top-k tokens.
        top_k_pred, top_k_indices = top_k(
            probabilities,
            k=self.k,
            sorted=False)
        # Sample the next token from the probability distribution.
        sample_indices = categorical(
            # tf does not support half precision multinomial sampling, so make
            # sure we have full precision here.
            np.asarray(np.log1p(top_k_pred), "float32"),
            1,
            seed=self.seed_generator,
            dtype="int32")
        # Rearrange to get the next token idx from the original order.
        output = np.take_along_axis(top_k_indices, sample_indices, axis=-1)
        return np.squeeze(output, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "seed": self.seed,
            }
        )
        return config
