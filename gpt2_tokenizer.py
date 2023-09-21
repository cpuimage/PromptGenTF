# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import bisect
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from io import open
from typing import Dict, Union, Tuple, Any, overload
from typing import List, Optional

import numpy as np
import regex as re
import unicodedata

logger = logging.getLogger(__name__)


class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self):
        self.data = {}

    def add(self, word: str):
        """
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged

        Example:

        # python
        # trie = Trie()
        # trie.add("Hello 友達")
        # trie.data
        {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}

        # trie.add("Hello")
        # trie.data
        {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        """
        if not word:
            # Prevent empty string
            return
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def split(self, text: str) -> List[str]:
        """
        Will look for the words added to the trie within `text`. Output is the original string splitted along the
        boundaries of the words found.

        This trie will match the longest possible word first !

        Example:

        python
        # trie = Trie()
        # trie.split("[CLS] This is a extra_id_100")
        ["[CLS] This is a extra_id_100"]

        # trie.add("[CLS]")
        # trie.add("extra_id_1")
        # trie.add("extra_id_100")
        # trie.split("[CLS] This is a extra_id_100")
        ["[CLS]", " This is a ", "extra_id_100"]
        """
        # indexes are counted left of the chars index.
        # "hello", index 0, is left of h, index 1 is between h and e.
        # index 5 is right of the "o".

        # States are going to capture every possible start (indexes as above)
        # as keys, and have as values, a pointer to the position in the trie
        # where we're at. This is a partial match for now.
        # This enables to keep track of multiple matches while we're iterating
        # the string
        # If the trie contains, "blowing", and "lower" and we encounter the
        # string "blower", we need to split into ["b", "lower"].
        # This is where we need to keep track of multiple possible starts.
        states = OrderedDict()

        # This will contain every indices where we need
        # to cut.
        # We force to cut at offset 0 and len(text) (added later)
        offsets = [0]

        # This is used by the lookahead which needs to skip over
        # some text where the full match exceeded the place in the initial
        # for loop
        skip = 0
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue

            # This will track every state
            # that stop matching, we need to stop tracking them.
            # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
            # fail on "b", we need to remove 0 from the valid states.
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False
            end = 0
            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    # This is a final match, we need to reset and
                    # store the results in `offsets`.

                    # Lookahead to match longest first
                    # Important in case of extra_id_1 vs extra_id_100
                    # Here we are also actively looking for other earlier partial
                    # matches
                    # "[CLS]", "L", we need to match CLS even if L is special
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            # This partial match is later, we can stop looking
                            break
                        elif lookstart < start:
                            # This partial match is earlier, the trie pointer
                            # was already updated, so index is + 1
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            # Here lookstart == start and
                            #      looktrie_pointer == trie_pointer
                            # It wasn't updated yet so indices are current ones
                            lookahead_index = current
                            end = current
                        next_char = text[lookahead_index] if lookahead_index < len(text) else None
                        if "" in looktrie_pointer:
                            start = lookstart
                            end = lookahead_index
                            skip = lookahead_index

                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                    # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current >= skip and current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    @staticmethod
    def cut_text(text, offsets):
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                logger.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it"
                    " anyway."
                )
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((33 <= cp <= 47)
            or (58 <= cp <= 64)
            or (91 <= cp <= 96)
            or (123 <= cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


def _insert_one_token_to_ordered_list(token_list: List[str], new_token: str):
    """
    Inserts one token to an ordered list if it does not already exist. Note: token_list must be sorted.
    """
    insertion_idx = bisect.bisect_left(token_list, new_token)
    # Checks if new_token is already in the ordered token_list
    if insertion_idx < len(token_list) and token_list[insertion_idx] == new_token:
        # new_token is in token_list, don't add
        return
    else:
        token_list.insert(insertion_idx, new_token)


@dataclass(frozen=True, eq=True)
class AddedToken:
    """
    AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
    way it should behave.
    """

    content: str = field(default_factory=str)
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = True

    def __getstate__(self):
        return self.__dict__

    def __str__(self):
        return self.__dict__['content']


class PreTrainedTokenizer(object):
    """
    Base class for all tokenizers.

    Handle all the shared methods for tokenization and special tokens, methods
    dowloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.
    This class also contains the added tokens in a unified way on top of all tokenizers, so we don't
    have to handle the specific vocabulary augmentation methods of the various underlying
    dictionary structures (BPE, sentencepiece...).

    Class attributes (overridden by derived classes):

        ``vocab_files_names``: a python ``dict`` with, as keys, the ``__init__`` keyword name of
        each vocabulary file required by the model, and as associated values, the filename for
        saving the associated file (string).

        ``pretrained_vocab_files_map``: a python ``dict of dict`` the high-level keys being the
        ``__init__`` keyword name of each vocabulary file required by the model, the low-level
        being the `short-cut-names` (string) of the pretrained models with, as associated values,
        the `url` (string) to the associated pretrained vocabulary file.

        ``max_model_input_sizes``: a python ``dict`` with, as keys, the `short-cut-names` (string)
        of the pretrained models, and as associated values, the maximum length of the sequence
        inputs of this model, or None if the model has no maximum input size.

        ``pretrained_init_configuration``: a python ``dict`` with, as keys, the `short-cut-names`
        (string) of the pretrained models, and as associated values, a dictionnary of specific
        arguments to pass to the ``__init__`` method of the tokenizer class for this pretrained
        model when loading the tokenizer with the ``from_pretrained()`` method.

    Args:
        bos_token (:obj:`str`, `optional`): A special token representing the beginning of a
            sentence.
        eos_token (:obj:`str`, `optional`): A special token representing the end of a sentence.
        unk_token (:obj:`str`, `optional`): A special token representing an out-of-vocabulary token.
        sep_token (:obj:`str`, `optional`): A special token separating two different sentences in
            the same input (used by BERT for instance).
        pad_token (:obj:`str`, `optional`): A special token used to make arrays of tokens the same
            size for batching purpose.
            Will then be ignored by attention mechanisms or loss computation.
        cls_token (:obj:`str`, `optional`): A special token representing the class of the input
            (used by BERT for instance).
        mask_token (:obj:`str`, `optional`): A special token representing a masked token (used by
            masked-language modeling pretraining objectives, like BERT).
        eod_token (:obj:`str`, `optional`): A special token representing the end of a document.
            additional_special_tokens (tuple or list of :obj:`str`, `optional`):
            A tuple or a list of additional special tokens.
    """
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "eod_token",
        "additional_special_tokens",
    ]

    def __init__(self, verbose=True, **kwargs):
        # Added tokens - We store this for both slow and fast tokenizers
        # until the serialization of Fast tokenizers is updated
        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.unique_no_split_tokens: List[str] = []
        self.tokens_trie = Trie()

        self._decode_use_source_tokenizer = False

        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._eod_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []
        self.verbose = verbose

        # We directly set the hidden value to allow initialization with special tokens
        # which are not yet in the vocabulary. Necessary for serialization/de-serialization
        # TODO clean this up at some point (probably by switching to fast tokenizers)
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    assert all(
                        isinstance(t, (str, AddedToken)) for t in value
                    ), "One of the tokens is not a string or an AddedToken"
                    setattr(self, key, value)
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)
                else:
                    raise TypeError(f"special token {key} has to be either str or AddedToken but got: {type(value)}")

    def sanitize_special_tokens(self) -> int:
        """
        Make sure that all the special tokens attributes of the tokenizer (`tokenizer.mask_token`,
        `tokenizer.cls_token`, etc.) are in the vocabulary.

        Add the missing ones to the vocabulary if needed.

        Return:
            `int`: The number of tokens added in the vocabulary during the operation.
        """
        return self.add_tokens(self.all_special_tokens_extended, special_tokens=True)

    def add_special_tokens(
            self, special_tokens_dict: Dict[str, Union[str, AddedToken]], replace_additional_special_tokens=True
    ) -> int:
        """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
        special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
        current vocabulary).

        Note,None When adding new tokens to the vocabulary, you should make sure to also resize the token embedding
        matrix of the model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

        Using `add_special_tokens` will ensure your special tokens can be used in several ways:

        - Special tokens are carefully handled by the tokenizer (they are never split).
        - You can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This
          makes it easy to develop model-agnostic training and fine-tuning scripts.

        When possible, special tokens are already registered for provided pretrained models (for instance
        [`BertTokenizer`] `cls_token` is already registered to be :obj*'[CLS]'* and XLM's one is also registered to be
        `'</s>'`).

        Args:
            special_tokens_dict (dictionary *str* to *str* or `tokenizers.AddedToken`):
                Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
                `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].

                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
                assign the index of the `unk_token` to them).
            replace_additional_special_tokens (`bool`, *optional*,, defaults to `True`):
                If `True`, the existing list of additional special tokens will be replaced by the one specified in
                `special_tokens_dict`. Otherwise, `self._additional_special_tokens` is updated. In the former case, the
                tokens will NOT be removed from the tokenizer's full vocabulary - they are only being flagged as
                non-special tokens.

        Returns:
            `int`: Number of tokens added to the vocabulary.

        Examples:

        ```python
        # Let's see how to add a new classification token to GPT-2
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2Model.from_pretrained("gpt2")

        special_tokens_dict = {"cls_token": "<CLS>"}

        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))

        assert tokenizer.cls_token == "<CLS>"
        ```"""
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f"Key {key} is not a special token"

            if self.verbose:
                logger.info(f"Assigning {value} to the {key} key of the tokenizer")

            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(
                    isinstance(t, (str, AddedToken)) for t in value
                ), f"Tokens {value} for key {key} should all be str or AddedToken instances"

                if replace_additional_special_tokens:
                    setattr(self, key, value)
                else:
                    # This is a copy of `self._additional_special_tokens`
                    additional_special_tokens = getattr(self, key)
                    additional_special_tokens_set = set(additional_special_tokens)
                    to_add = []
                    for token in value:
                        if str(token) not in additional_special_tokens_set and str(token) not in to_add:
                            to_add.append(token)
                    # update the property
                    additional_special_tokens.extend(to_add)
                    self.additional_special_tokens = additional_special_tokens

                added_tokens += self.add_tokens(value, special_tokens=True)
            else:
                assert isinstance(
                    value, (str, AddedToken)
                ), f"Token {value} for key {key} should be a str or an AddedToken instance"
                setattr(self, key, value)
                added_tokens += self.add_tokens([value], special_tokens=True)

        return added_tokens

    def add_tokens(
            self, new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]], special_tokens: bool = False
    ) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary and and will be isolated before the tokenization
        algorithm is applied. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore
        not treated in the same way.

        Note, when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix
        of the model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

        Args:
            new_tokens (`str`, `tokenizers.AddedToken` or a list of *str* or `tokenizers.AddedToken`):
                Tokens are only added if they are not already in the vocabulary. `tokenizers.AddedToken` wraps a string
                token to let you personalize its behavior: whether this token should only match against a single word,
                whether this token should strip all potential whitespaces on the left side, whether this token should
                strip all potential whitespaces on the right side, etc.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Can be used to specify if the token is a special token. This mostly change the normalization behavior
                (special tokens like CLS or [MASK] are usually not lower-cased for instance).

                See details for `tokenizers.AddedToken` in HuggingFace tokenizers library.

        Returns:
            `int`: Number of tokens added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]

        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    @property
    def bos_token(self):
        """
        `str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        if self._bos_token is None:
            if self.verbose:
                logger.error("Using bos_token, but it is not set yet.")
            return None
        return str(self._bos_token)

    @property
    def eos_token(self):
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._eos_token is None:
            if self.verbose:
                logger.error("Using eos_token, but it is not set yet.")
            return None
        return str(self._eos_token)

    @property
    def unk_token(self):
        """
        `str`: Unknown token. Log an error if used while not having been set.
        """
        if self._unk_token is None:
            if self.verbose:
                logger.error("Using unk_token, but it is not set yet.")
            return None
        return str(self._unk_token)

    @property
    def sep_token(self):
        """
        `str`: Separation token, to separate context and query in an input sequence. Log an error if used while not
        having been set.
        """
        if self._sep_token is None:
            if self.verbose:
                logger.error("Using sep_token, but it is not set yet.")
            return None
        return str(self._sep_token)

    @property
    def pad_token(self):
        """
        `str`: Padding token. Log an error if used while not having been set.
        """
        if self._pad_token is None:
            if self.verbose:
                logger.error("Using pad_token, but it is not set yet.")
            return None
        return str(self._pad_token)

    @property
    def cls_token(self):
        """
        `str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the full
        depth of the model. Log an error if used while not having been set.
        """
        if self._cls_token is None:
            if self.verbose:
                logger.error("Using cls_token, but it is not set yet.")
            return None
        return str(self._cls_token)

    @property
    def mask_token(self):
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.
        """
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @property
    def additional_special_tokens(self):
        """
        `List[str]`: All the additional special tokens you may want to use. Log an error if used while not having been
        set.
        """
        if self._additional_special_tokens is None:
            if self.verbose:
                logger.error("Using additional_special_tokens, but it is not set yet.")
            return None
        return [str(tok) for tok in self._additional_special_tokens]

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
        been set.
        """
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the unknown token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
        sequence. Returns `None` if the token has not been set.
        """
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self) -> int:
        """
        `int`: Id of the padding token type in the vocabulary.
        """
        return self._pad_token_type_id

    @property
    def cls_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input sequence
        leveraging self-attention along the full depth of the model.

        Returns `None` if the token has not been set.
        """
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
        modeling. Returns `None` if the token has not been set.
        """
        if self._mask_token is None:
            return None
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self) -> List[int]:
        """
        `List[int]`: Ids of all the additional special tokens in the vocabulary. Log an error if used while not having
        been set.
        """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @bos_token_id.setter
    def bos_token_id(self, value):
        self._bos_token = self.convert_ids_to_tokens(value) if value is not None else None

    @eos_token_id.setter
    def eos_token_id(self, value):
        self._eos_token = self.convert_ids_to_tokens(value) if value is not None else None

    @unk_token_id.setter
    def unk_token_id(self, value):
        self._unk_token = self.convert_ids_to_tokens(value) if value is not None else None

    @sep_token_id.setter
    def sep_token_id(self, value):
        self._sep_token = self.convert_ids_to_tokens(value) if value is not None else None

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._pad_token = self.convert_ids_to_tokens(value) if value is not None else None

    @cls_token_id.setter
    def cls_token_id(self, value):
        self._cls_token = self.convert_ids_to_tokens(value) if value is not None else None

    @mask_token_id.setter
    def mask_token_id(self, value):
        self._mask_token = self.convert_ids_to_tokens(value) if value is not None else None

    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        self._additional_special_tokens = [self.convert_ids_to_tokens(value) for value in values]

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        `Dict[str, Union[str, List[str]]]`: A dictionary mapping special token class attributes (`cls_token`,
        `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Convert potential tokens of `tokenizers.AddedToken` type to string.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = (
                    type(attr_value)(str(attr_value_sub) for attr_value_sub in attr_value)
                    if isinstance(attr_value, (list, tuple))
                    else str(attr_value)
                )
        return set_attr

    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        """
        `Dict[str, Union[str, tokenizers.AddedToken, List[Union[str, tokenizers.AddedToken]]]]`: A dictionary mapping
        special token class attributes (`cls_token`, `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        """
        `List[Union[str, tokenizers.AddedToken]]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.) mapped to class
        attributes.

        Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
        all_toks = []
        set_attr = self.special_tokens_map_extended
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(OrderedDict.fromkeys(all_toks))
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        raise NotImplementedError

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return self.added_tokens_encoder

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return self.vocab_size + len(self.added_tokens_encoder)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.

        Args:
            new_tokens (`List[str]`or `List[tokenizers.AddedToken]`):
                Token(s) to add in vocabulary. A token is only added if it's not already in the vocabulary (tested by
                checking if the tokenizer assign the index of the `unk_token` to them).
            special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the tokens should be added as special tokens.

        Returns:
            `int`: The number of tokens actually added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        new_tokens = [str(tok) for tok in new_tokens]

        tokens_to_add = []
        for token in new_tokens:
            if not isinstance(token, str):
                raise TypeError(f"Token {token} is not a string but a {type(token)}.")
            if not special_tokens and hasattr(self, "do_lower_case") and self.do_lower_case:
                token = token.lower()
            if (
                    token != self.unk_token
                    and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                    and token not in tokens_to_add
            ):
                tokens_to_add.append(token)
                if self.verbose:
                    logger.info(f"Adding {token} to the vocabulary")

        added_tok_encoder = {tok: len(self) + i for i, tok in enumerate(tokens_to_add)}
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        # Make sure we don't split on any special tokens (even they were already in the vocab before e.g. for Albert)
        if special_tokens:
            if len(new_tokens) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, new_tokens[0])
            else:
                self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(new_tokens)))
        else:
            # Or on the newly added tokens
            if len(tokens_to_add) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, tokens_to_add[0])
            else:
                self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(tokens_to_add)))
        self._create_trie(self.unique_no_split_tokens)

        return len(tokens_to_add)

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            if hasattr(self, "do_lower_case") and self.do_lower_case and token not in self.all_special_tokens:
                trie.add(token.lower())
            else:
                trie.add(token)
        self.tokens_trie = trie

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """
        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        all_special_tokens_extended = {
            str(t): t for t in self.all_special_tokens_extended if isinstance(t, AddedToken)
        }

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok) for s_tok in (self.unique_no_split_tokens + self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        no_split_token = set(self.unique_no_split_tokens)
        tokens = self.tokens_trie.split(text)
        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = all_special_tokens_extended.get(token, None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                else:
                    # We strip left and right by default
                    if right:
                        tokens[i + 1] = right.lstrip()
                    if left:
                        tokens[i - 1] = left.rstrip()
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for
        word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces).
        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]):
        """Converts a token string (or a sequence of tokens) in a single integer id
        (or a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        if len(tokens) > 0 and isinstance(tokens[0], list):
            ids = []
            for ts in tokens:
                ids_x = []
                for token in ts:
                    ids_x.append(self._convert_token_to_id_with_added_voc(token))
                ids.append(ids_x)
            return ids

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    @staticmethod
    def convert_to_tensors(token_ids):
        return np.array(token_ids, dtype=np.int64)

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def encode(self, text):
        if isinstance(text, bytes):
            text = str(text, encoding="utf-8")
        if isinstance(text, str):
            tokens = self.tokenize(text)
            token_ids = self.convert_tokens_to_ids(tokens)
            token_ids = self.build_inputs_with_special_tokens(token_ids)
            token_ids = self.convert_to_tensors(token_ids)
            return token_ids
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
            tokens = [self.tokenize(t) for t in text]
            token_ids_list = self.convert_tokens_to_ids(tokens)
            token_ids_list = [
                self.build_inputs_with_special_tokens(token_ids) for token_ids in token_ids_list
            ]
            token_ids_list = self.convert_to_tensors(
                token_ids_list
            )
            return token_ids_list
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
            return text
        else:
            raise ValueError(
                "Input is not valid. Should be a string, a list/tuple of strings or "
                "a list/tuple of integers."
            )

    def prepare_for_tokenization(
            self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            text (`str`):
                The text to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments to use for the tokenization.

        Returns:
            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        return text, kwargs

    def get_special_tokens_mask(
            self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str:
        ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]:
        ...

    def convert_ids_to_tokens(
            self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens to a single string. The most simple way to do it is
        ``" ".join(tokens)`` but we often want to remove sub-word tokenization artifacts
        at the same time.

        Args:
            tokens (:obj:`List[str]`): The token to join in a string.

        Returns:
            :obj:`str`: The joined tokens.
        """
        return " ".join(tokens)

    def decode(
            self,
            token_ids,
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = None,
            spaces_between_special_tokens: bool = True,
    ):
        """
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids: list of tokenized input ids. Can be obtained using the `encode` or
            `encode_plus` methods.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
        """
        # Convert inputs to python lists
        token_ids = token_ids.tolist()
        filtered_tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    @property
    def eod_token(self):
        """
        :obj:`str`: End of document token. Log an error if used while not having been set.
        """
        if self._eod_token is None and self.verbose:
            logger.error("Using eod_token, but it is not set yet.")
            return None
        return str(self._eod_token)

    @property
    def start_token(self):
        """
        :obj:`str`: Start token of sentence. Common name for bos_token and cls_token.
        """
        if self._bos_token is not None and self._cls_token is not None:
            if self._bos_token == self._cls_token:
                return str(self._bos_token)
            else:
                logger.error("Conflict between bos_token and cls_token.")
                return None
        elif self._bos_token is None and self._cls_token is not None:
            return str(self._cls_token)
        elif self._bos_token is not None and self._cls_token is None:
            return str(self._bos_token)
        else:
            logger.error("Using start_token, but it is not set yet.")
            return None

    @property
    def end_token(self):
        """
        :obj:`str`: End token of sentence. Common name for eos_token and sep_token.
        Note: eod_token is not considered, because it is often same with eos_token.
        """
        if self._eos_token is not None and self._sep_token is not None:
            if self._eos_token == self._sep_token:
                return str(self._eos_token)
            else:
                logger.error("Conflict between eos_token and _sep_token.")
                return None
        elif self._eos_token is None and self._sep_token is not None:
            return str(self._sep_token)
        elif self._eos_token is not None and self._sep_token is None:
            return str(self._eos_token)
        else:
            logger.error("Using end_token, but it is not set yet.")
            return None

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @eod_token.setter
    def eod_token(self, value):
        self._eod_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def eod_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the end of document token in the vocabulary.
        Returns :obj:`None` if the token has not been set.
        """
        if self._eod_token is None:
            return None
        return self.convert_tokens_to_ids(self.eod_token)

    @property
    def start_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the start token in the vocabulary.
        Returns :obj:`None` if the token has not been set.
        """
        start_token = self.start_token
        if start_token is None:
            return None
        else:
            return self.convert_tokens_to_ids(start_token)

    @property
    def end_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the end token in the vocabulary.
        Returns :obj:`None` if the token has not been set.
        """
        end_token = self.end_token
        if end_token is None:
            return None
        else:
            return self.convert_tokens_to_ids(end_token)

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (`str`): The text to clean up.

        Returns:
            `str`: The cleaned-up string.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        This implementation does not add special tokens and this method should be overridden in a subclass.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
            list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GPT2Tokenizer(PreTrainedTokenizer):
    """
    Construct a GPT-2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:
    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            vocab_file="vocab.json",
            merges_file="merges.txt",
            errors="replace",
            unk_token="<|endoftext|>",
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            pad_token=None,
            add_prefix_space=False,
            add_bos_token=False,
            **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            **kwargs,
        )
        self.add_bos_token = add_bos_token
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.model_max_length = 1024

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def _tokenize(self, text, **kwargs):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return text, kwargs

    def _build_conversation_input_ids(self, conversation) -> List[int]:
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.encode(text) + [self.eos_token_id])
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length:]
        return input_ids


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer()
    z = tokenizer.encode("Hello world")
    x = tokenizer.decode(z)
    print(z, x)
