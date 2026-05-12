"""Adapter that exposes NanoChat's ``RustBPETokenizer`` through an HF-like API.

EBT code expects a HuggingFace-style tokenizer interface. This adapter lets
EBT run with the NanoChat tokenizer (``vocab_size == 32768``) used to train
the Nova checkpoints, without forking the training code.
"""
import sys
import os
from typing import Any, Dict, List, Optional, Sequence, Union

# CRITICAL: Clean up dummy ``nanochat`` modules created by
# ``train_model.py``'s checkpoint loader. That loader stubs ``nanochat.*``
# modules in :data:`sys.modules` to allow cross-repo checkpoint loading, but
# those stubs do not expose the real ``RustBPETokenizer.from_directory``.
if 'nanochat.tokenizer' in sys.modules:
    mod = sys.modules['nanochat.tokenizer']
    # Detect a dummy module created by ``create_module_shim`` in
    # ``train_model.py``.
    if hasattr(mod, 'RustBPETokenizer'):
        cls = mod.RustBPETokenizer
        # The real ``RustBPETokenizer`` exposes ``from_directory`` as a
        # classmethod; the dummy subclass does not (or the attribute is not
        # callable).
        has_from_directory = hasattr(cls, 'from_directory') and callable(getattr(cls, 'from_directory', None))
        if not has_from_directory:
            print("[TokenizerAdapter] Detected dummy nanochat module from checkpoint loader")
            print("[TokenizerAdapter] Cleaning up sys.modules to load real nanochat tokenizer...")
            to_remove = [k for k in sys.modules.keys() if k.startswith('nanochat')]
            for k in to_remove:
                del sys.modules[k]
            print(f"[TokenizerAdapter] Removed {len(to_remove)} dummy modules: {to_remove}")

# Add real nanochat path
nanochat_path = "/mnt/shared-storage-user/puyuan/code/nanochat"
if nanochat_path not in sys.path:
    sys.path.insert(0, nanochat_path)

# Now import the real RustBPETokenizer
from nanochat.tokenizer import RustBPETokenizer

# Verify we got the real one
if not hasattr(RustBPETokenizer, 'from_directory'):
    raise ImportError(
        "Failed to import real RustBPETokenizer. "
        "The imported class doesn't have 'from_directory' method. "
        "This likely means the dummy module is still being used."
    )


class NanoChatTokenizerWrapper:
    """Wrap NanoChat's ``RustBPETokenizer`` in a HuggingFace-style interface.

    The wrapper exposes ``__call__``, ``encode``, ``decode`` and
    ``batch_decode`` so that existing EBT code can treat it as a drop-in
    HuggingFace tokenizer.
    """

    def __init__(
        self,
        tokenizer_obj: Any = None,
        tokenizer_dir: str = "/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer",
    ) -> None:
        """Initialize the wrapper.

        :param tokenizer_obj: Optional existing ``RustBPETokenizer`` instance.
            When provided, ``tokenizer_dir`` is ignored.
        :type tokenizer_obj: Any
        :param tokenizer_dir: Directory to load the tokenizer from when
            ``tokenizer_obj`` is ``None``.
        :type tokenizer_dir: str
        """
        if tokenizer_obj is not None:
            self.tokenizer = tokenizer_obj
        else:
            self.tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

        # NanoChat conflates BOS with EOS/PAD for compatibility; the HF API
        # requires all three attributes to be defined.
        self.eos_token_id = self.tokenizer.get_bos_token_id()
        self.bos_token_id = self.tokenizer.get_bos_token_id()
        self.pad_token_id = self.eos_token_id
        self.unk_token_id = 0

        print(f"[NanoChatTokenizerWrapper] Vocab size: {self.tokenizer.get_vocab_size()}")
        print(f"[NanoChatTokenizerWrapper] EOS/BOS/PAD token ID: {self.eos_token_id}")

    def __len__(self) -> int:
        """Return the tokenizer's vocabulary size.

        :return: Vocabulary size.
        :rtype: int
        """
        return self.tokenizer.get_vocab_size()

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        return_tensors: Optional[str] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """HuggingFace-style tokenization call.

        :param text: String or list of strings to encode.
        :type text: Union[str, Sequence[str]]
        :param return_tensors: ``"pt"`` to return PyTorch tensors, ``None``
            for plain Python lists.
        :type return_tensors: Optional[str]
        :param padding: When truthy, pads sequences to ``max_length`` or to
            the longest sample in the batch.
        :type padding: Union[bool, str]
        :param truncation: When ``True`` and ``max_length`` is given,
            truncates sequences to ``max_length`` tokens.
        :type truncation: bool
        :param max_length: Maximum sequence length.
        :type max_length: Optional[int]
        :return: Dict with keys ``"input_ids"`` and ``"attention_mask"``.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``text`` is not a string or sequence of strings.
        """
        import torch

        if isinstance(text, str):
            ids = self.tokenizer.encode(text)
            ids_list = [ids]
        elif isinstance(text, (list, tuple)):
            ids_list = [self.tokenizer.encode(t) for t in text]
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")

        if truncation and max_length is not None:
            ids_list = [ids[:max_length] for ids in ids_list]

        if padding:
            if max_length is not None:
                target_length = max_length
            else:
                target_length = max(len(ids) for ids in ids_list)

            padded_ids = []
            attention_masks = []
            for ids in ids_list:
                seq_len = len(ids)
                if seq_len < target_length:
                    padded = ids + [self.pad_token_id] * (target_length - seq_len)
                    mask = [1] * seq_len + [0] * (target_length - seq_len)
                else:
                    padded = ids
                    mask = [1] * len(ids)
                padded_ids.append(padded)
                attention_masks.append(mask)
        else:
            padded_ids = ids_list
            attention_masks = [[1] * len(ids) for ids in ids_list]

        if return_tensors == 'pt':
            import torch
            input_ids = torch.tensor(padded_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        else:
            input_ids = padded_ids
            attention_mask = attention_masks

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def encode(self, text: Union[str, Sequence[str]], add_special_tokens: bool = False, **kwargs: Any) -> Union[List[int], List[List[int]]]:
        """Encode text to token ids.

        :param text: String or sequence of strings.
        :type text: Union[str, Sequence[str]]
        :param add_special_tokens: Accepted for HF compatibility but currently
            ignored by the underlying tokenizer.
        :type add_special_tokens: bool
        :return: List of token ids (or list of lists for batch input).
        :rtype: Union[List[int], List[List[int]]]
        :raises ValueError: If ``text`` is not a string or sequence of strings.
        """
        if isinstance(text, str):
            return self.tokenizer.encode(text)
        elif isinstance(text, (list, tuple)):
            return [self.tokenizer.encode(t) for t in text]
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")

    def decode(self, token_ids: Any, skip_special_tokens: bool = False, **kwargs: Any) -> str:
        """Decode a sequence of token ids to text.

        :param token_ids: List or tensor of token ids.
        :type token_ids: Any
        :param skip_special_tokens: When ``True``, drops BOS/EOS/PAD and the
            NanoChat chat-template tokens (``<|user_start|>`` etc.) before
            decoding.
        :type skip_special_tokens: bool
        :return: Decoded text.
        :rtype: str
        """
        import torch
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            # Filter every NanoChat special token (BOS/EOS/PAD plus the chat
            # template markers).
            special_token_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id}
            for name in ["<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
                         "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]:
                tid = self.tokenizer.encode_special(name) if hasattr(self.tokenizer, 'encode_special') else None
                if tid is not None:
                    special_token_ids.add(tid)
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]

        return self.tokenizer.decode(token_ids)

    def batch_decode(self, sequences: Any, skip_special_tokens: bool = False, **kwargs: Any) -> List[str]:
        """Batch variant of :meth:`decode`.

        :param sequences: Batched token ids (list of lists or 2-D tensor).
        :type sequences: Any
        :param skip_special_tokens: See :meth:`decode`.
        :type skip_special_tokens: bool
        :return: Decoded strings, one per sequence.
        :rtype: List[str]
        """
        import torch
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        result = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()

            if skip_special_tokens:
                special_token_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id}
                for name in ["<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
                             "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]:
                    tid = self.tokenizer.encode_special(name) if hasattr(self.tokenizer, 'encode_special') else None
                    if tid is not None:
                        special_token_ids.add(tid)
                seq = [tid for tid in seq if tid not in special_token_ids]

            result.append(self.tokenizer.decode(seq))
        return result


def get_nanochat_tokenizer() -> NanoChatTokenizerWrapper:
    """Build a :class:`NanoChatTokenizerWrapper` using the default tokenizer directory.

    :return: A wrapper around the NanoChat ``RustBPETokenizer``.
    :rtype: NanoChatTokenizerWrapper
    """
    return NanoChatTokenizerWrapper()


if __name__ == "__main__":
    tokenizer = get_nanochat_tokenizer()

    text = "Hello, world!"
    ids = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Token IDs: {ids}")
    print(f"Decoded: {tokenizer.decode(ids)}")

    result = tokenizer([text, "Another text"], return_tensors='pt', padding=True, max_length=20)
    print(f"\nHF-style result:")
    print(f"input_ids shape: {result['input_ids'].shape}")
    print(f"input_ids:\n{result['input_ids']}")
    print(f"attention_mask:\n{result['attention_mask']}")
