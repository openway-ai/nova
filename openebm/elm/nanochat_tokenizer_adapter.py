"""
Adapter to use nanochat's RustBPETokenizer in EBT with HuggingFace-compatible interface.
This allows EBT to use the correct tokenizer (vocab_size=32768) that was used to train Nova checkpoints.
"""
import sys
import os

# CRITICAL FIX: Clean up dummy nanochat modules created by train_model.py's checkpoint loader
# The checkpoint loader creates dummy modules in sys.modules to handle cross-repo checkpoints,
# but these dummy modules don't have the real RustBPETokenizer.from_directory method.
if 'nanochat.tokenizer' in sys.modules:
    mod = sys.modules['nanochat.tokenizer']
    # Check if this is a dummy module (created by create_module_shim in train_model.py)
    if hasattr(mod, 'RustBPETokenizer'):
        cls = mod.RustBPETokenizer
        # Real RustBPETokenizer has from_directory as a classmethod
        # Dummy RustBPETokenizer (DummyBase subclass) doesn't have it or it's not callable
        has_from_directory = hasattr(cls, 'from_directory') and callable(getattr(cls, 'from_directory', None))
        if not has_from_directory:
            print("[TokenizerAdapter] Detected dummy nanochat module from checkpoint loader")
            print("[TokenizerAdapter] Cleaning up sys.modules to load real nanochat tokenizer...")
            # Remove all nanochat-related dummy modules
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
    """
    Wrapper around nanochat's RustBPETokenizer to provide HuggingFace-compatible interface.
    This allows it to be used as a drop-in replacement in EBT code.
    """

    def __init__(self, tokenizer_obj=None, tokenizer_dir="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer"):
        """
        Initialize wrapper with either an existing tokenizer object or by loading from directory.

        Args:
            tokenizer_obj: Optional existing RustBPETokenizer instance
            tokenizer_dir: Directory to load tokenizer from if tokenizer_obj is None
        """
        if tokenizer_obj is not None:
            # Use provided tokenizer object
            self.tokenizer = tokenizer_obj
        else:
            # Load from directory
            self.tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

        self.eos_token_id = self.tokenizer.get_bos_token_id()  # Use BOS as EOS for compatibility
        self.bos_token_id = self.tokenizer.get_bos_token_id()
        self.pad_token_id = self.eos_token_id
        self.unk_token_id = 0  # Fallback

        print(f"[NanoChatTokenizerWrapper] Vocab size: {self.tokenizer.get_vocab_size()}")
        print(f"[NanoChatTokenizerWrapper] EOS/BOS/PAD token ID: {self.eos_token_id}")

    def __len__(self):
        """Return vocab size"""
        return self.tokenizer.get_vocab_size()

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None, **kwargs):
        """
        HuggingFace-style __call__ interface for tokenization.

        Args:
            text: str or list of str
            return_tensors: 'pt' for PyTorch tensors, None for lists
            padding: bool or str, whether to pad sequences
            truncation: bool, whether to truncate sequences
            max_length: int, maximum sequence length

        Returns:
            dict with 'input_ids' and 'attention_mask'
        """
        import torch

        # Encode the text
        if isinstance(text, str):
            ids = self.tokenizer.encode(text)
            ids_list = [ids]
        elif isinstance(text, (list, tuple)):
            ids_list = [self.tokenizer.encode(t) for t in text]
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")

        # Truncate if needed
        if truncation and max_length is not None:
            ids_list = [ids[:max_length] for ids in ids_list]

        # Padding
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
                    # Pad with pad_token_id
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

        # Convert to tensors if requested
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

    def encode(self, text, add_special_tokens=False, **kwargs):
        """Encode text to token IDs"""
        if isinstance(text, str):
            return self.tokenizer.encode(text)
        elif isinstance(text, (list, tuple)):
            return [self.tokenizer.encode(t) for t in text]
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """Decode token IDs to text"""
        import torch
        # Convert tensor to list if needed (nanochat tokenizer expects list/sequence)
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Filter out special tokens if requested
        if skip_special_tokens:
            # 过滤所有 nanochat special tokens（bos/eos/pad 以及 chat 模板 token）
            special_token_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id}
            # 动态加入所有 chat 模板 special token（<|user_start|> 等）
            for name in ["<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
                         "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]:
                tid = self.tokenizer.encode_special(name) if hasattr(self.tokenizer, 'encode_special') else None
                if tid is not None:
                    special_token_ids.add(tid)
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]

        return self.tokenizer.decode(token_ids)

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        """Batch decode token IDs to text"""
        import torch
        # Convert tensors to lists if needed
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        result = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()

            # Filter out special tokens if requested
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


def get_nanochat_tokenizer():
    """
    Convenience function to get the nanochat tokenizer with HuggingFace-compatible interface.

    Returns:
        NanoChatTokenizerWrapper instance
    """
    return NanoChatTokenizerWrapper()


# For testing
if __name__ == "__main__":
    tokenizer = get_nanochat_tokenizer()

    # Test basic encoding
    text = "Hello, world!"
    ids = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Token IDs: {ids}")
    print(f"Decoded: {tokenizer.decode(ids)}")

    # Test HuggingFace-style interface
    result = tokenizer([text, "Another text"], return_tensors='pt', padding=True, max_length=20)
    print(f"\nHF-style result:")
    print(f"input_ids shape: {result['input_ids'].shape}")
    print(f"input_ids:\n{result['input_ids']}")
    print(f"attention_mask:\n{result['attention_mask']}")
