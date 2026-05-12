"""NanoChat-based Energy-Based Transformer (EBT) variant.

Subclasses :class:`nanochat.gpt.GPT` with EBT-specific additions such as
MCMC-step time embeddings and a scalar energy head.
"""

from nanochat.gpt import GPT, GPTConfig

class NanoChatEBT(GPT):
    """EBT variant built on top of the NanoChat GPT backbone.

    Adds a per-MCMC-step time embedding and a final linear head that projects
    the hidden state to a scalar energy value.
    """

    def __init__(self, config, pad_vocab_size_to=64):
        """Construct the backbone and the EBT-specific modules.

        :param config: GPT configuration object.
        :type config: nanochat.gpt.GPTConfig
        :param pad_vocab_size_to: Pad the vocabulary size up to a multiple of
            this value for efficient matrix multiplication.
        :type pad_vocab_size_to: int
        """
        super().__init__(config, pad_vocab_size_to)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.config.n_layer):
            self.layers.append(TransformerBlock(layer_id, params))

        if params.ebt_norm == "rms":
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        elif params.ebt_norm == "layer":
            self.norm = nn.LayerNorm(params.dim)
        elif params.ebt_norm == "none":
            self.norm = nn.Identity()
        elif params.ebt_norm == "dyt":
            # NOTE: Disable the learnable bias term because the final bias
            # cannot receive gradients through the EBT energy objective.
            self.norm = DyT(params.dim, alpha_init_value=params.dyt_alpha_init, bias_learnable = False)
        elif params.ebt_norm == "ebm_backwards_norm":
            self.norm = EBMBackwardsRMSNorm(params.dim, eps=params.norm_eps)
        else:
            raise ValueError(f"Invalid ebt_norm value: {params.ebt_norm}")

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.config.n_head, self.config.sequence_len
        )

        self.time_embeddings = nn.Embedding(max_mcmc_steps, params.dim)

        self.final_layer = nn.Linear(params.dim, 1, bias = False)
        init_whole_model_weights(self.final_layer, self.params.weight_initialization)

    def forward(self, embeddings: torch.Tensor, start_pos: int, mcmc_step = 0):
        """Run a forward pass through the EBT transformer.

        :param embeddings: Input embeddings of shape ``(B, S, D)``. Embeddings
            are used instead of raw token ids so that the same module can
            accept outputs from upstream modalities.
        :type embeddings: torch.Tensor
        :param start_pos: Starting position used for attention key/value caching.
        :type start_pos: int
        :param mcmc_step: Current MCMC refinement step. Used to look up the
            time embedding that is prepended to the input sequence.
        :type mcmc_step: int
        :return: Per-position energies predicted by the model.
        :rtype: torch.Tensor
        """
        _bsz = embeddings.shape[0]
        mcmc_step = torch.full(size=(_bsz,), fill_value=mcmc_step, device = embeddings.device, dtype=torch.long)
        # Broadcast the (B,) step index to a (B, 1, D) time embedding that is
        # prepended to the sequence before self-attention.
        time_embeddings = self.time_embeddings(mcmc_step).unsqueeze(dim=1)
        embeddings = torch.cat((time_embeddings, embeddings), dim = 1) # (B, 2S-1+1, D)

        _bsz, seqlen = embeddings.shape[:2]
        # Recover the original sequence length: the incoming seqlen is
        # ``2*(S-1)+1`` after concatenating the prepended time embedding, so
        # ``(seqlen+3)//2`` yields ``S+1``.
        seqlen = (seqlen+3) // 2
        self.freqs_cis = self.freqs_cis.to(embeddings.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=embeddings.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching we only score the new portion
            # of the sequence; the scores matrix therefore has shape
            # ``(seqlen, cache_len + seqlen)`` and only entries ``(i, j)`` with
            # ``j > cache_len + i`` need masking.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=embeddings.device),
                mask
            ]).type_as(embeddings)
            # Resulting causal mask layout (upper-triangular minus-inf):
            #     0, -inf, -inf
            #     0,    0, -inf
            #     0,    0,    0

            for i, layer in enumerate(self.transformer):
                embeddings = layer(embeddings, start_pos, freqs_cis, mask)
            embeddings = self.norm(embeddings)
            embeddings = embeddings[:, 1:] # drop the prepended time embedding
            energies = self.final_layer(embeddings)

            energies = energies[:, embeddings.shape[1] // 2:]
            return energies