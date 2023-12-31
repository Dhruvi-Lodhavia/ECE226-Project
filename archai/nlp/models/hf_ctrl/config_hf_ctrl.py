# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Transformer-XL configurations.
"""

from typing import List, Optional

from archai.nlp.models.config_base import Config, SearchConfigParameter, SearchConfig
from transformers import CONFIG_MAPPING


class HfCTRLConfig(Config):
    """Huggingface's Transformer-XL default configuration.

    """

    attribute_map = {
        'n_token': 'vocab_size',
        'tgt_len': 'n_positions',
        'd_model': 'n_embd',
        'd_inner': 'dff',
        'dropout': 'resid_pdrop',
        'weight_init_std': 'initializer_range'
    }
    attribute_map.update(CONFIG_MAPPING['ctrl']().attribute_map)

    def __init__(self,
                 n_token: Optional[int] = 246534,
                 tgt_len: Optional[int] = 256,
                 d_model: Optional[int] = 512,
                 d_inner: Optional[int] = 2048,
                 d_head: Optional[int] = 16,
                 d_embed: Optional[int] = 1280,
                 n_layer: Optional[int] = 48,
                 n_head: Optional[int] = 16,
                 dropout: Optional[float] = 0.1,
                 dropatt: Optional[float] = 0.0,
                 weight_init_std: Optional[float] = 0.02,
                 embd_pdrop: Optional[float] = 0.1,
                 tie_weight: Optional[bool] = True,
                 **kwargs) -> None:
        """Initializes the class by overriding default arguments.

        Args:
            n_token: Size of the vocabulary (number of tokens).
            tgt_len: Maximum length of sequences (positional embeddings).
            d_model: Dimensionality of the model.
            d_inner: Dimensionality of inner feed-forward layers.
            d_head: Dimensionality of attention heads (`0` for using `d_model` // `n_head`)
            d_embed: Dimensionality of embedding layer (`0` for using same as `d_model`)
            n_layer: Number of layers.
            n_head: Number of attention heads.
            dropout: Dropout probability.
            dropatt: Attention dropout probability.
            div_val: Adaptive embedding/softmax divident.
            pre_lnorm: Whether layer normalization should be performed to input instead of output.
            cutoffs: Cutoffs values for adaptive embedding/softmax.
            mem_len: Maximum length of the memory.
            same_length: Whether every incoming sample should use the same attention length.
            attn_type: Type of attention mechanism (`0` for default attention).
            clamp_len: Uses the same positional embeddings after clamp_len (`0` for no clamp).
            sample_softmax: Number of samples in the sampled softmax (`-1` for disabling).
            adaptive: Whether to use adaptive softmax.
            weight_init_type: Type of weight initialization (`normal` for default).
            weight_init_range: Range to initialize the weights.
            weight_init_std: Standard deviation to initialize the weights.
            proj_init_std: Standard deviation to initialize the projections.
            tie_weight: Whether embedding and softmax weights should be tied.

        """

        self.n_token = n_token
        self.tgt_len = tgt_len
        self.d_model = d_model
        self.d_inner = d_inner
        # self.d_inner = self._map_to_list(self.d_inner,n_layer)
        self.d_head = d_head if d_head > 0 else d_model // n_head
        self.d_embed = d_embed if d_embed > 0 else d_model
        self.n_layer = n_layer
        self.n_head = n_head
        # self.n_head = self._map_to_list(self.n_head,n_layer)
        self.dropout = dropout
        self.dropatt = dropatt
        self.embd_pdrop = embd_pdrop
        self.weight_init_std = weight_init_std
        # self.dff = dff
        self.tie_weight = tie_weight

        additional_config = CONFIG_MAPPING['ctrl']().to_dict()
        for key, value in additional_config.items():
            if key not in self.__dict__.keys():
                setattr(self, key, value)

        super().__init__(**kwargs)
        


class HfCTRLSearchConfig(SearchConfig):
    """Huggingface's Transformer-XL search configuration.

    """

    def __init__(self) -> None:
        """Initializes the class by setting default parameters that are used during search.
        
        """
        
        # Default HfTransfoXL search options: n_layer, d_model, d_inner and n_head
        n_layer = SearchConfigParameter(per_layer=False, value=[3, 4, 5, 6, 7, 8, 9, 10])
        d_model = SearchConfigParameter(per_layer=False, value=list(range(128, 1024, 64)))
        d_inner = SearchConfigParameter(per_layer=False, value=list(range(128, 4096, 64)))
        n_head = SearchConfigParameter(per_layer=False, value=[2, 4, 8])

        super().__init__(n_layer=n_layer,
                         d_model=d_model,
                         d_inner=d_inner,
                         n_head=n_head)


